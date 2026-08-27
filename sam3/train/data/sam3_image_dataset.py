# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Dataset class for modulated detection"""

import io
import json
import logging
import mmap
import os
import random
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.data
import torchvision
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from PIL.Image import DecompressionBombError
from sam3.model.box_ops import box_xywh_to_xyxy
from torchvision.datasets.vision import VisionDataset

from .coco_json_loaders import COCO_FROM_JSON


@dataclass
class InferenceMetadata:
    """Metadata required for postprocessing"""

    # Coco id that corresponds to the "image" for evaluation by the coco evaluator
    # This is used for our own "class agnostic" evaluation
    coco_image_id: int

    # id in the original dataset, such that we can use the original evaluator
    original_image_id: int

    # Original category id (if we want to use the original evaluator)
    original_category_id: int

    # Size of the raw image (height, width)
    original_size: Tuple[int, int]

    # Id of the object in the media
    object_id: int

    # Index of the frame in the media (0 if single image)
    frame_index: int

    # Whether it is for conditioning only, e.g., 0-th frame in TA is for conditioning
    # as we assume GT available in frame 0.
    is_conditioning_only: Optional[bool] = False


@dataclass
class FindQuery:
    query_text: str

    image_id: int

    # In case of a find query, the list of object ids that have to be predicted
    object_ids_output: List[int]

    # This is "instance exhaustivity".
    # true iff all instances are separable and annotated
    # See below the slightly different "pixel exhaustivity"
    is_exhaustive: bool

    # The order in which the queries are processed (only meaningful for video)
    query_processing_order: int = 0

    # Input geometry, initially in denormalized XYXY format. Then
    # 1. converted to normalized CxCyWH by the Normalize transform
    input_bbox: Optional[torch.Tensor] = None
    input_bbox_label: Optional[torch.Tensor] = None

    # Only for the PVS task
    input_points: Optional[torch.Tensor] = None

    semantic_target: Optional[torch.Tensor] = None

    # pixel exhaustivity: true iff the union of all segments (including crowds)
    # covers every pixel belonging to the target class
    # Note that instance_exhaustive implies pixel_exhaustive
    is_pixel_exhaustive: Optional[bool] = None


@dataclass
class FindQueryLoaded(FindQuery):
    # Must have default value since FindQuery has entries with default values
    inference_metadata: Optional[InferenceMetadata] = None


@dataclass
class Object:
    # Initially in denormalized XYXY format, gets converted to normalized CxCyWH by the Normalize transform
    bbox: torch.Tensor
    area: float

    # Id of the object in the media
    object_id: Optional[int] = -1

    # Index of the frame in the media (0 if single image)
    frame_index: Optional[int] = -1

    segment: Optional[Union[torch.Tensor, dict]] = None  # RLE dict or binary mask

    is_crowd: bool = False

    source: Optional[str] = None


@dataclass
class Image:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]
    size: Tuple[int, int]  # (height, width)

    # For blurring augmentation
    blurring_mask: Optional[Dict[str, Any]] = None


@dataclass
class Datapoint:
    """Refers to an image/video and all its annotations"""

    find_queries: List[FindQueryLoaded]
    images: List[Image]
    raw_images: Optional[List[PILImage.Image]] = None


class CustomCocoDetectionAPI(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        load_segmentation: bool,
        fix_fname: bool = False,
        training: bool = True,
        blurring_masks_path: Optional[str] = None,
        use_caching: bool = True,
        zstd_dict_path=None,
        filter_query=None,
        coco_json_loader: Callable = COCO_FROM_JSON,
        # pyrefly: ignore [bad-function-definition]
        limit_ids: int = None,
        limit_ratio: float = 1.0,
        cache_images: Union[bool, str] = "none",
    ) -> None:
        super().__init__(root)

        self.annFile = annFile
        self.use_caching = use_caching
        self.zstd_dict_path = zstd_dict_path

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.load_segmentation = load_segmentation
        self.fix_fname = fix_fname
        self.filter_query = filter_query

        self.coco = None
        self.coco_json_loader = coco_json_loader
        self.limit_ids = limit_ids
        self.limit_ratio = limit_ratio
        self.set_sharded_annotation_file(0)
        self.training = training
        self.blurring_masks_path = blurring_masks_path

        # cache_images 三档 (兼容 bool: True→"ram", False→"none"):
        #   "none": 不缓存, 每样本从磁盘读 (默认, 行为同上游)
        #   "ram" : init 时把全部图片字节读进内存 (压缩格式不解码), 之后从 RAM 解码;
        #           必须在 DataLoader fork worker 之前建好, worker 以 COW 共享
        #   "disk": 把全部图片打包成注解旁的单个大文件 (顺序写, 只需构建一次),
        #           之后 mmap 读; 多进程/多任务经 OS page cache 共享, 不占额外内存
        self.cache_mode = self._normalize_cache_mode(cache_images)
        self._image_cache: Optional[Dict[str, bytes]] = None
        self._pack_index: Optional[Dict[str, Tuple[int, int]]] = None
        self._pack_mmap = None
        if self.cache_mode == "ram":
            self._image_cache = self._build_image_cache()
        elif self.cache_mode == "disk":
            self._pack_index = self._load_or_build_image_pack()

    def _load_images(
        self, datapoint_id: int, img_ids_to_load: Optional[Set[int]] = None
    ) -> Tuple[List[Tuple[int, PILImage.Image]], List[Dict[str, Any]]]:
        all_images = []
        all_img_metadata = []
        # pyrefly: ignore [missing-attribute]
        for current_meta in self.coco.loadImagesFromDatapoint(datapoint_id):
            img_id = current_meta["id"]
            if img_ids_to_load is not None and img_id not in img_ids_to_load:
                continue
            if self.fix_fname:
                current_meta["file_name"] = current_meta["file_name"].split("/")[-1]
            path = current_meta["file_name"]
            if self.blurring_masks_path is not None:
                mask_fname = os.path.basename(path).replace(".jpg", "-mask.json")
                mask_path = os.path.join(self.blurring_masks_path, mask_fname)
                if os.path.exists(mask_path):
                    with open(mask_path, "r") as fopen:
                        current_meta["blurring_mask"] = json.load(fopen)

            all_img_metadata.append(current_meta)
            path = os.path.join(self.root, path)
            try:
                if ".mp4" in path and path[-4:] == ".mp4":
                    # Going to load a video frame
                    from decord import cpu, VideoReader

                    video_path, frame = path.split("@")
                    video = VideoReader(video_path, ctx=cpu(0))
                    # Convert to PIL image
                    all_images.append(
                        (
                            img_id,
                            torchvision.transforms.ToPILImage()(
                                video[int(frame)].asnumpy()
                            ),
                        )
                    )
                else:
                    cached = self._read_cached(current_meta["file_name"])
                    if cached is not None:
                        all_images.append(
                            (
                                img_id,
                                PILImage.open(io.BytesIO(cached)).convert("RGB"),
                            )
                        )
                    else:
                        with g_pathmgr.open(path, "rb") as fopen:
                            all_images.append(
                                (img_id, PILImage.open(fopen).convert("RGB"))
                            )
            except FileNotFoundError as e:
                print(f"File not found: {path} from dataset: {self.annFile}")
                raise e

        return all_images, all_img_metadata

    @staticmethod
    def _normalize_cache_mode(cache_images) -> str:
        if isinstance(cache_images, str):
            mode = cache_images.lower()
            assert mode in ("none", "ram", "disk"), (
                f"cache_images 只支持 none/ram/disk (得到 {cache_images})"
            )
            return mode
        return "ram" if cache_images else "none"

    def _cached_file_names(self) -> List[str]:
        """self.ids 覆盖到的图片相对文件名 (已按 fix_fname 处理, 排序去重)。

        缓存键用文件名而不是图片 id: COCO_FROM_JSON 返回的图片 id 恒为 0。
        视频帧 (.mp4@frame) 不缓存, 仍走 decord 实时解码。
        """
        file_names = set()
        for dp_id in self.ids.tolist():
            for meta in self.coco.loadImagesFromDatapoint(dp_id):
                fname = meta["file_name"]
                if self.fix_fname:
                    fname = fname.split("/")[-1]
                if ".mp4" in fname:
                    continue
                file_names.add(fname)
        return sorted(file_names)

    def _read_file(self, fname: str) -> bytes:
        with g_pathmgr.open(os.path.join(self.root, fname), "rb") as f:
            return f.read()

    def _read_cached(self, fname: str) -> Optional[bytes]:
        if self._image_cache is not None:
            return self._image_cache.get(fname)
        if self._pack_index is not None and self._pack_mmap is not None:
            span = self._pack_index.get(fname)
            if span is not None:
                return self._pack_mmap[span[0] : span[0] + span[1]]
        return None

    def _build_image_cache(self) -> Dict[str, bytes]:
        """把 self.ids 覆盖到的所有图片文件读入内存 (保留压缩字节)。"""
        file_names = self._cached_file_names()
        t0 = time.time()
        cache: Dict[str, bytes] = {}
        # 多线程掩盖小文件 open/seek 开销; 读出来就是顺序字节流
        with ThreadPoolExecutor(max_workers=16) as pool:
            for fname, buf in pool.map(lambda fn: (fn, self._read_file(fn)), file_names):
                cache[fname] = buf
        total_mb = sum(len(b) for b in cache.values()) / 1024 / 1024
        logging.info(
            f"[cache_images] {self.annFile}: cached {len(cache)} images "
            f"({total_mb:.1f} MB compressed) in {time.time() - t0:.1f}s"
        )
        return cache

    def _load_or_build_image_pack(self) -> Optional[Dict[str, Tuple[int, int]]]:
        """加载 (必要时构建) 注解旁的图片打包文件, 返回 {文件名: (偏移, 长度)}。

        打包文件: ``<annFile>.imgcache.bin`` (图片字节顺序拼接) +
        ``<annFile>.imgcache.json`` (索引)。文件名集合或总字节数对不上即重建。
        """
        bin_path = self.annFile + ".imgcache.bin"
        idx_path = self.annFile + ".imgcache.json"
        file_names = self._cached_file_names()

        index = None
        try:
            with open(idx_path, "r") as f:
                pack_meta = json.load(f)
            if sorted(pack_meta["files"].keys()) == file_names and os.path.getsize(
                bin_path
            ) == pack_meta["total_bytes"]:
                index = {k: tuple(v) for k, v in pack_meta["files"].items()}
                logging.info(
                    f"[cache_images] reuse pack {bin_path} "
                    f"({len(index)} images, {pack_meta['total_bytes'] / 1024 / 1024:.1f} MB)"
                )
        except (OSError, ValueError, KeyError):
            index = None

        if index is None:
            index = self._build_image_pack(bin_path, idx_path, file_names)
        if index is None:
            return None

        fd = os.open(bin_path, os.O_RDONLY)
        self._pack_mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        os.close(fd)  # 映射已建立, fd 可关
        return index

    def _build_image_pack(
        self, bin_path: str, idx_path: str, file_names: List[str]
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """多线程读小文件 + 顺序写单个大文件; tmp+rename 保证并发任务落盘的是完整文件。"""
        t0 = time.time()
        pid = os.getpid()
        tmp_bin = f"{bin_path}.tmp{pid}"
        tmp_idx = f"{idx_path}.tmp{pid}"
        try:
            offset = 0
            index: Dict[str, Tuple[int, int]] = {}
            with ThreadPoolExecutor(max_workers=16) as pool, open(tmp_bin, "wb") as out:
                for fname, buf in pool.map(
                    lambda fn: (fn, self._read_file(fn)), file_names
                ):
                    index[fname] = (offset, len(buf))
                    out.write(buf)
                    offset += len(buf)
            with open(tmp_idx, "w") as f:
                json.dump({"total_bytes": offset, "files": index}, f)
            os.replace(tmp_bin, bin_path)
            os.replace(tmp_idx, idx_path)
        except OSError as e:
            # 如数据集目录只读: 清理临时文件, 回退为不缓存
            logging.warning(
                f"[cache_images] 打包文件构建失败 ({e}), 回退为不缓存"
            )
            for p in (tmp_bin, tmp_idx):
                if os.path.exists(p):
                    os.remove(p)
            return None
        logging.info(
            f"[cache_images] packed {len(index)} images "
            f"({offset / 1024 / 1024:.1f} MB) -> {bin_path} "
            f"in {time.time() - t0:.1f}s"
        )
        return index

    def set_curr_epoch(self, epoch: int):
        self.curr_epoch = epoch

    def set_epoch(self, epoch: int):
        pass

    def set_sharded_annotation_file(self, data_epoch: int):
        if self.coco is not None:
            return

        assert g_pathmgr.isfile(self.annFile), (
            f"please provide valid annotation file. Missing: {self.annFile}"
        )
        annFile = g_pathmgr.get_local_path(self.annFile)

        if self.coco is not None:
            del self.coco

        self.coco = self.coco_json_loader(annFile)
        # Use a torch tensor here to optimize memory usage when using several dataloaders
        ids_list = list(sorted(self.coco.getDatapointIds()))
        if self.limit_ratio is not None and self.limit_ratio < 1.0:
            local_random = random.Random(len(ids_list))
            local_random.shuffle(ids_list)
            ids_list = ids_list[: int(len(ids_list) * self.limit_ratio)]
        if self.limit_ids is not None:
            local_random = random.Random(len(ids_list))
            local_random.shuffle(ids_list)
            ids_list = ids_list[: self.limit_ids]
        self.ids = torch.as_tensor(ids_list, dtype=torch.long)

    def __getitem__(self, index: int) -> Datapoint:
        return self._load_datapoint(index)

    def _load_datapoint(self, index: int) -> Datapoint:
        """A separate method for easy overriding in subclasses."""
        id = self.ids[index].item()
        # pyrefly: ignore [bad-argument-type]
        pil_images, img_metadata = self._load_images(id)
        # pyrefly: ignore [missing-attribute]
        queries, annotations = self.coco.loadQueriesAndAnnotationsFromDatapoint(id)
        return self.load_queries(pil_images, annotations, queries, img_metadata)

    def load_queries(self, pil_images, annotations, queries, img_metadata):
        """Transform the raw image and queries into a Datapoint sample."""
        images: List[Image] = []
        id2index_img = {}
        id2index_obj = {}
        id2index_find_query = {}
        id2imsize = {}
        assert len(pil_images) == len(img_metadata)
        for i in range(len(pil_images)):
            w, h = pil_images[i][1].size
            blurring_mask = None
            if "blurring_mask" in img_metadata[i]:
                blurring_mask = img_metadata[i]["blurring_mask"]
            images.append(
                Image(
                    data=pil_images[i][1],
                    objects=[],
                    size=(h, w),
                    blurring_mask=blurring_mask,
                )
            )
            id2index_img[pil_images[i][0]] = i
            id2imsize[pil_images[i][0]] = (h, w)

        for annotation in annotations:
            image_id = id2index_img[annotation["image_id"]]
            bbox = box_xywh_to_xyxy(torch.as_tensor(annotation["bbox"])).view(1, 4)
            h, w = id2imsize[annotation["image_id"]]
            bbox[:, 0::2].mul_(w).clamp_(min=0, max=w)
            bbox[:, 1::2].mul_(h).clamp_(min=0, max=h)
            segment = None
            if self.load_segmentation and "segmentation" in annotation:
                # We're not decoding the RLE here, a transform will do it lazily later
                segment = annotation["segmentation"]
            images[image_id].objects.append(
                Object(
                    bbox=bbox[0],
                    area=annotation["area"],
                    object_id=(
                        annotation["object_id"] if "object_id" in annotation else -1
                    ),
                    frame_index=(
                        annotation["frame_index"] if "frame_index" in annotation else -1
                    ),
                    segment=segment,
                    is_crowd=(
                        annotation["is_crowd"] if "is_crowd" in annotation else None
                    ),
                    source=annotation["source"] if "source" in annotation else "",
                )
            )
            id2index_obj[annotation["id"]] = len(images[image_id].objects) - 1

        find_queries = []
        stage2num_queries = Counter()
        for i, query in enumerate(queries):
            stage2num_queries[query["query_processing_order"]] += 1
            id2index_find_query[query["id"]] = i

        # Sanity check: all the stages should have the same number of queries
        if len(stage2num_queries) == 0:
            num_queries_per_stage = 0
        else:
            num_queries_per_stage = stage2num_queries.most_common(1)[0][1]
        for stage, num_queries in stage2num_queries.items():
            assert num_queries == num_queries_per_stage, (
                f"Number of queries in stage {stage} is {num_queries}, expected {num_queries_per_stage}"
            )

        for query in queries:
            h, w = id2imsize[query["image_id"]]
            if (
                "input_box" in query
                and query["input_box"] is not None
                and len(query["input_box"]) > 0
            ):
                bbox = box_xywh_to_xyxy(torch.as_tensor(query["input_box"])).view(-1, 4)
                bbox[:, 0::2].mul_(w).clamp_(min=0, max=w)
                bbox[:, 1::2].mul_(h).clamp_(min=0, max=h)
                if "input_box_label" in query and query["input_box_label"] is not None:
                    bbox_label = torch.as_tensor(
                        query["input_box_label"], dtype=torch.long
                    ).view(-1)
                    assert len(bbox_label) == len(bbox)
                else:
                    # assume the boxes are positives
                    bbox_label = torch.ones(len(bbox), dtype=torch.long)
            else:
                bbox = None
                bbox_label = None

            if "input_points" in query and query["input_points"] is not None:
                points = torch.as_tensor(query["input_points"]).view(1, -1, 3)
                points[:, :, 0:1].mul_(w).clamp_(min=0, max=w)
                points[:, :, 1:2].mul_(h).clamp_(min=0, max=h)
            else:
                points = None

            try:
                original_image_id = int(
                    img_metadata[id2index_img[query["image_id"]]]["original_img_id"]
                )
            except ValueError:
                original_image_id = -1

            try:
                img_metadata_query = img_metadata[id2index_img[query["image_id"]]]
                coco_image_id = (
                    int(img_metadata_query["coco_img_id"])
                    if "coco_img_id" in img_metadata_query
                    else query["id"]
                )
            except KeyError:
                coco_image_id = -1

            try:
                original_category_id = int(query["original_cat_id"])
            except (ValueError, KeyError):
                original_category_id = -1

            # For evaluation, we associate the ids of the object to be tracked to the query
            if query["object_ids_output"]:
                obj_id = query["object_ids_output"][0]
                obj_idx = id2index_obj[obj_id]
                image_idx = id2index_img[query["image_id"]]
                object_id = images[image_idx].objects[obj_idx].object_id
                frame_index = images[image_idx].objects[obj_idx].frame_index
            else:
                object_id = -1
                frame_index = -1

            find_queries.append(
                FindQueryLoaded(
                    # id=query["id"],
                    # query_type=qtype,
                    query_text=(
                        query["query_text"] if query["query_text"] is not None else ""
                    ),
                    image_id=id2index_img[query["image_id"]],
                    input_bbox=bbox,
                    input_bbox_label=bbox_label,
                    input_points=points,
                    object_ids_output=[
                        id2index_obj[obj_id] for obj_id in query["object_ids_output"]
                    ],
                    is_exhaustive=query["is_exhaustive"],
                    is_pixel_exhaustive=(
                        query["is_pixel_exhaustive"]
                        if "is_pixel_exhaustive" in query
                        else (
                            query["is_exhaustive"] if query["is_exhaustive"] else None
                        )
                    ),
                    query_processing_order=query["query_processing_order"],
                    inference_metadata=InferenceMetadata(
                        coco_image_id=-1 if self.training else coco_image_id,
                        original_image_id=(-1 if self.training else original_image_id),
                        frame_index=frame_index,
                        original_category_id=original_category_id,
                        original_size=(h, w),
                        object_id=object_id,
                    ),
                )
            )

        return Datapoint(
            find_queries=find_queries,
            images=images,
            raw_images=[p[1] for p in pil_images],
        )

    def __len__(self) -> int:
        return len(self.ids)


class Sam3ImageDataset(CustomCocoDetectionAPI):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        max_ann_per_img: int,
        multiplier: int,
        training: bool,
        load_segmentation: bool = False,
        max_train_queries: int = 81,
        max_val_queries: int = 300,
        fix_fname: bool = False,
        is_sharded_annotation_dir: bool = False,
        blurring_masks_path: Optional[str] = None,
        use_caching: bool = True,
        zstd_dict_path=None,
        filter_query=None,
        coco_json_loader: Callable = COCO_FROM_JSON,
        # pyrefly: ignore [bad-function-definition]
        limit_ids: int = None,
        limit_ratio: float = 1.0,
        cache_images: Union[bool, str] = "none",
    ):
        super(Sam3ImageDataset, self).__init__(
            img_folder,
            ann_file,
            fix_fname=fix_fname,
            load_segmentation=load_segmentation,
            training=training,
            blurring_masks_path=blurring_masks_path,
            use_caching=use_caching,
            zstd_dict_path=zstd_dict_path,
            filter_query=filter_query,
            coco_json_loader=coco_json_loader,
            limit_ids=limit_ids,
            limit_ratio=limit_ratio,
            cache_images=cache_images,
        )

        self._transforms = transforms
        self.training = training
        self.max_ann_per_img = max_ann_per_img
        self.max_train_queries = max_train_queries
        self.max_val_queries = max_val_queries

        self.repeat_factors = torch.ones(len(self.ids), dtype=torch.float32)

        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.ids)}")

        self._MAX_RETRIES = 100

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(self, idx):
        return self.__orig_getitem__(idx)

    def __orig_getitem__(self, idx):
        for _ in range(self._MAX_RETRIES):
            try:
                datapoint = super(Sam3ImageDataset, self).__getitem__(idx)

                # This can be done better by filtering the offending find queries
                # However, this requires care:
                # - Delete any find/get query that may depend on the deleted one
                # - Re-compute the indexes in the pointers to account for the deleted finds
                for q in datapoint.find_queries:
                    if len(q.object_ids_output) > self.max_ann_per_img:
                        raise DecompressionBombError(
                            f"Too many outputs ({len(q.object_ids_output)})"
                        )

                max_queries = (
                    self.max_train_queries if self.training else self.max_val_queries
                )

                if len(datapoint.find_queries) > max_queries:
                    raise DecompressionBombError(
                        f"Too many find queries ({len(datapoint.find_queries)})"
                    )

                if len(datapoint.find_queries) == 0:
                    raise DecompressionBombError("No find queries")
                for transform in self._transforms:
                    datapoint = transform(datapoint, epoch=self.curr_epoch)

                break
            except (DecompressionBombError, OSError, ValueError) as error:
                sys.stderr.write(f"ERROR: got loading error on datapoint {idx}\n")
                sys.stderr.write(f"Exception: {error}\n")
                sys.stderr.write(traceback.format_exc())
                idx = (idx + 1) % len(self)
        else:
            raise RuntimeError(
                f"Failed {self._MAX_RETRIES} times trying to load an image."
            )

        return datapoint


class ConcatSam3Datasets(torch.utils.data.ConcatDataset):
    """拼接多个 Sam3ImageDataset, 并把 set_epoch/set_curr_epoch 转发给子集。

    torch.utils.data.ConcatDataset 本身不实现 set_epoch, 而 TorchDataset.get_loader
    会调 dataset.set_epoch(epoch) 把 epoch 传给 Sam3ImageDataset (它依赖 epoch 控制缓存
    等)。直接用原生 ConcatDataset 会让子集收不到 epoch, 故加此薄包装。
    """

    def __init__(self, datasets: List):
        super().__init__(datasets)

    def _fan_out(self, method: str, value):
        for ds in self.datasets:
            fn = getattr(ds, method, None)
            if callable(fn):
                fn(value)

    def set_epoch(self, epoch: int):
        self._fan_out("set_epoch", epoch)

    def set_curr_epoch(self, epoch: int):
        self._fan_out("set_curr_epoch", epoch)

    @property
    def epoch(self):
        # 取第一个有该属性的子集, 供 TorchDataset.get_loader 的 self.dataset.epoch 读取
        for ds in self.datasets:
            if hasattr(ds, "epoch"):
                return getattr(ds, "epoch")
        return None

    @epoch.setter
    def epoch(self, value):
        self._fan_out("__setattr__", ("epoch", value))
        # 上面的 fan_out 对 __setattr__ 不适用, 直接设到各子集
        for ds in self.datasets:
            if hasattr(ds, "epoch") or hasattr(ds, "set_epoch"):
                try:
                    ds.epoch = value
                except AttributeError:
                    pass
