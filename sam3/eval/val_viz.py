# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Validation visualization meter (YOLO 风格, 分割任务)。

每次验证时, rank 0 把验证图跨 batch 累积, 每凑满 per_file 张拼一张 4x4 马赛克,
写到实验根目录 (与 ultralytics 一致, 文件名固定, 每轮验证覆盖):
  val_batch{f}_labels.jpg  — GT (mask 半透明叠加 + 类别名)
  val_batch{f}_pred.jpg    — 预测 (mask 半透明叠加 + 类别名 + 置信度)
共 max_files 个文件 (默认 3 × 16 = 48 张图)。训练进行中直接打开最新文件
即可看当前模型表现。仅供人工查看, 不产生指标; 任何失败仅告警不中断训练。

作为 trainer.meters 的一员, 与 PredictionDumper 共用同一套 meter 协议:
update() 每 batch 调用一次, compute_synced() 每轮验证结束时调用, reset() 收尾。
"""

import colorsys
import hashlib
import logging
import math
import os
from collections import defaultdict

import cv2
import numpy as np
import torch

from sam3.train.utils.distributed import is_main_process


def _text_color(text: str):
    """由类别文本确定性生成 BGR 颜色 (跨 epoch 一致, 便于对比)。"""
    h = int(hashlib.md5(text.encode()).hexdigest(), 16) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def _cxcywh_norm_to_xyxy(boxes, width, height):
    """normalized cxcywh [N,4] → 像素 xyxy [N,4]"""
    cx, cy, w, h = boxes.unbind(-1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    scale = torch.tensor([width, height, width, height], device=xyxy.device)
    return (xyxy * scale).round().long()


def _draw_panel(img, items):
    """单张图的分割可视化: mask 半透明叠加 + 类别文字 (分割任务不画检测框)。

    items: [(xyxy 仅用于无 mask 时的文字定位, mask|None, color, label)]
    """
    out = img.copy()
    for box, mask, color, label in items:
        if mask is not None and mask.shape == out.shape[:2]:
            m = mask.astype(bool)
            out[m] = (
                out[m].astype(np.float32) * 0.5 + np.array(color, dtype=np.float32) * 0.5
            ).astype(np.uint8)
    for box, mask, color, label in items:
        if mask is not None and mask.shape == out.shape[:2]:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            tx, ty = int(xs.min()), max(12, int(ys.min()) - 4)
        else:
            # 无 mask (解码失败等兜底): 文字放框左上角, 仍不画框
            tx, ty = int(box[0]), max(12, int(box[1]) - 4)
        cv2.putText(
            out, label, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return out


def _mosaic(panels, cell_h, cell_w):
    """把一组图画按近方形网格拼成一张马赛克 (不足补灰底)。"""
    n = len(panels)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.full((rows * cell_h, cols * cell_w, 3), 114, dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = p
    return grid


class ValPredictionVisualizer:
    def __init__(
        self,
        out_dir: str,
        postprocessor,
        max_files: int = 3,
        per_file: int = 16,
        score_threshold: float = 0.3,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5),
    ):
        self.out_dir = out_dir
        self.postprocessor = postprocessor
        self.max_files = max_files
        self.per_file = per_file
        self.score_threshold = score_threshold
        self.norm_mean = torch.tensor(norm_mean).view(3, 1, 1)
        self.norm_std = torch.tensor(norm_std).view(3, 1, 1)
        self._reset_buffer()

    def _reset_buffer(self):
        self._file_idx = 0
        self._gt_panels = []
        self._pred_panels = []

    @torch.no_grad()
    def update(self, find_stages, find_metadatas, model, batch, key):
        if not is_main_process() or self._file_idx >= self.max_files:
            return
        try:
            self._update(find_stages, find_metadatas, batch)
        except Exception:
            # 可视化是附属功能, 任何失败都不应影响训练
            logging.exception("ValPredictionVisualizer: render failed, skipping")

    def _update(self, find_stages, find_metadatas, batch):
        imgs = batch.img_batch  # [B, 3, H, W], 已按 val transforms 归一化
        B, _, H, W = imgs.shape

        # process_results 里的 loss_stages 重排逻辑, 保持一致
        metas = find_metadatas
        if getattr(find_stages, "loss_stages", None) is not None:
            metas = [find_metadatas[i] for i in find_stages.loss_stages]

        # 注意: 一个 find stage 的"行"是 (图片, 类别查询) 对, 数量 ≠ 图片数;
        # 行 → 图片/文本 的映射在 find_inputs 的 img_ids / text_ids 里
        gt = defaultdict(list)   # img_idx -> [(xyxy, mask, color, label)]
        pred = defaultdict(list)

        for stage_idx, (outputs, meta) in enumerate(zip(find_stages, metas)):
            R = outputs["pred_logits"].shape[0]
            stage_in = batch.find_inputs[stage_idx]
            row_img = stage_in.img_ids.cpu().tolist()       # 行 → img_batch 下标
            row_text = [batch.find_text_batch[t] for t in stage_in.text_ids.cpu().tolist()]
            sizes = torch.tensor([[H, W]] * R, dtype=torch.long, device=imgs.device)

            # ── GT: boxes_padded 归一化 cxcywh + segments mask ──
            tgt = batch.find_targets[stage_idx]
            n_boxes = tgt.num_boxes.cpu().tolist()
            seg = tgt.segments
            seg_packed = (
                seg is not None and seg.dim() == 3 and seg.shape[0] == sum(n_boxes)
            )
            seg_padded = (
                seg is not None and seg.dim() == 4 and seg.shape[0] == R
            )
            seg_split = torch.split(seg, n_boxes) if seg_packed else None
            for r in range(R):
                n = n_boxes[r]
                if n == 0:
                    continue
                text = row_text[r]
                color = _text_color(text)
                boxes = _cxcywh_norm_to_xyxy(
                    tgt.boxes_padded[r, :n].float().cpu(), W, H
                )
                for j in range(n):
                    mask = None
                    if seg_packed:
                        mask = seg_split[r][j].cpu().numpy()
                    elif seg_padded:
                        mask = seg[r, j].cpu().numpy()
                    gt[row_img[r]].append((boxes[j].tolist(), mask, color, text))

            # ── Pred: postprocessor 直接出画布尺度 box/mask ──
            results = self.postprocessor(
                outputs, sizes, sizes, consistent=True
            )  # list of R dicts: scores/labels/boxes[/masks]
            for r in range(R):
                text = row_text[r]
                color = _text_color(text)
                res = results[r]
                keep = res["scores"] > self.score_threshold
                for k in keep.nonzero(as_tuple=True)[0].tolist():
                    mask = res["masks"][k].cpu().numpy() if "masks" in res else None
                    box = res["boxes"][k].cpu().tolist()
                    score = res["scores"][k].item()
                    pred[row_img[r]].append((box, mask, color, f"{text} {score:.2f}"))

        # ── 攒图: 每凑满 per_file 张落盘一张马赛克 ──
        for i in range(B):
            if self._file_idx >= self.max_files:
                break
            canvas = (imgs[i].float().cpu() * self.norm_std + self.norm_mean)
            canvas = (canvas.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            self._gt_panels.append(_draw_panel(canvas, gt.get(i, [])))
            self._pred_panels.append(_draw_panel(canvas, pred.get(i, [])))
            if len(self._gt_panels) >= self.per_file:
                self._flush()

    def _flush(self):
        """把缓冲的图写成一对 labels/pred 马赛克并清空缓冲。"""
        if not self._gt_panels:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        h, w = self._gt_panels[0].shape[:2]
        f = self._file_idx
        cv2.imwrite(
            os.path.join(self.out_dir, f"val_batch{f}_labels.jpg"),
            _mosaic(self._gt_panels, h, w),
        )
        cv2.imwrite(
            os.path.join(self.out_dir, f"val_batch{f}_pred.jpg"),
            _mosaic(self._pred_panels, h, w),
        )
        logging.info(
            f"ValPredictionVisualizer: wrote val_batch{f}_labels/pred.jpg "
            f"({len(self._gt_panels)} images) to {self.out_dir}"
        )
        self._file_idx += 1
        self._gt_panels = []
        self._pred_panels = []

    def synchronize_between_processes(self):
        pass  # 只在 rank 0 本地出图, 无需同步

    def compute_synced(self):
        if is_main_process() and self._file_idx < self.max_files:
            try:
                self._flush()  # 验证收尾: 落盘不足 per_file 的零头
            except Exception:
                logging.exception("ValPredictionVisualizer: flush failed, skipping")
        return {}

    def compute(self):
        return {}

    def reset(self):
        self._reset_buffer()
