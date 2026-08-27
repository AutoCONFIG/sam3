# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Validation visualization meter (YOLO 风格, 检测+分割, 异步渲染)。

每次验证时, rank 0 把验证图跨 batch 累积, 每凑满 per_file 张拼一张近方形马赛克,
写到实验根目录 (与 ultralytics 一致, 文件名固定, 每轮验证覆盖):
  val_batch{f}_labels.jpg  — GT (检测框 + mask 半透明叠加 + 类别名)
  val_batch{f}_pred.jpg    — 预测 (检测框 + mask 半透明叠加 + 类别名 + 置信度)
共 max_files 个文件 (默认 3 × 16 = 48 张图)。训练进行中直接打开最新文件
即可看当前模型表现。仅供人工查看, 不产生指标; 任何失败仅告警不中断训练。

后处理 (仅可视化侧, 不影响 mAP 评测):
  - 置信度阈值 score_threshold 过滤 + min_per_img 兜底防空图 + max_per_img 截断
  - 可选 NMS 去重 (iou_threshold, 默认 0.5): DETR 训练初期多 query 抢同一目标
    产生冗余框, 去重让面板更干净; 设 0 关闭。后端 PostProcessImage 不做 NMS
    (mAP 靠匈牙利一对一匹配), 此处仅为可视化清晰
  - 有 mask 标注时画 mask+框, 无 mask 时只画框 (自适应检测/分割任务)

渲染 (mask 叠加/画框/拼图/JPEG 编码) 在后台 daemon 线程执行, 不阻塞训练;
上次未渲染完则跳过本次 (参考 yolo-project 的 plot 回调做法)。

作为 trainer.meters 的一员, 与 PredictionDumper 共用同一套 meter 协议:
update() 每 batch 调用一次, compute_synced() 每轮验证结束时调用, reset() 收尾。
"""

import colorsys
import hashlib
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import cv2
import numpy as np
import torch

from sam3.model.box_ops import box_iou
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


def _nms(boxes, scores, iou_threshold):
    """贪心 NMS (仅可视化用): 按分数降序, 依次抑制 IoU > 阈值的框。

    SAM3 是 DETR query 架构, 后端评测靠匈牙利一对一匹配不做 NMS;
    这里只在可视化侧去重训练初期多个 query 抢同一目标的冗余框,
    不影响 mAP。返回保留的索引 (相对输入顺序)。
    """
    n = boxes.shape[0]
    if n <= 1:
        return torch.arange(n, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    suppressed = torch.zeros(n, dtype=torch.bool, device=boxes.device)
    for idx in order.tolist():
        if suppressed[idx]:
            continue
        keep.append(idx)
        iou, _ = box_iou(boxes[idx].unsqueeze(0), boxes)
        suppressed |= iou.squeeze(0) > iou_threshold
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def _draw_panel(img, items):
    """单张图可视化: 检测框 + mask 半透明叠加 + YOLO 式标签。

    每条 item 始终画检测框 (色边); 有 mask 时额外叠半透明色块,
    标签贴 mask 左上角; 无 mask 时标签贴框左上角带底色块 (YOLO box_label 风格)。

    items: [(xyxy 像素坐标, mask|None, color BGR, label 文本)]
    """
    out = img.copy()
    h = out.shape[0]
    fs = max(0.6, h / 640.0 * 0.6)          # 字号随图尺寸缩放, 马赛克里也可读
    thick = max(1, int(round(fs * 1.5)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 第一遍: mask 半透明叠加 (在画框/文字之前, 避免框被半透明色冲淡)
    for _, mask, color, _ in items:
        if mask is not None and mask.shape == out.shape[:2]:
            m = mask.astype(bool)
            out[m] = (
                out[m].astype(np.float32) * 0.5 + np.array(color, dtype=np.float32) * 0.5
            ).astype(np.uint8)
    # 第二遍: 检测框 + 标签
    drawn = []  # 已画标签的矩形, 需错开避免叠字
    for box, mask, color, label in items:
        x1, y1, x2, y2 = [int(v) for v in box]
        # 检测框 (始终画, YOLO 风格色边)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        # 标签定位: 有 mask 贴 mask 左上角, 无 mask 贴框左上角
        if mask is not None and mask.shape == out.shape[:2]:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                tx, ty = x1, y1
            else:
                tx, ty = int(xs.min()), int(ys.min())
        else:
            tx, ty = x1, y1
        (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
        ty = max(ty, th + 6)
        rect = (tx, ty - th - 4, tx + tw + 4, ty + 2)
        for _ in range(8):
            if not any(
                not (rect[2] < r[0] or rect[0] > r[2] or rect[3] < r[1] or rect[1] > r[3])
                for r in drawn
            ):
                break
            ty += th + 8
            rect = (tx, ty - th - 4, tx + tw + 4, ty + 2)
        drawn.append(rect)
        cv2.rectangle(out, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
        cv2.putText(out, label, (tx + 2, ty - 2), font, fs, (255, 255, 255), thick, cv2.LINE_AA)
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


def _render_mosaic_pair(gt_raw, pred_raw, path_prefix):
    """后台线程任务: 画图 + 拼图 + 写盘。raw = [(img_uint8, items), ...]

    GT 不随训练变化, labels 文件已存在则跳过, 只重画 pred。
    """
    pred_panels = [_draw_panel(img, items) for img, items in pred_raw]
    h, w = pred_panels[0].shape[:2]
    labels_path = path_prefix + "_labels.jpg"
    if not os.path.exists(labels_path):
        gt_panels = [_draw_panel(img, items) for img, items in gt_raw]
        cv2.imwrite(labels_path, _mosaic(gt_panels, h, w))
    cv2.imwrite(path_prefix + "_pred.jpg", _mosaic(pred_panels, h, w))


class ValPredictionVisualizer:
    def __init__(
        self,
        out_dir: str,
        max_files: int = 3,
        per_file: int = 16,
        score_threshold: float = 0.1,
        min_per_img: int = 3,
        max_per_img: int = 20,
        iou_threshold: float = 0.5,
        use_presence: bool = True,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5),
    ):
        self.out_dir = out_dir
        self.max_files = max_files
        self.per_file = per_file
        self.score_threshold = score_threshold
        self.min_per_img = min_per_img
        self.max_per_img = max_per_img
        self.iou_threshold = iou_threshold
        self.use_presence = use_presence
        self.norm_mean = torch.tensor(norm_mean).view(3, 1, 1)
        self.norm_std = torch.tensor(norm_std).view(3, 1, 1)
        # 线程池并行渲染多批马赛克 (画 mask + 拼图 + imwrite 是 CPU/IO 密集);
        # max_files 通常 ≤ 几张, 池大小给 max(4, max_files) 足够不排队
        self._pool = ThreadPoolExecutor(max_workers=max(4, max_files))
        self._reset_buffer()

    def _reset_buffer(self):
        self._gt_raw = []    # [(img_uint8, gt_items)]
        self._pred_raw = []  # [(img_uint8, pred_items)]

    @torch.no_grad()
    def update(self, find_stages, find_metadatas, model, batch, key):
        max_imgs = self.max_files * self.per_file
        if not is_main_process() or len(self._gt_raw) >= max_imgs:
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

            # ── Pred: 得分公式与后端 PostProcessImage(use_presence) 一致 ──
            # (logits.sigmoid() × presence.sigmoid()); mask 只对筛出的高分 query
            # 插值 —— 若像后处理器那样对全部 200 个 query 的 mask 上采样到画布,
            # 1008 分辨率下一次要 ~10GB 显存, 会 OOM
            out_probs = outputs["pred_logits"].sigmoid()  # [R, Q, 1]
            if self.use_presence and "presence_logit_dec" in outputs:
                presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
                out_probs = out_probs * presence
            pred_masks = outputs["pred_masks"] if "pred_masks" in outputs else None
            for r in range(R):
                text = row_text[r]
                color = _text_color(text)
                scores = out_probs[r, :, 0]
                keep = (scores > self.score_threshold).nonzero(as_tuple=True)[0]
                # 零样本/训练初期分数普遍很低, 至少保留 top min_per_img 让面板不空
                topk = torch.argsort(scores, descending=True)[: self.min_per_img]
                keep = torch.unique(torch.cat([keep, topk]))
                keep = keep[torch.argsort(scores[keep], descending=True)]
                keep = keep[: self.max_per_img]
                # NMS 去重 (仅可视化): DETR 训练初期多个 query 会抢同一目标,
                # 产生高度重叠的冗余框; iou_threshold=0 关闭。在 CPU 上对已截断
                # 到 max_per_img 的少量框做, 开销可忽略; mask 只对幸存 query 插值
                if self.iou_threshold > 0 and len(keep) > 1:
                    nms_boxes = _cxcywh_norm_to_xyxy(
                        outputs["pred_boxes"][r, keep].float().cpu(), W, H
                    )
                    nms_idx = _nms(nms_boxes, scores[keep].cpu(), self.iou_threshold)
                    keep = keep[nms_idx]
                boxes = _cxcywh_norm_to_xyxy(
                    outputs["pred_boxes"][r, keep].float().cpu(), W, H
                ).tolist()
                masks = None
                if pred_masks is not None:
                    m = pred_masks[r, keep].float()
                    if m.dim() == 3:  # [K, h, w] → [K, 1, h, w]
                        m = m.unsqueeze(1)
                    masks = (
                        torch.nn.functional.interpolate(
                            m, (H, W), mode="bilinear", align_corners=False,
                        ).sigmoid() > 0.5
                    ).squeeze(1).cpu().numpy()
                for j, k in enumerate(keep.cpu().tolist()):
                    mask = masks[j] if masks is not None else None
                    score = scores[k].item()
                    pred[row_img[r]].append(
                        (boxes[j], mask, color, f"{text} {score:.2f}")
                    )

        # ── 攒图 (仅取数/反归一化, 轻量); 攒够 max_files*per_file 张就停,
        #    不逐批提交 — val 结束后统一交给线程池后台渲染, 不丢图 ──
        max_imgs = self.max_files * self.per_file
        for i in range(B):
            if len(self._gt_raw) >= max_imgs:
                break
            canvas = (imgs[i].float().cpu() * self.norm_std + self.norm_mean)
            canvas = (canvas.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            self._gt_raw.append((canvas, gt.get(i, [])))
            self._pred_raw.append((canvas, pred.get(i, [])))

    def _submit_all(self):
        """val 结束时调用: 把攒好的图按 per_file 切批, 统一提交线程池渲染。"""
        if not self._gt_raw:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        total = len(self._gt_raw)
        n_batches = min(self.max_files, (total + self.per_file - 1) // self.per_file)
        for b in range(n_batches):
            chunk = slice(b * self.per_file, min((b + 1) * self.per_file, total))
            gt_raw = self._gt_raw[chunk]
            pred_raw = self._pred_raw[chunk]
            prefix = os.path.join(self.out_dir, f"val_batch{b}")
            n = len(gt_raw)

            def _job(gt_raw=gt_raw, pred_raw=pred_raw, prefix=prefix, n=n):
                try:
                    _render_mosaic_pair(gt_raw, pred_raw, prefix)
                    logging.info(
                        f"ValPredictionVisualizer: wrote {prefix}_labels/pred.jpg ({n} images)"
                    )
                except Exception:
                    logging.exception("ValPredictionVisualizer: async render failed")

            self._pool.submit(_job)
        self._gt_raw = []
        self._pred_raw = []

    def synchronize_between_processes(self):
        pass  # 只在 rank 0 本地出图, 无需同步

    def compute_synced(self):
        if is_main_process():
            try:
                self._submit_all()  # val 结束: 统一提交后台渲染
            except Exception:
                logging.exception("ValPredictionVisualizer: flush failed, skipping")
        return {}

    def compute(self):
        return {}

    def reset(self):
        self._reset_buffer()
