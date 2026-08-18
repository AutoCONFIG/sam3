"""SAM 3.1 视频推理脚本（最小可跑版本，不依赖 flash-attn）。

用法:
    python infer_video.py \
        --checkpoint sam3.1/sam3.1_multiplex.pt \
        --input path/to/video.mp4 或 jpg帧目录 \
        --output runs/inference/test \
        --text "person"

说明:
    - use_fa3=False, 不需要安装 flash-attn-3, 推理稍慢但能跑通。
    - 输入支持 mp4/avi 等视频文件, 或 JPEG 帧序列目录。
    - 输出: vis/ 下每帧叠加图, masks/ 下每帧 npz(mask+obj_id+score)。
"""

import argparse
import os
import sys
from pathlib import Path

# 减少 CUDA 显存碎片, 对 16G 显卡有帮助 (须在 torch 初始化前设置)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np


def get_frames(input_path: str):
    """返回 (帧列表[HWC,RGB], fps, 总帧数)。支持视频文件或图片目录。"""
    p = Path(input_path)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(
            [f for f in p.iterdir() if f.suffix.lower() in exts],
            key=lambda x: x.name,
        )
        if not files:
            raise ValueError(f"目录 {input_path} 下没有图片")
        frames = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
        return frames, None, len(frames)
    elif p.is_file():
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频 {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames, fps, len(frames)
    else:
        raise ValueError(f"输入路径不存在: {input_path}")


def save_frame_results(
    frame_idx: int,
    frame_rgb: np.ndarray,
    outputs: dict,
    out_dir: Path,
    save_vis: bool,
    save_masks: bool,
):
    """保存单帧的可视化叠加图和 mask 数据。"""
    obj_ids = outputs.get("out_obj_ids", [])
    masks = outputs.get("out_binary_masks", [])
    probs = outputs.get("out_probs", [])
    boxes = outputs.get("out_boxes_xywh", [])

    if save_masks:
        # 把每个对象的 mask 叠成一个 label map (0=背景, oid+1=对象), 便于后续处理
        h, w = frame_rgb.shape[:2]
        label_map = np.zeros((h, w), dtype=np.int32)
        meta = []
        for i, (oid, m) in enumerate(zip(obj_ids, masks)):
            m = np.asarray(m).astype(bool)
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            real_id = int(oid) if oid is not None else i
            label_map[m] = real_id + 1  # 0 保留给背景
            meta.append(
                {
                    "obj_id": real_id,
                    "score": float(probs[i]) if i < len(probs) else None,
                    "box_xywh": [float(v) for v in boxes[i]] if i < len(boxes) else None,
                }
            )
        np.savez_compressed(out_dir / "masks" / f"{frame_idx:06d}.npz", label_map=label_map, meta=np.array(meta, dtype=object))

    if save_vis:
        vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
        palette = np.random.default_rng(42).integers(64, 255, size=(256, 3), dtype=np.uint8)
        for i, (oid, m) in enumerate(zip(obj_ids, masks)):
            m = np.asarray(m).astype(bool)
            if m.shape != vis.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            color = palette[int(oid) % 256] if oid is not None else palette[i % 256]
            vis[m] = (vis[m].astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5).astype(np.uint8)
            # 画框 + id
            if i < len(boxes):
                x, y, bw, bh = boxes[i]
                x, y, bw, bh = float(x), float(y), float(bw), float(bh)
                cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)), color.tolist(), 2)
            label = f"id:{int(oid) if oid is not None else i}"
            if i < len(probs):
                label += f" {float(probs[i]):.2f}"
            cv2.putText(vis, label, (int(x) + 2, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "vis" / f"{frame_idx:06d}.jpg"), vis)


def main():
    ap = argparse.ArgumentParser(description="SAM 3.1 视频推理")
    ap.add_argument("--checkpoint", "-c", required=True, help="sam3.1_multiplex.pt 路径")
    ap.add_argument("--input", "-i", required=True, help="视频文件或 jpg 帧目录")
    ap.add_argument("--output", "-o", default="runs/inference/sam3", help="输出目录")
    ap.add_argument("--text", "-t", required=True, help="文本提示, 如 'person'")
    ap.add_argument("--frame-index", type=int, default=0, help="添加文本提示的帧索引")
    ap.add_argument("--no-fa3", action="store_true", default=True, help="不使用 flash-attn(默认, 无需安装)")
    ap.add_argument("--compile", action="store_true", help="启用 torch.compile(首次慢)")
    ap.add_argument("--save-vis", action="store_true", default=True, help="保存可视化叠加图")
    ap.add_argument("--save-masks", action="store_true", default=True, help="保存 mask npz")
    ap.add_argument("--save-video", action="store_true", help="把 vis 合成 mp4")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                    help="模型精度 (模型自带 bf16 autocast, 一般用默认 fp32 即可)")
    ap.add_argument("--image-size", type=int, default=1008,
                    help="推理分辨率, 须为 336 的倍数 (14*24); 16G 显存用 672 (默认 1008)")
    args = ap.parse_args()

    out_dir = Path(args.output)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    print(f"加载视频/帧: {args.input}")
    frames, fps, n = get_frames(args.input)
    print(f"共 {n} 帧, fps={fps}")
    if n == 0:
        print("错误: 没有帧可处理"); sys.exit(1)

    print(f"构建模型 (use_fa3={not args.no_fa3})...")
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=args.checkpoint,
        use_fa3=not args.no_fa3,   # 不装 flash-attn 时设 False
        use_rope_real=True,        # 用实数 RoPE (避免 complex64 buffer)
        compile=args.compile,
        warm_up=args.compile,
        image_size=args.image_size,
    )
    print(f"image_size={args.image_size} (backbone ViT 仍用 1008 预训练位置编码, tile_abs_pos 自动适配)")
    # 注意: Sam3MultiplexVideoPredictor 构造时已全局开启 bf16 autocast,
    # 权重保持 fp32, 前向自动用 bf16 计算。不要手动转权重 dtype
    # (decoder FFN 内有 autocast(enabled=False) 会与 bf16 权重冲突)。
    import torch as _torch
    _torch.cuda.empty_cache()

    def _req(request):
        """调用 handle_request (模型构造时已开 bf16 autocast, 无需再包)。"""
        return predictor.handle_request(request=request)

    def _stream(request):
        """调用 handle_stream_request (模型构造时已开 bf16 autocast, 无需再包)。"""
        for r in predictor.handle_stream_request(request=request):
            yield r

    # start_session: resource_path 需要是磁盘路径, 我们先把帧写成临时 jpg 目录
    import tempfile, shutil

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_frames_"))
    try:
        for i, fr in enumerate(frames):
            cv2.imwrite(str(tmp_dir / f"{i:08d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        print(f"帧已写入临时目录: {tmp_dir}")

        resp = _req({"type": "start_session", "resource_path": str(tmp_dir),
                     "offload_video_to_cpu": True})
        sid = resp["session_id"]
        print(f"session: {sid} (offload_video_to_cpu=True 省 GPU 显存)")

        # 添加文本提示
        resp = _req(
            {"type": "add_prompt", "session_id": sid, "frame_index": args.frame_index, "text": args.text}
        )
        out0 = resp["outputs"]
        n_obj = len(out0.get("out_obj_ids", []))
        print(f"帧 {args.frame_index} 检测到 {n_obj} 个对象 '{args.text}'")

        # 传播到所有帧 (stream 请求是生成器, 每次 yield 一帧)
        print("传播到全部帧 ...")
        total = 0
        for response in _stream(
            {"type": "propagate_in_video", "session_id": sid}
        ):
            fi = response["frame_index"]
            if fi >= len(frames):
                break
            outputs = response.get("outputs", {})
            save_frame_results(fi, frames[fi], outputs, out_dir, args.save_vis, args.save_masks)
            n_obj_i = len(outputs.get("out_obj_ids", []))
            total += n_obj_i
            if fi % 20 == 0 or fi == n - 1:
                print(f"  帧 {fi}/{n-1}: {n_obj_i} 对象")

        # 关闭会话
        try:
            _req({"type": "close_session", "session_id": sid})
        except Exception:
            pass

        print(f"\n完成。总对象-帧数: {total}")
        print(f"可视化: {out_dir/'vis'}")
        print(f"mask:   {out_dir/'masks'}")

        if args.save_video and fps:
            vis_files = sorted((out_dir / "vis").glob("*.jpg"))
            if vis_files:
                h, w = frames[0].shape[:2]
                vw = cv2.VideoWriter(str(out_dir / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for vf in vis_files:
                    vw.write(cv2.imread(str(vf)))
                vw.release()
                print(f"视频: {out_dir/'result.mp4'}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
