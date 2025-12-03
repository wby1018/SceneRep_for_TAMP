# client_demo.py
import os, io, json, base64, argparse, time
import requests
import numpy as np
from PIL import Image
import cv2

def decode_mask_b64_to_bool(mask_b64: str) -> np.ndarray:
    """base64 PNG (L) -> bool numpy (H,W)"""
    data = base64.b64decode(mask_b64.encode("ascii"))
    arr = np.frombuffer(data, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # (H,W) uint8
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return (im > 127)

def render_overlay_client(image_rgb: np.ndarray, detections: list, alpha: float = 0.5) -> np.ndarray:
    """在客户端把 mask+box+label 叠加到原图（RGB uint8）"""
    canvas = image_rgb.copy()
    overlay = canvas.copy()

    palette = [
        (255, 0, 0), (0, 255, 0), (0, 128, 255),
        (255, 0, 255), (0, 255, 255), (255, 128, 0),
        (128, 0, 255), (0, 0, 255), (0, 128, 128),
        (128, 128, 0),
    ]

    # 1) 叠加 mask
    for i, det in enumerate(detections):
        mask = decode_mask_b64_to_bool(det["mask"])
        color = palette[i % len(palette)]
        idx = mask.astype(bool)
        overlay[idx] = (overlay[idx] * (1 - alpha) + np.array(color, dtype=np.uint8)[None, :] * alpha).astype(np.uint8)

    canvas = cv2.addWeighted(overlay, 1.0, canvas, 0.0, 0)

    # 2) 画框+标签（取第一候选类别）
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        color = palette[i % len(palette)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = det["detection"][0]["label"] if det["detection"] else ""
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1),
                      color=(255, 255, 255), thickness=-1)
        cv2.putText(canvas, label, (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)

    return canvas  # RGB

def detect_image(rgb_image, frame_idx, datapath, server_url="http://127.0.0.1:8000"):
    """
    检测单张RGB图像并保存结果
    
    Args:
        rgb_image: RGB图像数组 (H, W, 3)
        frame_idx: 帧索引
        datapath: 数据保存路径
        server_url: 检测服务器URL
    
    Returns:
        dict: 检测结果，格式为 {"detections": [...]}
    """
    # 确保输出目录存在
    output_dir = os.path.join(datapath, "detection_boxes")
    os.makedirs(output_dir, exist_ok=True)
    
    # 将numpy数组转换为PIL图像
    if isinstance(rgb_image, np.ndarray):
        img = Image.fromarray(rgb_image.astype(np.uint8))
    else:
        img = rgb_image
    
    # 将图像保存为临时文件
    temp_image_path = os.path.join(output_dir, f"temp_rgb_{frame_idx:06d}.png")
    img.save(temp_image_path)
    
    try:
        # 1) 上传图片 + 计时
        url = server_url.rstrip("/") + "/infer"
        with open(temp_image_path, "rb") as f:
            files = {"file": (f"rgb_{frame_idx:06d}.png", f, "application/octet-stream")}
            print(f"[POST] {url}  uploading frame {frame_idx}")
            t0 = time.time()
            r = requests.post(url, files=files, timeout=600)
            t1 = time.time()

        r.raise_for_status()
        resp = r.json()

        # 时间统计
        server_time_sec = float(resp.get("time_sec", -1))
        client_elapsed_sec = round(t1 - t0, 3)

        job_id = resp["job_id"]
        json_path = os.path.join(output_dir, f"detection_{frame_idx:06d}.json")
        png_path = os.path.join(output_dir, f"detection_{frame_idx:06d}.png")

        # 2) 保存 JSON（包含时间信息）
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "detections": resp["detections"],
            }, f, indent=2, ensure_ascii=False)

        # 3) 客户端叠加并保存 PNG
        img_np = np.array(img)
        overlay = render_overlay_client(img_np, resp["detections"], alpha=0.5)
        Image.fromarray(overlay).save(png_path)

        # print(f"[TIME] server_time_sec = {server_time_sec:.3f}s")
        # print(f"[TIME] client_elapsed_sec = {client_elapsed_sec:.3f}s")
        # print(f"[DONE] saved JSON → {json_path}")
        # print(f"[DONE] saved PNG  → {png_path}")
        
        # 返回检测结果
        return {"detections": resp["detections"]}
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--image",  type=str, help="path to local image",
                        default="/media/wby/2AB9-4188/data_move/rgb/rgb_000000.png")
    parser.add_argument("--out_dir", type=str, default="./client_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取原图（客户端用它来叠加）
    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)

    # 1) 上传图片 + 计时
    url = args.server.rstrip("/") + "/infer"
    with open(args.image, "rb") as f:
        files = {"file": (os.path.basename(args.image), f, "application/octet-stream")}
        print(f"[POST] {url}  uploading {args.image}")
        t0 = time.time()
        r = requests.post(url, files=files, timeout=600)
        t1 = time.time()

    r.raise_for_status()
    resp = r.json()

    # 时间统计
    server_time_sec = float(resp.get("time_sec", -1))
    client_elapsed_sec = round(t1 - t0, 3)

    job_id = resp["job_id"]
    json_path = os.path.join(args.out_dir, f"{job_id}.json")
    png_path  = os.path.join(args.out_dir, f"{job_id}.png")

    # 2) 保存 JSON（包含时间信息）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "width": resp["width"],
            "height": resp["height"],
            "detections": resp["detections"],
            "server_time_sec": server_time_sec,      # 服务器端检测耗时（模型执行+打包）
            "client_elapsed_sec": client_elapsed_sec # 客户端端到端耗时（HTTP请求+网络+服务器处理）
        }, f, indent=2, ensure_ascii=False)

    # 3) 客户端叠加并保存 PNG
    overlay = render_overlay_client(img_np, resp["detections"], alpha=0.5)
    Image.fromarray(overlay).save(png_path)

    print(f"[TIME] server_time_sec = {server_time_sec:.3f}s")
    print(f"[TIME] client_elapsed_sec = {client_elapsed_sec:.3f}s")
    print(f"[DONE] saved JSON → {json_path}")
    print(f"[DONE] saved PNG  → {png_path}")

if __name__ == "__main__":
    main()
