import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import YOLOv1
from utils import non_max_suppression

# --- Config ---
DEVICE = "mps"  # or "cuda" / "cpu"
MODEL_PATH = "yolo_v1_best.pth"
OUTPUT_PATH = "test_results.png"

S, B, C = 7, 2, 1
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 2
CLASS_NAMES = ["object"]

IMAGE_PATHS = ["dataset/test/images/e706cec0-783a-42bc-b934-64788481cd92___4-Innova-After-2-jpg_jpeg.rf.16034922ef0ee389ee29128a6a235ab2.jpg","dataset/test/images/e29b6eea-504a-4da2-8503-5bd3443f3bd8___maruti-suzuki-wagon-r-front-left-rim-jpg_jpeg.rf.20e01be26a15b74b163c939f24c5a1f5.jpg"]
transform = transforms.Compose([transforms.Resize((448, 448)),transforms.ToTensor()])

def decode_predictions(output: torch.Tensor):
    out = output.reshape(S, S, C + B * 5)
    boxes = []

    for i in range(S):
        for j in range(S):
            class_scores = out[i, j, :C]
            class_id = int(torch.argmax(class_scores).item())
            class_prob = float(torch.clamp(class_scores[class_id], 0.0, 1.0).item())

            for b in range(B):
                start = C + b * 5
                x, y, w, h = out[i, j, start:start + 4]
                objectness = float(torch.clamp(out[i, j, start + 4], 0.0, 1.0).item())
                conf = objectness * class_prob

                x_abs = (j + float(torch.clamp(x, 0.0, 1.0).item())) / S
                y_abs = (i + float(torch.clamp(y, 0.0, 1.0).item())) / S
                w_abs = float(torch.clamp(torch.abs(w), 0.0, 1.0).item())
                h_abs = float(torch.clamp(torch.abs(h), 0.0, 1.0).item())

                boxes.append([class_id, conf, x_abs, y_abs, w_abs, h_abs])
    return boxes

def to_abs_xywh(box, orig_w, orig_h):
    cls, conf, x, y, w, h = box
    x1 = (x - w / 2) * orig_w
    y1 = (y - h / 2) * orig_h
    bw = w * orig_w
    bh = h * orig_h
    return cls, conf, x1, y1, bw, bh

# --- Main ---
def main():
    model = YOLOv1(in_channels=3, split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    with torch.inference_mode():
        for ax, img_path in zip(axes, IMAGE_PATHS):
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            inp = transform(image).unsqueeze(0).to(DEVICE)
            output = model(inp).squeeze(0).cpu()

            raw_boxes = decode_predictions(output)
            final_boxes = non_max_suppression(raw_boxes,iou_threshold=IOU_THRESHOLD,threshold=CONF_THRESHOLD)
            final_boxes = sorted(final_boxes, key=lambda b: b[1], reverse=True)[:MAX_DETECTIONS]

            ax.imshow(image)
            ax.axis("off")

            for box in final_boxes:
                cls, conf, x1, y1, bw, bh = to_abs_xywh(box, orig_w, orig_h)

                ax.add_patch(patches.Rectangle((x1, y1), bw, bh, linewidth=5, edgecolor="red", facecolor="none"))
                ax.text(x1, max(0, y1 - 8),f"{CLASS_NAMES[int(cls)]} {conf:.2f}",color="black", fontsize=15,bbox=dict(facecolor="white", alpha=0.7, pad=2, edgecolor="none"))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()