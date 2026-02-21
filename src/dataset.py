import os
import torch
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, S=7, B=2, C=2, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.images = sorted([f for f in os.listdir(self.img_dir)if os.path.splitext(f.lower())[1] == ".jpg"])

    def __len__(self):
        return len(self.images)

    def _label_path_from_image(self, img_name: str) -> str:
        base, _ = os.path.splitext(img_name)
        return os.path.join(self.label_dir, base + ".txt")

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = self._label_path_from_image(img_name)
        image = Image.open(img_path).convert("RGB")

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls, x, y, w, h = line.split()
                    boxes.append([int(float(cls)), float(x), float(y), float(w), float(h)])
        else:
            boxes = []

        if self.transform:
            image = self.transform(image)

        label_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5), dtype=torch.float32)

        box1_coord_start = self.C
        box1_coord_end   = self.C + 4
        box1_conf_idx    = self.C + 4

        for (class_label, x, y, width, height) in boxes:
            x = min(max(x, 0.0), 1.0 - 1e-6)
            y = min(max(y, 0.0), 1.0 - 1e-6)

            i = int(self.S * y)
            j = int(self.S * x)

            x_cell = self.S * x - j
            y_cell = self.S * y - i

            if label_matrix[i, j, box1_conf_idx] == 0:
                label_matrix[i, j, box1_conf_idx] = 1.0

                label_matrix[i, j, box1_coord_start:box1_coord_end] = torch.tensor([x_cell, y_cell, width, height], dtype=torch.float32)

                if 0 <= class_label < self.C:
                    label_matrix[i, j, class_label] = 1.0

        return image, label_matrix