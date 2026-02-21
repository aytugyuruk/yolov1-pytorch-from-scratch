import torch
import os
import pandas as pd
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S, B, C, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in line.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image = self.transform(image)

        label_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            box1_coord_start = self.C
            box1_coord_end   = self.C + 4
            box1_conf_idx    = self.C + 4

            if label_matrix[i, j, box1_conf_idx] == 0:
                label_matrix[i, j, box1_conf_idx] = 1
                label_matrix[i, j, box1_coord_start:box1_coord_end] = torch.tensor([x_cell, y_cell, width, height])

                if class_label < self.C:
                    label_matrix[i, j, class_label] = 1

        return image, label_matrix