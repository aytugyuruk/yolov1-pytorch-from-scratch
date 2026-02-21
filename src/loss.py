import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # --- Before start I will show how the predictions and target look like ---
        # --- Since we are using B = 2 and C = 2 we will have the length of 12 for the predictions and target ---
        # --- B = 2 means our model predict 2 boxes for each cell and C = 2 means we have 2 classes to predict ---

        # --- Prediction ---
        # prediction >> (class1_prob, class2_prob, x1, y1, w1, h1, conf1, x2, y2, w2, h2, conf2)
        # --- Target ---
        # target >> (class1_prob, class2_prob, x, y, w, h, conf, x, y, w, h, conf)

        # --- STEP 1 ---
        # --- In this step we will decide which box is better --- 
        # --- To decide is we will use the function that we defined in the utils.py ---
        iou_b1 = intersection_over_union(predictions[..., 2:6],target[...,2:6])
        iou_b2 = intersection_over_union(predictions[..., 7:11],target[...,2:6])
        ious = torch.stack([iou_b1, iou_b2], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        bestbox = bestbox.unsqueeze(-1).float()

        # --- Step 2 ---
        # --- In this step we will calculate the loss for the box which is responsible for the object ---
        # bestbox = 1 means box2 is better, bestbox = 0 means box1 is better
        exists_box = target[..., 6:7]
        box_predictions = bestbox * predictions[..., 7:11] + (1 - bestbox) * predictions[..., 2:6]
        box_targets = target[..., 2:6].clone()
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4] + 1e-6)
        box_loss = self.mse(torch.flatten(exists_box * box_predictions, end_dim=-2),torch.flatten(exists_box * box_targets, end_dim=-2))

        # --- STEP 3 ---
        # --- In this step we will penalize the confidence of the box which has the object ---
        pred_conf = bestbox * predictions[..., 11:12] + (1 - bestbox) * predictions[..., 6:7]
        object_loss = self.mse(torch.flatten(exists_box * pred_conf),torch.flatten(exists_box * target[..., 6:7]))

        # --- STEP 4 ---
        # --- In this step we will penalize the confidence of the box which doesn't have the object ---
        no_object_loss = self.mse(torch.flatten((1 - exists_box) * predictions[..., 6:7]),torch.flatten((1 - exists_box) * target[..., 6:7]))
        no_object_loss += self.mse(torch.flatten((1 - exists_box) * predictions[..., 11:12]),torch.flatten((1 - exists_box) * target[..., 6:7]))

        # --- STEP 5 ---
        # --- In this step we will penalize the class probabilities ---
        class_loss = self.mse(torch.flatten(exists_box * predictions[..., :2], end_dim=-2),torch.flatten(exists_box * target[..., :2], end_dim=-2),)

        # --- STEP 6 ---
        # --- In this step we will calculate the total loss as a sum of all the losses ---
        loss = ((self.lambda_coord * box_loss) + object_loss + (self.lambda_noobj * no_object_loss) + class_loss)

        return loss
    
# NOTE: This file is the most difficult one that I have written in this project. So if you understand the logic behind the loss function congratulations.