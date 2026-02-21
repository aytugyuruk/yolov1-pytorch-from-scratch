import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, C=1):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.C = C

    def forward(self, predictions, target):
        # --- Before start I will show how the predictions and target look like ---
        # --- Since we are using B = 2 and C = 1 we will have the length of 11 for the predictions and target ---
        # --- B = 2 means our model predict 2 boxes for each cell and C = 1 means we have 1 class to predict ---

        # --- Prediction ---
        # prediction >> (class_prob, x1, y1, w1, h1, conf1, x2, y2, w2, h2, conf2)
        # --- Target ---
        # target >> (class_prob, x, y, w, h, conf, x, y, w, h, conf)

        # --- STEP 1 ---
        # --- In this step we will decide which box is better --- 
        # --- To decide is we will use the function that we defined in the utils.py ---
        box1_coords = slice(self.C, self.C + 4)
        box1_conf = slice(self.C + 4, self.C + 5)
        box2_coords = slice(self.C + 5, self.C + 9)
        box2_conf = slice(self.C + 9, self.C + 10)

        iou_b1 = intersection_over_union(predictions[..., box1_coords], target[..., box1_coords]).squeeze(-1)
        iou_b2 = intersection_over_union(predictions[..., box2_coords], target[..., box1_coords]).squeeze(-1)
        ious = torch.stack([iou_b1, iou_b2], dim=-1)
        iou_maxes, bestbox = torch.max(ious, dim=-1)
        bestbox = bestbox.unsqueeze(-1).float()

        # --- Step 2 ---
        # --- In this step we will calculate the loss for the box which is responsible for the object ---
        # bestbox = 1 means box2 is better, bestbox = 0 means box1 is better
        exists_box = target[..., box1_conf]
        box_predictions = bestbox * predictions[..., box2_coords] + (1 - bestbox) * predictions[..., box1_coords]
        box_targets = target[..., box1_coords].clone()
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4] + 1e-6)
        box_loss = self.mse(torch.flatten(exists_box * box_predictions, end_dim=-2),torch.flatten(exists_box * box_targets, end_dim=-2))

        # --- STEP 3 ---
        # --- In this step we will penalize the confidence of the box which has the object ---
        pred_conf = bestbox * predictions[..., box2_conf] + (1 - bestbox) * predictions[..., box1_conf]
        object_loss = self.mse(torch.flatten(exists_box * pred_conf),torch.flatten(exists_box * target[..., box1_conf]))

        # --- STEP 4 ---
        # --- In this step we will penalize the confidence of the box which doesn't have the object ---
        no_object_loss = self.mse(torch.flatten((1 - exists_box) * predictions[..., box1_conf]),torch.flatten((1 - exists_box) * target[..., box1_conf]))
        no_object_loss += self.mse(torch.flatten((1 - exists_box) * predictions[..., box2_conf]),torch.flatten((1 - exists_box) * target[..., box1_conf]))

        # --- STEP 5 ---
        # --- In this step we will penalize the class probabilities ---
        class_loss = self.mse(torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),torch.flatten(exists_box * target[..., :self.C], end_dim=-2),)

        # --- STEP 6 ---
        # --- In this step we will calculate the total loss as a sum of all the losses ---
        loss = ((self.lambda_coord * box_loss) + object_loss + (self.lambda_noobj * no_object_loss) + class_loss)

        return loss
    
# NOTE: This file is the most difficult one that I have written in this project. So if you understand the logic behind the loss function congratulations.
