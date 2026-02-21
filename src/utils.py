import torch

# --- This function is used to calculate the IoU between two bounding boxes ---
# --- In our model.py file, it produce the output of shape 7x7x12
# --- We take the 4 output values of the predicted box
# --- And the 4 output values of the ground truth box to calculate the IoU ---
# --- Our model.py produce 2 box so thanks to this function we decide which box is the best ---
def intersection_over_union(boxes_preds, boxes_labels):
    box1_x = boxes_preds[..., 0:1]
    box1_y = boxes_preds[..., 1:2]
    box1_w = boxes_preds[..., 2:3]
    box1_h = boxes_preds[..., 3:4]

    box2_x = boxes_labels[..., 0:1]
    box2_y = boxes_labels[..., 1:2]
    box2_w = boxes_labels[..., 2:3]
    box2_h = boxes_labels[..., 3:4]

    box1_x1 = box1_x - box1_w / 2
    box1_y1 = box1_y - box1_h / 2
    box1_x2 = box1_x + box1_w / 2
    box1_y2 = box1_y + box1_h / 2

    box2_x1 = box2_x - box2_w / 2
    box2_y1 = box2_y - box2_h / 2
    box2_x2 = box2_x + box2_w / 2
    box2_y2 = box2_y + box2_h / 2
    

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs(box1_w * box1_h)
    box2_area = abs(box2_w * box2_h)
    union_area = box1_area + box2_area - intersection_area

    
    return intersection_area / (union_area + 1e-6)

# --- This function is used to perform Non Maximum Suppression on the predicted bounding boxes ---
# --- This function is not be used in the training phase but it is used in the inference phase to remove the redundant bounding boxes ---
# --- Thanks to this function we filter the bad predicted boxes and we keep only the best ones in the prediction phase ---
def non_max_suppression(bboxes, iou_threshold, threshold):
    
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes.sort(key=lambda x: x[1], reverse=True)
    
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or intersection_over_union(torch.tensor(chosen_box[2:]),torch.tensor(box[2:])) < iou_threshold]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms