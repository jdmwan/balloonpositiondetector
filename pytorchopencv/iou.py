import torch.nn.functional as F

def calculate_iou(pred_box, true_box):
    # Convert from [x, y, w, h] → [x1, y1, x2, y2]
    pred_x1 = pred_box[0]
    pred_y1 = pred_box[1]
    pred_x2 = pred_box[0] + pred_box[2]
    pred_y2 = pred_box[1] + pred_box[3]

    true_x1 = true_box[0]
    true_y1 = true_box[1]
    true_x2 = true_box[0] + true_box[2]
    true_y2 = true_box[1] + true_box[3]

    # Calculate intersection
    inter_x1 = max(pred_x1, true_x1)
    inter_y1 = max(pred_y1, true_y1)
    inter_x2 = min(pred_x2, true_x2)
    inter_y2 = min(pred_y2, true_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
    union_area = pred_area + true_area - inter_area

    # Return IoU
    if union_area == 0:
        return 0
    return inter_area / union_area
