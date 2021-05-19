from mixed_detection.faster_rcnn import FastRCNNPredictor
from mixed_detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn
from mixed_detection import vision_transforms as T
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score

from collections import Counter
import os
import random
import numpy as np
import torch
from torch import Tensor

try:
    from torchvision.ops import box_iou
except ModuleNotFoundError:  # pragma: no-cover
    box_iou = None
def seed_all(seed=27):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def mean_average_precision(
    pred_image_indices: Tensor,
    pred_probs: Tensor,
    pred_labels: Tensor,
    pred_bboxes: Tensor,
    target_image_indices: Tensor,
    target_labels: Tensor,
    target_bboxes: Tensor,
    iou_threshold: float,
    ap_calculation: str
) -> Tensor:
    """
    Compute mean average precision for object detection task
    Args:
        pred_image_indices: an (N,)-shaped Tensor of image indices of the predictions
        pred_probs: an (N,)-shaped Tensor of probabilities of the predictions
        pred_labels: an (N,)-shaped Tensor of predicted labels
        pred_bboxes: an (N, 4)-shaped Tensor of predicted bounding boxes
        target_image_indices: an (M,)-shaped Tensor of image indices of the groudn truths
        target_labels: an (M,)-shaped Tensor of ground truth labels
        target_bboxes: an (M, 4)-shaped Tensor of ground truth bounding boxes
        iou_threshold: threshold for IoU score for determining true positive and
                       false positive predictions.
        ap_calculation: method to calculate the average precision of the precision-recall curve
            - ``'step'``: calculate the step function integral, the same way as
            :func:`~pytorch_lightning.metrics.functional.average_precision.average_precision`
            - ``'VOC2007'``: calculate the 11-point sampling of interpolation of the precision recall curve
            - ``'VOC2010'``: calculate the step function integral of the interpolated precision recall curve
            - ``'COCO'``: calculate the 101-point sampling of the interpolated precision recall curve
    Returns:
        mean of the average precisions of all classes in object detection task.
    References:
        - host.robots.ox.ac.uk/pascal/VOC/
        - https://ccc.inaoep.mx/~villasen/bib/AN%20OVERVIEW%20OF%20EVALUATION%20METHODS%20IN%20TREC%20AD%20HOC%20IR%20AND%20TREC%20QA.pdf#page=15
        - https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    """
    if box_iou is None:
        raise ImportError('`mean_average_precision` metric requires `torchvision`, which is not installed. '
                          ' install it with `pip install torchvision`.')
    classes = torch.cat([pred_labels, target_labels]).unique()
    average_precisions = torch.zeros(len(classes))
    for class_idx, c in enumerate(classes):
        # Descending indices w.r.t. class probability for class c
        desc_indices = torch.argsort(pred_probs, descending=True)[pred_labels == c]
        # No predictions for this class so average precision is 0
        if len(desc_indices) == 0:
            continue
        targets_per_images = Counter([idx.item() for idx in target_image_indices[target_labels == c]])
        targets_assigned = {
            image_idx: torch.zeros(count, dtype=torch.bool) for image_idx, count in targets_per_images.items()
        }
        tps = torch.zeros(len(desc_indices))
        fps = torch.zeros(len(desc_indices))
        for i, pred_idx in enumerate(desc_indices):
            image_idx = pred_image_indices[pred_idx].item()
            # Get the ground truth bboxes of class c and the same image index as the prediction
            gt_bboxes = target_bboxes[(target_image_indices == image_idx) & (target_labels == c)]
            ious = box_iou(torch.unsqueeze(pred_bboxes[pred_idx], dim=0), gt_bboxes)
            best_iou, best_target_idx = ious.squeeze(0).max(0) if len(gt_bboxes) > 0 else (0, -1)
            # Prediction is a true positive is the IoU score is greater than the threshold and the
            # corresponding ground truth has only one prediction assigned to it
            if best_iou > iou_threshold and not targets_assigned[image_idx][best_target_idx]:
                targets_assigned[image_idx][best_target_idx] = True
                tps[i] = 1
            else:
                fps[i] = 1
        tps_cum, fps_cum = torch.cumsum(tps, dim=0), torch.cumsum(fps, dim=0)
        precision = tps_cum / (tps_cum + fps_cum)
        num_targets = len(target_labels[target_labels == c])
        recall = tps_cum / num_targets if num_targets else tps_cum
        precision = torch.cat([reversed(precision), torch.tensor([1.])])
        recall = torch.cat([reversed(recall), torch.tensor([0.])])
        if ap_calculation == "step":
            average_precision = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
        elif ap_calculation == "VOC2007":
            average_precision = 0
            recall_thresholds = torch.linspace(0, 1, 11)
            for threshold in recall_thresholds:
                points = precision[:-1][recall[:-1] >= threshold]
                average_precision += torch.max(points) / 11 if len(points) else 0
        elif ap_calculation == "VOC2010":
            average_precision = 0
            for i in range(len(precision)):
                precision[i] = torch.max(precision[:i + 1])
            average_precision = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
        elif ap_calculation == "COCO":
            average_precision = 0
            recall_thresholds = torch.linspace(0, 1, 101)
            for threshold in recall_thresholds:
                points = precision[:-1][recall[:-1] >= threshold]
                average_precision += torch.max(points) / 101 if len(points) else 0
        else:
            raise NotImplementedError(f"'{ap_calculation}' is not supported.")
        average_precisions[class_idx] = average_precision
    mean_average_precision = torch.mean(average_precisions)
    return mean_average_precision, average_precisions

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_instance_segmentation_model(num_classes, pretrained_backbone=None, pretrained_on_coco=False,**kwargs):
    # load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(pretrained_coco=pretrained_on_coco,
                                  pretrained_backbone_checkpoint=pretrained_backbone,**kwargs)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def getClassificationMetrics(preds, labels_test, print_results=True):
    labels_test_num = [0 if l == 'B' else 1 for l in labels_test]
    preds_num = [0 if p == 'B' else 1 for p in preds]
    try:
        fpr, tpr, th = roc_curve(labels_test_num, preds_num)
        auc = roc_auc_score(labels_test_num, preds_num)
        print('AUC: {:.3f}'.format(auc))
    except Exception as e:
        print('could calculate roc curve because error: {}'.format(e))
        auc = np.nan
    tn, fp, fn, tp = confusion_matrix(labels_test_num, preds_num).ravel()

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    acc = (tp + tn) / (tn + fp + fn + tp)
    f1score = 2 * ppv * sens / (ppv + sens)
    if print_results:
        print(f'True negatives:{tn}\nTrue positives:{tp}\nFalse negatives:{fn}\nFalse positives:{fp}')
        print(f'\nSensitivity(recall):{sens:.2f}\nSpecificity:{spec:.2f}')
        print(f'PPV(precision):{ppv:.2f}\nNPV:{npv:.2f}\n')

        print('f1-score:{:.3f}'.format(f1score))
        print('accuracy:{:.3f}'.format(acc))
    else:
        return (tn, fp, fn, tp), (sens, spec, ppv, npv), (acc, f1score, auc)