from mixed_detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from mixed_detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn
from mixed_detection import vision_transforms as T
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve,brier_score_loss
from sklearn.metrics import auc as sklearnAUC
from mixed_detection.MixedLabelsDataset import MixedLabelsDataset #, MixedSampler

from collections import Counter
import os
import random
import numpy as np
import torch
from torch import Tensor
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from torchvision.ops import box_iou
except ModuleNotFoundError:  # pragma: no-cover
    box_iou = None

def prepareDatasets(config,output_dir,class_numbers,train_transform=None):

    raw_csv = pd.read_csv(config["raw_csv"])

    if config["revised_test_set"]:
        csv_revised = pd.read_csv(config["revised_test_set"])
        revised_test_idx = list(set(csv_revised.file_name.values))
        inter = set(revised_test_idx).intersection(set(raw_csv['file_name'].values))
        print('interseccion csv total con test revisado ',len(inter))
        L = len(raw_csv)
        raw_csv = raw_csv[~raw_csv.file_name.isin(list(revised_test_idx))].reset_index(drop=True)
        print('Len of original raw csv: {}, len after removing intersection with revised test set: {}'.format(L,len(raw_csv)))

    print('RAW CSV DESCRIPTION:')
    print(raw_csv.image_source.value_counts())
    print(raw_csv.label_level.value_counts())

    if 'label_level' not in raw_csv.columns:
        raw_csv['label_level'] = [None] * len(raw_csv)
        for i, row in raw_csv.iterrows():
            assigned = False
            if isinstance(row['mask_path'], str):
                if os.path.exists(row['mask_path']):
                    raw_csv.loc[i, 'label_level'] = 'mask'
                    assigned = True
            else:
                xmin = row['x1']
                xmax = row['x2']
                ymin = row['y1']
                ymax = row['y2']
                if ymax > ymin and xmax > xmin:
                    raw_csv.loc[i, 'label_level'] = 'box'
                    assigned = True
                else:
                    if isinstance(row['class_name'], str):
                        if len(row['class_name']) > 0:
                            raw_csv.loc[i, 'label_level'] = 'imagelabel'
                            assigned = True
            if not assigned:
                raw_csv.loc[i, 'label_level'] = 'nofinding'
        print('finished initialization: ')
        print(raw_csv.label_level.value_counts())
        #raw_csv.to_csv(trx_dir + f'Datasets/Opacidades/TX-RX-ds-{experiment_id}.csv', index=False)

    # --Only accept images with boxes or masks--#
    validAnnotations = []

    if 'mask' in config["experiment_type"].lower() or config['masks_as_boxes']:
        validAnnotations.append('mask')
    if 'box' in config["experiment_type"].lower():
        validAnnotations.append('box')


    csv = raw_csv[raw_csv.label_level.isin(validAnnotations)].reset_index(drop=True)
    print('Len of csv after keeping only {} annotations: {}'.format(validAnnotations,len(csv)))
    if config["existing_valid_set"]:
        csv_valid = pd.read_csv(config["existing_valid_set"])
        valid_idx = list(set(csv_valid.file_name.values))
        csv_train = csv[~csv.file_name.isin(valid_idx)].reset_index(drop=True)
        train_idx = list(set(csv_train.file_name.values))
    else:
        print('Creating new validation set ... ')

        image_ids = list(set(csv.file_name.values))
        class_series = pd.Series([clss.split('-')[0] for clss in csv['class_name'].values])
        csv['stratification'] = csv['image_source'].astype(str)+'_'+csv['label_level'].astype(str)+'_'+class_series
        stratification = [csv[csv.file_name == idx]['stratification'].values[0] for idx in image_ids]

        train_idx, valid_idx = train_test_split(image_ids, stratify=stratification,
                                               test_size=0.05,
                                               random_state=config["random_seed"])
        csv_train = csv[csv.file_name.isin(list(train_idx))].reset_index(drop=True)
        csv_valid = csv[csv.file_name.isin(list(valid_idx))].reset_index(drop=True)
        if config["no_findings_examples_in_valid"]:
            print('Len before appending no finding to valid set: {}'.format(len(csv_valid)))
            if isinstance(config["max_valid_set_size"],int):
                nofindings = raw_csv[raw_csv.label_level=='nofinding'].reset_index(drop=True)[:(config["max_valid_set_size"]-len(csv_valid))]
            else:
                if config["max_valid_set_size"]=='balanced':
                    current_0 = len(csv_valid[csv_valid.label_level=='nofinding'])
                    current_1 = len(csv_valid) - current_0
                    required_0 = current_1 - current_0
                    print('Adding {} no finding images'.format(required_0))
                    nofindings = raw_csv[raw_csv.label_level == 'nofinding'].reset_index(drop=True)[
                                 :required_0]

            csv_valid = csv_valid.append(nofindings,ignore_index=True).reset_index(drop=True)
            print('Len AFTER appending no finding to valid set: {}'.format(len(csv_valid)))
    assert len(set(csv_train.file_name).intersection(csv_valid.file_name)) == 0

    print('Len csv train: {}, len csv test: {}'.format(len(csv_train),len(csv_valid)))
    print('TRAIN SOURCES:')
    print(csv_train.image_source.value_counts(normalize=True))
    print(csv_train.label_level.value_counts(normalize=True))

    print('VALID SOURCES')
    print(csv_valid.image_source.value_counts(normalize=True))
    print(csv_valid.label_level.value_counts(normalize=True))

    """
    csv_train = csv[:30000].reset_index()
    csv_test = csv[30000:].reset_index() """
    csv_train.to_csv('{}/trainCSV.csv'.format(output_dir),index=False)
    csv_valid.to_csv('{}/testCSV.csv'.format(output_dir),index=False)
    dataset = MixedLabelsDataset(csv_train, class_numbers,get_transform(train=True), #transforms=train_transform,
                                 binary_opacity=config['opacityies_as_binary'],
                                 masks_as_boxes=config['masks_as_boxes'])
    dataset_valid = MixedLabelsDataset(csv_valid, class_numbers, get_transform(train=False),
                                       binary_opacity=config['opacityies_as_binary'],
                                       masks_as_boxes=config['masks_as_boxes'])
    print('TRAIN:')
    dataset.quantifyClasses()
    print('\nVALID:')
    dataset_valid.quantifyClasses()
    return dataset,dataset_valid

def process_output(outputs,total_area,min_score_threshold=0.1,min_box_proportionArea=1/20,max_detections=6):

    areas = []
    for x1, y1, x2,y2 in outputs['boxes']:
        area = (int(x2 - x1) * int(y2 - y1))/total_area
        # print(x1, x2, y1, y2,'-->',area)
        areas.append(area)
    outputs['areas'] = torch.as_tensor(areas, dtype=torch.float32)

    if min_box_proportionArea:
        bigBoxes = np.argwhere(outputs['areas'] > min_box_proportionArea).flatten()
        for k, v in outputs.items():
            outputs[k] = outputs[k][bigBoxes,]
    if isinstance(min_score_threshold, float):
        high_scores = np.argwhere(outputs['scores'] > min_score_threshold).flatten()
        for k, v in outputs.items():
            outputs[k] = outputs[k][high_scores,]
    if isinstance(min_score_threshold, dict):
        valid_detections_idx = np.array([], dtype=np.int64)
        for clss_idx, th in min_score_threshold.items():
            idxs_clss = np.argwhere(outputs['labels'] == clss_idx).flatten()
            print('class id: ', clss_idx, 'idxs clss', idxs_clss)
            if idxs_clss.shape[0] > 0:
                print(idxs_clss)
                high_scores = np.argwhere(outputs['scores'][idxs_clss] > th).flatten()
                print('idxs high scores', high_scores)
                if high_scores.shape[0] > 0:
                    print('th', th, 'allclassscores',
                          outputs['scores'][idxs_clss],
                          'highscores idx', high_scores)
                    valid_detections_idx = np.concatenate((valid_detections_idx, high_scores)).astype(np.int64)
        # valid_detections_idx = list(dict.fromkeys(valid_detections_idx))
        print('final idxs', valid_detections_idx)
        for k, v in outputs.items():
            outputs[k] = outputs[k][valid_detections_idx,]
        print('after min th\n', outputs)
    if max_detections:
        scores_sort = np.argsort(-outputs['scores'])[:max_detections]
        for k, v in outputs.items():
            outputs[k] = outputs[k][scores_sort,]
    return outputs


def update_regression_features(image_scores,image_areas):
    n_features = 7
    x_regresion_j = np.zeros(n_features)

    if image_scores is not None:
        N_detections = len(image_scores)
        if N_detections > 0:
            x_regresion_j[0] = torch.mean(image_scores)
            x_regresion_j[1] = torch.max(image_scores)
            x_regresion_j[2] = torch.min(image_scores)
            x_regresion_j[3] = torch.mean(image_areas)
            x_regresion_j[4] = torch.max(image_areas)
            x_regresion_j[5] = torch.min(image_areas)
            x_regresion_j[6] = N_detections
    return x_regresion_j


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

        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_object_detection_model(num_classes, pretrained_backbone=None, pretrained_on_coco=False,**kwargs):
    # load an instance segmentation model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained_coco=pretrained_on_coco,
                                  pretrained_backbone_checkpoint=pretrained_backbone,**kwargs)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    return model

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


def getClassificationMetrics(preds, labels_test, print_results=True,cont_preds=None):
    if cont_preds is None:
        cont_preds = preds.copy()
    try:
        fpr, tpr, th = roc_curve(labels_test, cont_preds)
        aucroc = roc_auc_score(labels_test, cont_preds)
        print('AUCROC: {:.3f}'.format(aucroc))
    except Exception as e:
        print('couldnt calculate roc curve because error: {}'.format(e))
        aucroc = np.nan

    try:
        precision, recall, thresholds = precision_recall_curve(labels_test, cont_preds)
        aucpr = sklearnAUC(recall, precision)
        print('AUCPR: {:.3f}'.format(aucpr))
    except Exception as e:
        print('couldnt calculate PR curve because error: {}'.format(e))
        aucpr = np.nan
    c = confusion_matrix(labels_test, preds).ravel()
    print('Confusion matrix: ',c)
    (tn, fp, fn, tp ) = c
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    acc = (tp + tn) / (tn + fp + fn + tp)
    f1score = 2 * ppv * sens / (ppv + sens)

    #TODO agregar brier score + y -
    positive_labels = labels_test[labels_test == 1]
    Npos = len(positive_labels)
    positive_preds = cont_preds[labels_test == 1]

    negative_labels = labels_test[labels_test == 0]
    negative_preds = cont_preds[labels_test == 0]

    assert len(positive_labels) + len(negative_labels) == len(labels_test)

    brierPos = brier_score_loss(positive_labels, positive_preds)
    brierNeg = brier_score_loss(negative_labels, negative_preds)
    brier = brier_score_loss(labels_test, preds)

    if print_results:
        print(f'True negatives:{tn}\nTrue positives:{tp}\nFalse negatives:{fn}\nFalse positives:{fp}')
        print(f'\nSensitivity(recall):{sens:.2f}\nSpecificity:{spec:.2f}')
        print(f'PPV(precision):{ppv:.2f}\nNPV:{npv:.2f}\n')

        print('f1-score:{:.3f}'.format(f1score))
        print('accuracy:{:.3f}'.format(acc))
        print('brier {:.3f}. brier+ {:.3f}. brier- {:.3f}'.format(brier,brierPos,brierNeg))

    return (tn, fp, fn, tp), (sens, spec, ppv, npv), (acc, f1score, aucroc,aucpr), (brier,brierPos,brierNeg)