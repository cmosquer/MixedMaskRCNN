import math
import time, sys
import torch

import torchvision.models.detection.mask_rcnn

from mixed_detection.coco_utils import get_coco_api_from_dataset
from mixed_detection.coco_eval import CocoEvaluator
from mixed_detection import vision_utils
from mixed_detection.utils import mean_average_precision


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', vision_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = vision_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = vision_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        #loss_value = losses_reduced.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     print(loss_dict_reduced)
        #     sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, saving_path=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    if saving_path:
        torch.save(model.state_dict(),saving_path)
        print('Saved model to ',saving_path)

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def train_one_epoch_resnet(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()

    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', vision_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    counter = 0
    train_running_loss = 0
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = vision_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, labels in metric_logger.log_every(data_loader, print_freq, header):
        counter += 1
        optimizer.zero_grad()
        print(images.shape,labels.shape)

        #images = list(image.to(device) for image in images)
        #labels = list(label.to(device) for label in labels)
        images = images.to(device)
        labels = labels.to(device)
        print(images.shape,labels.shape)

        outputs = model(images)

        loss = criterion(outputs,labels)

        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    train_loss = train_running_loss / counter


    return metric_logger,train_loss

@torch.no_grad()
def evaluate_resnet(model, dataloader, device, criterion, saving_path=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    if saving_path:
        torch.save(model.state_dict(),saving_path)
        print('Saved model to ',saving_path)
    counter = 0
    val_running_loss = 0.0
    for images, labels in metric_logger.log_every(dataloader, 100, header):
        counter += 1
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time


        loss = criterion(outputs, labels)
        val_running_loss += loss.item()
        metric_logger.update(loss=loss, evaluator_time=model_time)
    val_loss = val_running_loss / counter
    return val_loss