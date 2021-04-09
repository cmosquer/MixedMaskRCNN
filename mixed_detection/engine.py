import math
import time
import torch

import torchvision.models.detection.mask_rcnn

from mixed_detection.coco_utils import get_coco_api_from_dataset
from mixed_detection.coco_eval import CocoEvaluator
from mixed_detection import vision_utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,loss_type_weights=None,breaking_n=0):
    model.train()
    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', vision_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = vision_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    n=0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        if n > breaking_n and breaking_n!=0:
            break
        n+=1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print('processing')
        loss_dict = model(images, targets)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = vision_utils.reduce_dict(loss_dict)
        #print(loss_dict)

        if loss_type_weights is not None:
            n_losses_weighted = sum([loss_type_weights[curr_loss_type] for curr_loss_type in loss_dict.keys()])
            if int(n_losses_weighted) == 0:
                print('zero losses!!')
                continue
            else:
                losses = sum(loss*loss_type_weights[loss_name] for loss_name,loss in loss_dict.items())/n_losses_weighted
                losses_reduced = sum(loss* loss_type_weights[loss_name] for loss_name, loss in loss_dict_reduced.items())/n_losses_weighted
            #print('Weighted n losses: ',n_losses)


        else:
            n_losses = len(loss_dict.keys())
            losses = sum(loss for loss in loss_dict.values())/n_losses
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())/n_losses
            #print('unweighted n losses: ',n_losses)
        loss_value = losses_reduced.item()
        if not losses.requires_grad:
            print(loss_dict)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            #sys.exit(1)
        else:
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])


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

    if saving_path:
        torch.save(model.state_dict(),saving_path)
        print('Saved model to ',saving_path)

    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

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
