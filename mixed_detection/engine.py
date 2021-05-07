import math
import time, sys
import torch
import torchvision.models.detection.mask_rcnn
import numpy as np
from mixed_detection.coco_utils import get_coco_api_from_dataset
from mixed_detection.coco_eval import CocoEvaluator
from mixed_detection import vision_utils
from mixed_detection.utils import mean_average_precision
from tqdm import tqdm


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
    k=0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        if k>20:
            break
        else:
            k+=1
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
def evaluate(model, data_loader, device, model_saving_path=None, results_file=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    if model_saving_path:
        torch.save(model.state_dict(),model_saving_path)
        print('Saved model to ',model_saving_path)

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    if "segm" not in iou_types:
        iou_types.append("segm")
    coco_evaluator = CocoEvaluator(coco, iou_types)
    total_dice = []
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        if data_loader.dataset.binary:
            dice = 0
            for output,target in zip(outputs,targets):
                output_all = torch.squeeze((torch.sum(output['masks'],axis=0)>0).int())
                target_all = (torch.sum(target['masks'],axis=0)>0).int()
                area_gt = torch.sum(target['masks'])
                area_det = torch.sum(output_all)
                if area_gt>0:
                    intersection = torch.sum(output_all[target_all.bool()])
                    dice = intersection * 2. / (area_gt + area_det)
                else:
                    dice = 0
            print(type(dice),dice)
            total_dice.append(dice)
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
    coco_evaluator.summarize(saving_file_path=results_file)
    dice_avg = torch.mean(total_dice)
    print('AVG DICE {:.2f}'.format(dice_avg))
    if results_file:
        with open(results_file, 'w') as f:
            f.write(f'DICE: {dice_avg}')
    torch.set_num_threads(n_threads)
    return coco_evaluator


def train_one_epoch_resnet(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()

    #metric_logger = vision_utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', vision_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    counter = 0

    train_running_loss = 0
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = vision_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, labels in tqdm(data_loader):
        counter += 1
        optimizer.zero_grad()
        images = images.to(device)
        labels=labels.to(device)
        #print(images,labels)
        #print(type(images[0]))
        #images = torch.tensor(images,dtype=torch.float32,device=device)
        #labels = torch.tensor(labels,device=device)

        #images = list(image.to(device) for image in images)
        #labels = list(label.to(device) for label in labels)
        #images = tuple(image.to(device) for image in images)
        #labels = tuple(label.to(device) for label in labels)
        #print(images.shape,labels.shape)

        outputs = model(images)
        labels = labels.type_as(outputs)

        loss = criterion(outputs,labels)

        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        if counter%10==0:
            currloss = train_running_loss/counter
            print(f'{counter} step -- Loss: {currloss}')
        #metric_logger.update(loss=loss)
        #metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    lr = optimizer.param_groups[0]["lr"]
    train_loss = train_running_loss / counter
    print(f'Epoch {epoch} - loss {train_loss} - lr {lr}')

    return train_loss

@torch.no_grad()
def evaluate_resnet(model, dataloader, device, criterion, model_saving_path=None, num_classes=5):
    torch.set_num_threads(1)
    model.eval()
    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    if model_saving_path:
        torch.save(model.state_dict(),model_saving_path)
        print('Saved model to ',model_saving_path)
    counter = 0
    val_running_loss = 0.0

    val_runninng_loss_per_class = torch.zeros(num_classes,device=device,dtype=torch.float32)
    for images, labels in tqdm(dataloader):
        counter += 1
        images = images.to(device)
        labels=labels.to(device)
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        labels = labels.type_as(outputs)

        #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        for c in range(outputs.shape[1]): #Cantidad de classes
            class_outputs,class_labels = outputs[:, c], labels[:, c]
            current_loss_class = criterion(class_outputs,class_labels)
            val_runninng_loss_per_class[c] += current_loss_class
        loss = criterion(outputs, labels)
        val_running_loss += loss.item()
        metric_logger.update(loss=loss, evaluator_time=model_time)
    val_loss = val_running_loss / counter
    val_loss_per_class = val_runninng_loss_per_class / counter

    return val_loss, val_loss_per_class