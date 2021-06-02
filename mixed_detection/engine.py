import math
import time, sys,pickle
import torch
import torchvision.models.detection.mask_rcnn
import numpy as np
from mixed_detection.coco_utils import get_coco_api_from_dataset
from mixed_detection.coco_eval import CocoEvaluator
from mixed_detection import vision_utils
from mixed_detection.utils import getClassificationMetrics
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import psutil
import wandb

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,breaking_step=None,wandb_interval=200):


    model.train()

    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', vision_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = vision_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    step = 0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        #print('Memory before loading to GPU the batch: %', psutil.virtual_memory().percent)

        if breaking_step:
            if step > breaking_step:
                break
        step += 1
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

        if step % wandb_interval == 0:
            wandbdict=loss_dict.copy()
            wandbdict['epoch']=epoch
            wandbdict['total_loss']=losses
            wandb.log(wandbdict)

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
def evaluate_coco(model, data_loader, device, results_file=None, use_cpu=False):
    print('STARTING VALIDATION')
    # FIXME remove this and make paste_masks_in_image run on the GPU
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    if use_cpu:
        cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    #iou_types = _get_iou_types(model)
    iou_types = ["bbox","segm"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    leave = False
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        if psutil.virtual_memory().percent>80:
            leave=True
            break
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        if use_cpu:
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    if not leave:
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()


        # accumulate predictions from all images
        coco_evaluator.accumulate()
        results_dict = coco_evaluator.summarize(saving_file_path=results_file)
        torch.set_num_threads(n_threads)

        return results_dict
    else:
        return {'memory_reached':psutil.virtual_memory().percent}
@torch.no_grad()
def evaluate_classification(model, data_loader, device, results_file=None, test_clf=None,log_wandb=True):
    print('STARTING VALIDATION')
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    header = 'Test:'


    x_regresion = np.zeros((len(data_loader.dataset),2))
    y_regresion = np.zeros(len(data_loader.dataset))
    j = 0
    leave=False
    with torch.no_grad():
        #print('initial',psutil.virtual_memory().percent)
        for images, targets in metric_logger.log_every(data_loader, 5, header):
            if psutil.virtual_memory().percent > 80:
                leave = True
                break
            images = list(img.to(device) for img in images)
            #print('img loaded before inference',psutil.virtual_memory().percent)
            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)
            #print('after inference',psutil.virtual_memory().percent)
            outputs = [{k: v.to(cpu_device).detach() for k, v in t.items()} for t in outputs]
            targets = [{k: v.to(cpu_device).detach() for k, v in t.items()} for t in targets]
            model_time = time.time() - model_time
            evaluator_time = time.time()

            for img_id,output in enumerate(outputs):
                #print('beofre target',psutil.virtual_memory().percent)
                target = targets[img_id]
                N_targets = len(target['boxes'].detach().numpy())
                gt = 1 if N_targets > 0 else 0
                # y_regresion.append(gt)
                y_regresion[j] = gt

                #print('before scores',psutil.virtual_memory().percent)
                image_scores = output['scores'].detach().numpy()

                if image_scores is not None:
                    if len(image_scores)>0:
                        score_mean = np.mean(image_scores)
                        score_max = np.max(image_scores)
                        x_regresion[j,0] = score_mean
                        x_regresion[j,1] = score_max
                        del score_mean,score_max
                    #x_regresion.append([score_mean,score_max])
                #else:
                    #x_regresion.append([0, 0])

                j += 1
                #print('before del',psutil.virtual_memory().percent)
                del gt,image_scores,target
                #print('after del',psutil.virtual_memory().percent)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
            del images,targets,outputs
    if not leave:
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        if test_clf: #Testear con un regresor ya ajustado

            preds = test_clf.predict(x_regresion)
            print(pd.Series(preds).value_counts())
            y_test =y_regresion
            if results_file:
                with open(results_file.replace('cocoStats', 'test_classification_data').replace('.txt', ''), 'wb') as f:
                    classification_data = {'x_test': x_regresion, 'y_test': y_test,
                                           'preds_test': preds, 'clf': test_clf}
                    pickle.dump(classification_data, f)
        else:

            x_train,x_test, y_train, y_test = train_test_split(x_regresion, y_regresion, stratify=y_regresion,
                                                    test_size=0.2,
                                                    random_state=32)
            clf = LogisticRegression(random_state=32).fit(x_train, y_train)
            print(pd.Series(y_regresion).value_counts())
            print(pd.Series(y_train).value_counts())
            print(pd.Series(y_test).value_counts())

            preds = clf.predict(x_test)

            print(pd.Series(preds).value_counts())
            if results_file:
                with open(results_file.replace('cocoStats', 'classification_data').replace('.txt', ''), 'wb') as f:
                    classification_data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
                                           'preds_test': preds, 'clf': clf}
                    pickle.dump(classification_data, f)
        (tn, fp, fn, tp), (sens, spec, ppv, npv), (acc, f1score, aucroc,aucpr) = getClassificationMetrics(preds, y_test)
        classif_dict = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                        'sens':sens, 'spec':spec, 'ppv':ppv, 'npv':npv,
                        'acc':acc, 'f1':f1score, 'aucroc':aucroc, 'aucpr':aucpr}

        if results_file:
            if log_wandb:
                wandb.log({'results_file':results_file})
            with open(results_file, 'a') as f:
                f.write('\nClassification metrics\n')
                for k, v in classif_dict.items():
                    f.write("{}: {:.4f}\n".format(k.upper(), v))
        return classif_dict
    else:
        return {'memory_reached':psutil.virtual_memory().percent}

@torch.no_grad()
def evaluate_dice(model, data_loader, device, results_file=None):
    print('STARTING VALIDATION')
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = vision_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if data_loader.dataset.binary:
        total_dice = []
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            outputs = [{k: v.detach().to(cpu_device) for k, v in t.items()} for t in outputs]
            dice = 0
            for output, target in zip(outputs, targets):
                output_all = torch.squeeze((torch.sum(output['masks'], axis=0) > 0).int())
                target_all = (torch.sum(target['masks'], axis=0) > 0).int()
                area_gt = torch.sum(target['masks'])
                area_det = torch.sum(output_all)
                if area_gt > 0:
                    intersection = torch.sum(output_all[target_all.bool()])
                    dice_tensor = intersection * 2. / (area_gt + area_det)
                    dice = dice_tensor.item()
                    del dice_tensor, intersection
                else:
                    dice = 0
                del output_all, target_all, area_det, area_gt
            total_dice.append(dice)
            del images, targets, outputs

        dice_avg = np.mean(total_dice)

        print('AVG DICE {:.5f}'.format(dice_avg))
        if results_file:
            with open(results_file, 'a') as f:
                f.write(f'DICE: {dice_avg}')

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