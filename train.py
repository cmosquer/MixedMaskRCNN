import torch
import pandas as pd
import torch.utils.data
from mixed_detection.utils import get_instance_segmentation_model,prepareDatasets,collate_fn, get_object_detection_model
from mixed_detection.engine import train_one_epoch, evaluate_coco,evaluate_classification,evaluate_dice
import os, random, psutil
import numpy as np
import wandb
from datetime import datetime
import albumentations as A

def main(args=None):
    project = "mixed_mask_rcnn"

    print('starting training script')
    trx_dir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/TRx-v2/'
    experiment_dir = trx_dir+'Experiments/'

    config = {
        "batch_size": 4,
        "batch_size_valid":1,
        "initial_lr": 0.01,
        "lr_scheduler_epochs_interval": 3,
        'lr_scheduler_factor':0.1,
        'dataset': "TX-RX-ds-20210625-00_ubuntu",
        'revised_test_set' : '{}/{}'.format(experiment_dir,'test_groundtruth_validados.csv'),
        'unfreeze_only_mask': False,
        'data_augmentation': True,
        'existing_valid_set': '{}/{}'.format(experiment_dir,'test_groundtruth_validados.csv'),#None,#'{}/2021-06-16_boxes_binary/testCSV_debug.csv'.format(experiment_dir),
        'opacityies_as_binary':True,
        'no_findings_examples_in_valid': False,
        'no_findings_examples_in_train': 0.5,#None,
        'max_valid_set_size': 1000,
        'masks_as_boxes': True,
        'experiment_type': 'boxes',
        'date': datetime.today().strftime('%Y-%m-%d'),
        'epochs': 4,
        'random_seed': 40,
        'pretrained_checkpoint': experiment_dir + '/2021-07-05_binary/mixedMaskRCNN-0.pth',
        'pretrained_backbone_path': None #experiment_dir + '/17-04-21/resnetBackbone-8.pth',
    }

    class_numbers = {
     'NoduloMasa': 1,
     'Consolidacion': 2,
     'PatronIntersticial': 3,
     'Atelectasia': 4,
     'LesionesDeLaPared': 5,
     'Covid_Typical_Appearance':6,
        'Covid_Indeterminate_Appearance':7,
        'Covid_Atypical_Appearance':8,

     }

    if config['opacityies_as_binary']:
        num_classes = 2
    else:
        num_classes = len(class_numbers.keys())+1 #patologias + background
    pretrained_checkpoint = config['pretrained_checkpoint']
    pretrained_backbone_path = config['pretrained_backbone_path']
    experiment_id = config['date']
    if config['opacityies_as_binary']:
        experiment_id+='_binary'
    output_dir = '{}/{}/'.format(experiment_dir,experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    config["raw_csv"] = trx_dir + 'Datasets/Opacidades/{}.csv'.format(config['dataset'])
    dataset,dataset_valid = prepareDatasets(config,class_numbers=class_numbers,output_dir=output_dir)


    with wandb.init(config=config, project=project, name=experiment_id):
        config=wandb.config


        # split the dataset in train and test set
        torch.manual_seed(1)

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
            collate_fn=collate_fn,
            #sampler=train_sampler
             )

        data_loader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=config.batch_size_valid, shuffle=False, num_workers=0,
            collate_fn=collate_fn)

        print('N train: {}. N test: {}'.format(len(data_loader),len(data_loader_valid)))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if config['experiment_type']=='boxes':
            # get the model using our helper function
            model = get_object_detection_model(num_classes,
                                                pretrained_on_coco=False,
                                                # rpn_nms_thresh=0.5,rpn_fg_iou_thresh=0.5, #Parametros a probar
                                                pretrained_backbone=pretrained_backbone_path,
                                                # trainable_layers=0
                                                )
        if config['experiment_type']=='masks':
            # get the model using our helper function
            model = get_instance_segmentation_model(num_classes,
                                                    pretrained_on_coco=True,
                                                    #rpn_nms_thresh=0.5,rpn_fg_iou_thresh=0.5, #Parametros a probar
                                                    pretrained_backbone=pretrained_backbone_path,
                                                    #trainable_layers=0
                                                    )
        if pretrained_checkpoint is not None:
            print('Loading pretrained checkpoint ...')
            model.load_state_dict(torch.load(pretrained_checkpoint))
        print(model)
        # move model to the right device
        model.to(device)
        if config["unfreeze_only_mask"]:
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.roi_heads.box_roi_pool.parameters():
                param.requires_grad = False
            for param in model.roi_heads.box_head.parameters():
                param.requires_grad = False
            for param in model.roi_heads.box_predictor.parameters():
                param.requires_grad = False
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config.initial_lr,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # a gamma factor every interval of N epochs

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=config.lr_scheduler_epochs_interval,
                                                       gamma=config.lr_scheduler_factor)

        interval_steps = int(len(data_loader)/20)
        print('Wandb logging after {} steps'.format(interval_steps))
        wandb.watch(model, optimizer, log_freq=interval_steps)
        for epoch in range(config.epochs):
            print('Memory when starting epoch: ', psutil.virtual_memory().percent)
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=500,#breaking_step=10,
                            wandb_interval=interval_steps
                            )

            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            saving_path = '{}/mixedMaskRCNN-{}.pth'.format(output_dir,epoch) #None#

            if saving_path:
                torch.save(model.state_dict(), saving_path)
                print('Saved model to ', saving_path)
            wandb_valid = {'epoch': epoch}
            if config['experiment_type']=='masks':
                iou_types = ["bbox", "segm"]
            else:
                iou_types = ["bbox"]
            results_coco_file = '{}/cocoStats-{}.txt'.format(output_dir,epoch)
            results_coco = evaluate_coco(model, data_loader_valid, device=device,
                                         results_file=results_coco_file,use_cpu=True, iou_types=iou_types)
            wandb_valid.update(results_coco)

            results_classif = evaluate_classification(model, data_loader_valid, device=device, results_file=results_coco_file)
            wandb_valid.update(results_classif)

            #evaluate_dice(model, data_loader_valid, device=device, results_file=results_coco_file)



            wandb.log(wandb_valid)


if __name__ == '__main__':
    main()
