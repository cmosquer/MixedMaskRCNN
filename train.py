import torch
import pandas as pd
import torch.utils.data
from mixed_detection.utils import get_instance_segmentation_model,prepareDatasets,collate_fn
from mixed_detection.engine import train_one_epoch, evaluate
import os, random, psutil
import numpy as np
import wandb

def main(args=None):
    project = "mixed_mask_rcnn"

    print('starting training script')
    trx_dir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/TRx-v2/'
    experiment_dir = trx_dir+'Experiments/'

    config = {
        "batch_size": 2,
        "batch_size_valid":1,
        "initial_lr": 0.001,
        "lr_scheduler_epochs_interval": 3,
        'lr_scheduler_factor':0.1,
        'dataset': "TX-RX-ds-20210423-00_ubuntu",
        'revised_test_set' : '{}/{}'.format(experiment_dir,'test_groundtruth_validados.csv'),


        'opacityies_as_binary':True,
        'no_findings_examples_in_valid': True,
        'no_findings_examples_in_train': False,
        'max_valid_set_size':2000,
        'experiment_type':'masks_boxes',# 'masks'#
        'date': '12-05-21',
        'epochs':10,
        'random_seed': 35,

    }

    class_numbers = {
     'NoduloMasa': 1,
     'Consolidacion': 2,
     'PatronIntersticial': 3,
     'Atelectasia': 4,
     'LesionesDeLaPared': 5
     }

    if config['opacityies_as_binary']:
        num_classes = 2
    else:
        num_classes = len(class_numbers.keys())+1 #patologias + background
    pretrained_checkpoint = None #experiment_dir+'/19-03-21/maskRCNN-8.pth'
    pretrained_backbone_path = None #experiment_dir+'/17-04-21/resnetBackbone-8.pth'
    #experiment_id = '06-05-21_masksOnly'
    experiment_number = config['date']
    experiment_type = config['experiment_type']
    experiment_id = experiment_number+'_'+experiment_type
    output_dir = '{}/{}/'.format(experiment_dir,experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    config["raw_csv"] = trx_dir + 'Datasets/Opacidades/{}.csv'.format(config['dataset'])
    if config['opacityies_as_binary']:
        experiment_id+='_binary'

    prevalid = '{}/{}_masksAndBoxs_binary/testCSV.csv'.format(experiment_dir,experiment_number)
    if os.path.exists(prevalid):
        config["existing_valid_set"] = prevalid
    else:
        config["existing_valid_set"] = None

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


        # get the model using our helper function
        model = get_instance_segmentation_model(num_classes,
                                                pretrained_on_coco=True,
                                                #rpn_nms_thresh=0.5,rpn_fg_iou_thresh=0.5, #Parametros a probar
                                                pretrained_backbone=pretrained_backbone_path,
                                                #trainable_layers=0
                                                )
        if pretrained_checkpoint is not None:
            model.load_state_dict(torch.load(pretrained_checkpoint))

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config.initial_lr,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # a gamma factor every interval of N epochs

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=config.lr_scheduler_epochs_interval,
                                                       gamma=config.lr_scheduler_factor)

        for epoch in range(config.epochs):
            print('Memory when starting epoch: ', psutil.virtual_memory().percent)
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=500,#breaking_step=40,
                            )
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            saving_path = '{}/mixedMaskRCNN-{}.pth'.format(output_dir,epoch) #None#
            results_coco_file = '{}/cocoStats-{}.txt'.format(output_dir,epoch)
            evaluate(model, data_loader_valid, device=device, model_saving_path=saving_path,results_file=results_coco_file,
                     coco=True,dice=False,classification=True)
            #evaluate(model, data_loader_test, device=device, results_file=results_coco_file, coco=False,dice=True)


if __name__ == '__main__':
    main()
