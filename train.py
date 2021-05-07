import torch
import pandas as pd
import torch.utils.data
from mixed_detection.utils import get_instance_segmentation_model,get_transform,collate_fn
from mixed_detection.engine import train_one_epoch, evaluate
from sklearn.model_selection import train_test_split
from mixed_detection.MixedLabelsDataset import MixedLabelsDataset #, MixedSampler
import os, random
import numpy as np

def main(args=None):
    print('starting training script')
    trx_dir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/TRx-v2/'
    experiment_dir = trx_dir+'Experiments/'
    csv = pd.read_csv(trx_dir+'Datasets/Opacidades/TX-RX-ds-20210423-00_ubuntu.csv')
    class_numbers = {
     'NoduloMasa': 1,
     'Consolidacion': 2,
     'PatronIntersticial': 3,
     'Atelectasia': 4,
     'LesionesDeLaPared': 5
     }
    num_epochs = 10
    random_seed = 40
    binary = True
    if binary:
        num_classes = 2
    else:
        num_classes = len(class_numbers.keys())+1 #patologias + background
    pretrained_checkpoint = None #experiment_dir+'/19-03-21/maskRCNN-8.pth'
    pretrained_backbone_path = None #experiment_dir+'/17-04-21/resnetBackbone-8.pth'
    experiment_id = '06-05-21_masksAndBoxs_binary'
    if binary:
        experiment_id+='_binary'
    existing_test_set =  None #'{}/{}'.format(experiment_dir,'30-04-21_masksAndBoxs_binary/testCSV.csv')
    output_dir = '{}/{}/'.format(experiment_dir,experiment_id)

    # --Only accept images with boxes or masks--#
    csv = csv[csv.label_level.isin([
        'box',
        'mask'])].reset_index(drop=True)


    os.makedirs(output_dir,exist_ok=True)
    if 'label_level' not in csv.columns:
        csv['label_level'] = [None] * len(csv)
        for i, row in csv.iterrows():
            assigned = False
            if isinstance(row['mask_path'], str):
                if os.path.exists(row['mask_path']):
                    csv.loc[i, 'label_level'] = 'mask'
                    assigned = True
            else:
                xmin = row['x1']
                xmax = row['x2']
                ymin = row['y1']
                ymax = row['y2']
                if ymax > ymin and xmax > xmin:
                    csv.loc[i, 'label_level'] = 'box'
                    assigned = True
                else:
                    if isinstance(row['class_name'], str):
                        if len(row['class_name']) > 0:
                            csv.loc[i, 'label_level'] = 'imagelabel'
                            assigned = True
            if not assigned:
                csv.loc[i, 'label_level'] = 'nofinding'
        print('finished initialization: ')
        print(csv.label_level.value_counts())
        csv.to_csv(trx_dir + f'Datasets/Opacidades/TX-RX-ds-{experiment_id}.csv', index=False)


    if existing_test_set:
        csv_test = pd.read_csv(existing_test_set)
        test_idx = list(set(csv_test.file_name.values))
        csv_train = csv[~csv.file_name.isin(test_idx)].reset_index(drop=True)
        train_idx = list(set(csv_train.file_name.values))
    else:
        image_ids = list(set(csv.file_name.values))
        class_series = pd.Series([clss.split('-')[0] for clss in csv['class_name'].values])
        csv['stratification'] = csv['image_source'].astype(str)+'_'+csv['label_level'].astype(str)+'_'+class_series
        stratification = [csv[csv.file_name == idx]['stratification'].values[0] for idx in image_ids]

        train_idx, test_idx = train_test_split(image_ids, stratify=stratification,
                                               test_size=0.1,
                                               random_state=random_seed)
        csv_train = csv[csv.file_name.isin(list(train_idx))].reset_index(drop=True)
        csv_test = csv[csv.file_name.isin(list(test_idx))].reset_index(drop=True)
    assert len(set(csv_train.file_name).intersection(csv_test.file_name)) == 0
    print('Len csv:{}, Len csv train: {}, len csv test: {}\nLen train_idx:{} , Len test_idx: {}'.format(len(csv),len(csv_train),len(csv_test),
                                                                                                      len(train_idx),len(test_idx)))
    print('TRAIN SOURCES:')
    print(csv_train.image_source.value_counts(normalize=True))
    print(csv_train.label_level.value_counts(normalize=True))

    print('TEST SOURCES')
    print(csv_test.image_source.value_counts(normalize=True))
    print(csv_test.label_level.value_counts(normalize=True))

    """
    csv_train = csv[:30000].reset_index()
    csv_test = csv[30000:].reset_index() """
    csv_train.to_csv('{}/trainCSV.csv'.format(output_dir),index=False)
    csv_test.to_csv('{}/testCSV.csv'.format(output_dir),index=False)
    dataset = MixedLabelsDataset(csv_train, class_numbers, get_transform(train=False),binary_opacity=binary)
    dataset_test = MixedLabelsDataset(csv_test, class_numbers, get_transform(train=False),binary_opacity=binary)
    print('TRAIN:')
    dataset.quantifyClasses()
    print('\nTEST:')
    dataset_test.quantifyClasses()
    # split the dataset in train and test set
    torch.manual_seed(1)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=collate_fn,
        #sampler=train_sampler
         )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    print('N train: {}. N test: {}'.format(len(data_loader),len(data_loader_test)))
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
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=500
                        )
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        saving_path = '{}/mixedMaskRCNN-{}.pth'.format(output_dir,epoch)
        results_coco_file = '{}/cocoStats-{}.txt'.format(output_dir,epoch)
        evaluate(model, data_loader_test, device=device, model_saving_path=saving_path,results_file=results_coco_file)


if __name__ == '__main__':
    main()
