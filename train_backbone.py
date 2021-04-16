import torch
import pandas as pd
import torch.utils.data
from mixed_detection.utils import get_transform,collate_fn, create_resnet_head, seed_all
from mixed_detection.engine import train_one_epoch_resnet, evaluate_resnet
from sklearn.model_selection import train_test_split
from mixed_detection.MixedLabelsDataset import ImageLabelsDataset
import os
from torchvision.ops import misc as misc_nn_ops
from torch import nn
from torchvision.models import resnet
from torchvision import transforms




def main(args=None):
    print('starting training script')
    trx_dir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/TRx-v2/'
    experiment_dir = trx_dir+'Experiments/'
    csv = pd.read_csv(trx_dir+'Datasets/Opacidades/TX-RX-ds-20210415-00_ubuntu.csv')
    class_numbers = {
     'NoduloMasa': 0,
     'Consolidacion': 1,
     'PatronIntersticial': 2,
     'Atelectasia': 3,
     'LesionesDeLaPared': 4
     }
    num_epochs = 1
    seed_all(27)
    pretrained_checkpoint = None #experiment_dir+'/19-03-21/maskRCNN-8.pth'
    experiment_id = '16-04-21'
    output_dir = '{}/{}/'.format(experiment_dir,experiment_id)
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

    # --Only accept images with boxes or masks--#
    csv = csv[csv.label_level.isin(['imagelabel', 'nofinding'])].reset_index()

    image_ids = list(set(csv.file_name.values))
    image_sources = [csv[csv.file_name == idx]['image_source'].values[0] for idx in image_ids]
    train_idx, test_idx = train_test_split(image_ids, stratify=image_sources,  # --->sources o label level?
                                           test_size=0.1, random_state=42)

    csv_train = csv[csv.file_name.isin(list(train_idx))].reset_index()
    csv_test = csv[csv.file_name.isin(list(test_idx))].reset_index()
    assert len(set(csv_train.file_name).intersection(csv_test.file_name)) == 0
    print('Len csv:{}, Len csv train: {}, len csv test: {}\nLen train_idx:{} , Len test_idx: {}'.format(len(csv),
                                                                                                        len(csv_train),
                                                                                                        len(csv_test),
                                                                                                        len(train_idx),
                                                                                                        len(test_idx)))
    print('TRAIN SOURCES:')
    print(csv_train.image_source.value_counts(normalize=True))
    print('TEST SOURCES')
    print(csv_test.image_source.value_counts(normalize=True))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageLabelsDataset(csv_train, class_numbers,data_transforms)#get_transform(train=False))#
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=0,
        #collate_fn=collate_fn,
        #sampler=train_sampler
         )
    dataset_test = ImageLabelsDataset(csv_test, class_numbers,data_transforms)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=16, shuffle=False, num_workers=0)
        #collate_fn=collate_fn)

    print('N train: {}. N test: {}'.format(len(data_loader),len(data_loader_test)))

    # our dataset has two classes only - background and person
    num_classes = len(class_numbers.keys())

    backbone = resnet.resnet50(
        pretrained=True,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d
        )

    top_head = create_resnet_head(backbone.fc.in_features, num_classes)  # because ten classes
    backbone.fc = top_head  # replace the fully connected layer

    backbone.to(device)

    # construct an optimizer
    params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch_resnet(backbone, criterion, optimizer, data_loader, device, epoch, print_freq=200
                        )
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        model_saving_path = '{}/resnetBackbone-{}.pth'.format(output_dir,epoch)
        val_loss, val_loss_classes = evaluate_resnet(backbone, data_loader_test,
                                                     device, criterion,
                                                     model_saving_path=model_saving_path)
        print(f'Total validation loss {val_loss}')
        print(val_loss_classes)
        for c,name in class_numbers.items():
            loss = val_loss_classes[c].item()
            print(f'{name}: {loss}')



if __name__ == '__main__':
    main()
