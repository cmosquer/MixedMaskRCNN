import torch
import pandas as pd
import torch.utils.data
from mixed_detection.utils import get_instance_segmentation_model,get_transform,collate_fn
from mixed_detection.engine import train_one_epoch, evaluate
from sklearn.model_selection import train_test_split
from mixed_detection.MixedLabelsDataset import MixedLabelsDataset
import os

def main(args=None):
    print('starting training script')
    trx_dir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/TRx-v2/'
    experiment_dir = trx_dir+'Experiments/'
    csv = pd.read_csv(trx_dir+'Datasets/Opacidades/TX-RX-ds-20210330-00_ubuntu.csv')
    class_numbers = {
     'NoduloMasa': 1,
     'Consolidacion': 2,
     'PatronIntersticial': 3,
     'Atelectasia': 4,
     'LesionesDeLaPared': 5
     }
    num_epochs = 10
    pretrained_checkpoint = None #experiment_dir+'/19-03-21/maskRCNN-8.pth'
    output_dir = experiment_dir+'/09-04-21/'
    os.makedirs(output_dir,exist_ok=True)
    loss_type_weights = {
                "loss_classifier": 4,
                "loss_box_reg": 3,
                "loss_image_level": 1,
                "loss_mask": 5,
                "loss_objectness": 3,
                "loss_rpn_box_reg": 3,
            }
    image_ids = list(set(csv.file_name.values))
    image_sources = [csv[csv.file_name == idx]['image_source'].values[0] for idx in image_ids]
    train_idx, test_idx = train_test_split(image_ids,stratify=image_sources,test_size=0.1,random_state=42)
    csv_train = csv[csv.file_name.isin(list(train_idx))].reset_index()
    csv_test = csv[csv.file_name.isin(list(test_idx))].reset_index()
    assert len(set(csv_train.file_name).intersection(csv_test.file_name)) == 0
    print('Len csv:{}, Len csv train: {}, len csv test: {}\nLen train_idx:{} , Len test_idx: {}'.format(len(csv),len(csv_train),len(csv_test),
                                                                                                      len(train_idx),len(test_idx)))
    #csv_train = csv[:300]
    #csv_test = csv[:-300]
    dataset = MixedLabelsDataset(csv_train, class_numbers, get_transform(train=False))
    dataset_test = MixedLabelsDataset(csv_test, class_numbers, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    print('N train: {}. N test: {}'.format(len(data_loader),len(data_loader_test)))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 6  #5 patologias + background

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
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
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10,loss_type_weights=loss_type_weights)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        saving_path = '{}/mixedMaskRCNN-{}.pth'.format(output_dir,epoch)
        evaluate(model, data_loader_test, device=device, saving_path=saving_path)


if __name__ == '__main__':
    main()
