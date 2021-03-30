import os
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import pandas as pd
import torch.utils.data
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from mixed_detection.visualization import draw_annotations
from mixed_detection.utils import get_transform,get_instance_segmentation_model, collate_fn
from mixed_detection.MixedLabelsDataset import MixedLabelsDataset

def label_to_name(label):
    labels = {1:'NoduloMasa',
     2:'Consolidacion',
     3:'PatronIntersticial',
     4:'Atelectasia',
     5:'LesionesDeLaPared'
     }
    return labels[label]

def saveAsFiles(tqdm_loader,model,save_fig_dir,max_detections=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    j = 0
    for image, targets,image_sources,image_paths in tqdm_loader:
        image = list(img.to(device) for img in image)
        j+=1
        if j<15:
            continue
        image_source = image_sources[0]
        image_path = image_paths[0].replace('/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/',
                                            '')
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        outputs = model(image)
        print('finished outputs')
        outputs = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in outputs][0]
        if max_detections:
            scores_sort = np.argsort(-outputs['scores'])[:max_detections]
            for k,v in outputs.items():
                outputs[k] = outputs[k][scores_sort,]


        image = image[0].to(torch.device("cpu")).detach().numpy()[0,:,:]
        targets = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in targets][0]

        cmap = 'gray'
        #plt.imshow(image, cmap=cmap)
        #if plt.waitforbuttonpress():
        #    return False
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["bottom"].set_visible(False)
        ax[0].spines["left"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["bottom"].set_visible(False)
        ax[1].spines["left"].set_visible(False)
        if len(outputs['labels']) > 0:

            colorimage = np.zeros((image.shape[0],image.shape[1],3),dtype=image.dtype)
            colorimage[:,:,0]=image
            colorimage[:,:,1]=image
            colorimage[:,:,2]=image
            draw_annotations(colorimage, outputs, color=(0, 255, 0),label_to_name=label_to_name)

            # draw annotations on the image
        if len(targets)>0:
            print('boxes: ',len(targets['boxes']))
            if len(targets['boxes'])==0:
                caption = 'Image GT: '+','.join([label_to_name(lbl) for lbl in targets['labels']])
                print('caption',caption)
                cv2.putText(colorimage, caption,
                        (10, 10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), thickness=2)
            else:
                # draw annotations in red
                draw_annotations(colorimage, targets, color=(255, 0, 0),label_to_name=label_to_name)
            ax[0].imshow(colorimage)

        print('finished drawing')


        if len(targets['masks'])+len(outputs['masks']) > 0:
            ax[1].imshow(colorimage)
            for mask in targets['masks']:
                ax[1].imshow(np.squeeze(mask), alpha=0.2, cmap='Reds')

            for mask in outputs['masks']:
                ax[1].imshow(np.squeeze(mask), alpha=0.2, cmap='Greens')
        else:
            print('no boxes')
            ax[1].imshow(image,cmap='gray')
        plt.tight_layout()
        saving_path = "{}/{}_{}".format(save_fig_dir, image_source, os.path.basename(image_path))
        print(saving_path)
        plt.savefig(saving_path)
        print('finished')




def visualize(tqdm_loader,model,save_fig_dir=None,max_detections=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for image, targets,image_sources,image_paths in tqdm_loader:
        image = list(img.to(device) for img in image)
        image_source = image_sources[0]
        image_path = image_paths[0].replace('/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/',
                                            '')
        torch.cuda.synchronize()
        outputs = model(image)

        outputs = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in outputs][0]
        if max_detections:
            scores_sort = np.argsort(-outputs['scores'])[:max_detections]
            for k,v in outputs.items():
                outputs[k] = outputs[k][scores_sort,]


        image = image[0].to(torch.device("cpu")).detach().numpy()[0,:,:]
        targets = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in targets][0]

        cmap = 'gray'
        plt.imshow(image, cmap=cmap)
        #if plt.waitforbuttonpress():
        #    return False

        if len(outputs['labels']) > 0:

            colorimage = np.zeros((image.shape[0],image.shape[1],3))
            colorimage[:,:,0]=image
            colorimage[:,:,1]=image
            colorimage[:,:,2]=image
            draw_annotations(colorimage, outputs, color=(0, 255, 0),label_to_name=label_to_name)

            # draw annotations on the image
        if len(targets)>0:
            # draw annotations in red
            draw_annotations(colorimage, targets, color=(255, 0, 0),label_to_name=label_to_name)
            plt.imshow(colorimage)


        plt.imshow(colorimage)
        plt.title("{} - {}\n{}".format(image_source,image_path,[label_to_name(lbl) for lbl in targets['labels']]))
        plt.xlabel('Green: predictions. Red: ground-truth')

        if plt.waitforbuttonpress():
            return False
        for mask in targets['masks']:
            plt.imshow(np.squeeze(mask), alpha=0.2, cmap='Reds')
        if plt.waitforbuttonpress():
            return False
        plt.imshow(colorimage)
        for mask in outputs['masks']:

            plt.imshow(np.squeeze(mask), alpha=0.2, cmap='Greens')
        if plt.waitforbuttonpress():
            return False




def main(args=None):
    print('starting test script')
    save_as_files = True
    view_in_window = False
    loop = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    baseDir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/'
    # use our dataset and defined transformations
    csv = pd.read_csv(
        baseDir + 'TRx-v2/Datasets/Opacidades/TX-RX-ds-20210308-03_ubuntu.csv')
    output_dir = baseDir +'TRx-v2/Experiments'
    chosen_experiment = '19-03-21'
    save_fig_dir = f'{output_dir}/{chosen_experiment}/detections_test/'
    os.makedirs(save_fig_dir,exist_ok=True)
    print('Created dir')
    chosen_epoch = 8

    trainedModelPath = "{}/{}/maskRCNN-{}.pth".format(output_dir, chosen_experiment, chosen_epoch)
    #csv_test = csv[csv.image_source == 'hiba'].reset_index()
    image_ids = list(set(csv.file_name.values))
    image_sources = [csv[csv.file_name==idx]['image_source'].values[0] for idx in image_ids]
    train_idx, test_idx = train_test_split(image_ids,stratify=image_sources,
                                           test_size=0.1,
                                           random_state=42)
    csv_test = csv[csv.file_name.isin(list(test_idx))]
    csv_test = csv_test[csv_test.image_source=='hiba'].reset_index()


    print('{} images to evaluate'.format(len(csv_test)))

    class_numbers = {'NoduloMasa': 1,
     'Consolidacion': 2,
     'PatronIntersticial': 3,
     'Atelectasia': 4,
     'LesionesDeLaPared': 5
     }
    dataset_test = MixedLabelsDataset(csv_test, class_numbers, get_transform(train=False), return_image_source=True)
    torch.manual_seed(1)
    num_classes=6
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load(trainedModelPath))
    #model = torch.load(trainedModelPath)
    model.to(device)
    model.eval()
    # define data loader
    print('Model loaded')
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    tqdm_loader = tqdm(data_loader_test)
    if save_as_files:
        while saveAsFiles(tqdm_loader, model, save_fig_dir=save_fig_dir, max_detections=4):
            pass
    if view_in_window:
        if loop:
            while visualize(tqdm_loader,model,max_detections=4):
                pass
        else:
            visualize(tqdm_loader,model)


if __name__ == '__main__':
    main()
