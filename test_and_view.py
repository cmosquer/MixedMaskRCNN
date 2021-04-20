import os, random
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
from mixed_detection.engine import evaluate
def label_to_name(label):
    labels = {1:'NoduloMasa',
     2:'Consolidacion',
     3:'PatronIntersticial',
     4:'Atelectasia',
     5:'LesionesDeLaPared'
     }
    return labels[label]

def saveAsFiles(tqdm_loader,model,save_fig_dir,max_detections=None,
                min_score_threshold=None, #if int, the same threshold for all classes. If dict, should contain one element for each class (key: clas_idx, value: class threshold)
                save_csv=None #If not None, should be a str with filepath where to save dataframe with targets and predictions
                ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    j = 0
    if save_csv is not None:
        df = pd.DataFrame(columns=['image_name','box_type','label','score','area'])

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
        outputs = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in outputs][0]
        if isinstance(min_score_threshold,float):
            high_scores = np.argwhere(outputs['scores']>min_score_threshold)
            for k,v in outputs.items():
                outputs[k] = outputs[k][high_scores,]
        if isinstance(min_score_threshold,dict):
            valid_detections_idx = []
            for clss_idx,th in min_score_threshold.items():
                idxs_clss = np.argwhere(outputs['labels']==clss_idx)
                print('idxs clss',idxs_clss)
                high_scores = np.argwhere(outputs['scores'][idxs_clss]>th)
                print('idxs high scores',high_scores)
                valid_detections_idx.append(list(high_scores))
            valid_detections_idx = list(dict.fromkeys(valid_detections_idx))
            print('final idxs',valid_detections_idx)

            valid_detections_idx = np.array(valid_detections_idx)
            for k,v in outputs.items():
                outputs[k] = outputs[k][valid_detections_idx,]

        if max_detections:
            scores_sort = np.argsort(-outputs['scores'])[:max_detections]
            for k,v in outputs.items():
                outputs[k] = outputs[k][scores_sort,]


        image = image[0].to(torch.device("cpu")).detach().numpy()[0,:,:]
        targets = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in targets][0]

        if len(outputs['labels']) > 0:
            colorimage = np.zeros((image.shape[0],image.shape[1],3),dtype=image.dtype)
            colorimage[:,:,0]=255*image
            colorimage[:,:,1]=255*image
            colorimage[:,:,2]=255*image
            draw_annotations(colorimage, outputs, color=(0, 255, 0),label_to_name=label_to_name)

            # draw annotations on the image
        if len(targets)>0:
            if len(targets['boxes'])==0:
                caption = 'Image GT: '+','.join([label_to_name(lbl) for lbl in targets['labels']])
                cv2.putText(colorimage, caption,
                        (10, 10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), thickness=2)
            else:
                # draw annotations in red
                draw_annotations(colorimage, targets, color=(255, 0, 0),label_to_name=label_to_name)

        try:
            saving_path = "{}/{}_{}".format(save_fig_dir, image_source, os.path.basename(image_path.replace('\\','/')))
            cv2.imwrite(saving_path,colorimage)

            print('Saved ',saving_path)

        except:
            print('COULD SAVE ',saving_path)

        if save_csv is not None:
            results_list = []

            for i,label in enumerate(outputs['labels']):
                result = {'image_name':os.path.basename(image_path),
                          'box_type':'prediction',
                          'label':label_to_name(label),
                          'area':outputs['areas'][i],
                          'score':outputs['scores'][i]}
                results_list.append(result)

            for i, label in enumerate(targets['labels']):
                result = {'image_name': os.path.basename(image_path),
                          'box_type': 'ground-truth',
                          'label': label_to_name(label),
                          'area': targets['areas'][i],
                          }
                results_list.append(result)

            df = df.append(results_list,ignore_index=True)

        del outputs,targets, image
    if save_csv is not None:
        df.to_csv(save_csv,index=False)




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
    evaluate_coco = False
    loop = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    baseDir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/'
    # use our dataset and defined transformations

    output_dir = baseDir +'TRx-v2/Experiments'
    chosen_experiment = '18-04-21'
    chosen_epoch = 9
    save_fig_dir = f'{output_dir}/{chosen_experiment}/detections_test_epoch-{chosen_epoch}/'
    os.makedirs(save_fig_dir,exist_ok=True)
    output_csv_path = f'{output_dir}/{chosen_experiment}/targets_and_predictions_table-epoch{chosen_epoch}.csv'
    results_coco_file = f'{output_dir}/{chosen_experiment}/cocoStats-test-epoch_{chosen_epoch}.txt'

    trainedModelPath = "{}/{}/mixedMaskRCNN-{}.pth".format(output_dir, chosen_experiment, chosen_epoch)

    csv = pd.read_csv(
        baseDir + 'TRx-v2/Datasets/Opacidades/TX-RX-ds-20210415-00_ubuntu.csv')

    image_ids = set(csv.file_name.values)
    csv_train = pd.read_csv(f'{output_dir}/{chosen_experiment}/trainCSV.csv').reset_index(drop=True)
    image_ids_train = set(csv_train.file_name.values)
    image_ids_test = image_ids.difference(image_ids_train)
    print('Len total: {}, len train: {}, len test:{}'.format(len(image_ids),len(image_ids_train),len(image_ids_test)))

    csv_test = csv[csv.file_name.isin(list(image_ids_test))].reset_index(drop=True)
    #csv_test = csv[:30].reset_index()

    print('{} images to evaluate'.format(len(csv_test)))

    class_numbers = {'NoduloMasa': 1,
     'Consolidacion': 2,
     'PatronIntersticial': 3,
     'Atelectasia': 4,
     'LesionesDeLaPared': 5
     }


    torch.manual_seed(1)
    num_classes = len(class_numbers.keys())+1
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load(trainedModelPath))
    #model = torch.load(trainedModelPath)
    model.to(device)
    model.eval()
    # define data loader
    print('Model loaded')

    if evaluate_coco:
        dataset_test = MixedLabelsDataset(csv_test, class_numbers, get_transform(train=False),
                                          return_image_source=False)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)
        print('DATASET FOR COCO:')
        dataset_test.quantifyClasses()
        evaluate(model, data_loader_test, device=device,
                 results_file=results_coco_file)


    #Redefinir solo las que quiero guardar la imagen
    try:
        csv_test_files = csv_test[csv_test.image_source.isin(['hiba','jsrt','mimic_relabeled'])].reset_index(drop=True)
    except ValueError as e:
        print(e)
        print('Not reseting index')
        csv_test_files = csv_test[csv_test.image_source == 'hiba']

    dataset_test_files = MixedLabelsDataset(csv_test_files, class_numbers, get_transform(train=False), return_image_source=True)
    data_loader_test_files = torch.utils.data.DataLoader(
        dataset_test_files, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    tqdm_loader_files = tqdm(data_loader_test_files)
    print('DATASET FOR FIGURES:')
    print(dataset_test_files.quantifyClasses())

    min_score_thresholds = {1: 0.25, #'NoduloMasa',
       2: 0.6 , #'Consolidacion',
       3: 0.6, #'PatronIntersticial',
       4: 0.5, #'Atelectasia',
       5: 0.5 #'LesionesDeLaPared'
       }

    if save_as_files:
        while saveAsFiles(tqdm_loader_files, model, save_fig_dir=save_fig_dir,
                          max_detections=8, min_score_threshold=min_score_thresholds,
                          save_csv=output_csv_path):
            pass
    if view_in_window:
        if loop:
            while visualize(tqdm_loader_files,model,max_detections=4):
                pass
        else:
            visualize(tqdm_loader_files,model)


if __name__ == '__main__':
    main()
