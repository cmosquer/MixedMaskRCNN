import os, pickle
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import pandas as pd
import torch.utils.data
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from mixed_detection.visualization import draw_annotations, draw_masks
from mixed_detection.utils import get_transform,get_instance_segmentation_model, collate_fn
from mixed_detection.MixedLabelsDataset import MixedLabelsDataset
from mixed_detection.engine import evaluate_coco, evaluate_classification



def label_to_name(label):

    labels = {1:'NoduloMasa',
     2:'Consolidacion',
     3:'PatronIntersticial',
     4:'Atelectasia',
     5:'LesionesDeLaPared'
     }
    return labels[label]

def saveAsFiles(tqdm_loader,model,save_fig_dir, save_figures=True,
                max_detections=None,
                min_score_threshold=None, #if int, the same threshold for all classes. If dict, should contain one element for each class (key: clas_idx, value: class threshold)
                min_box_proportionArea=None, draw='boxes',binary=False,
                save_csv=None #If not None, should be a str with filepath where to save dataframe with targets and predictions
                ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    j = 0
    if save_csv is not None:
        df = pd.DataFrame(columns=['image_name','box_type','label','score','x1','x2','y1','y2','original_file_name','image_source'])

    for image, targets,image_sources,image_paths in tqdm_loader:
        image = list(img.to(device) for img in image)
        j += 1
        if j < 15:
            continue
        image_source = image_sources[0]
        image_path = image_paths[0].replace('/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/',
                                            '')
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        outputs = model(image)
        outputs = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in outputs][0]
        """
        if min_box_proportionArea:
            total_area = image.shape[0]*image.shape[1]  #IMAGE ES LIST!! POR EL BATCH
            minimum_area = total_area*min_box_proportionArea
            outputs['areas'] = [(x2-x1)*(y2-y1) for x1,x2,y1,y2 in outputs['boxes']]
            bigBoxes = np.argwhere(outputs['areas']>minimum_area)
            for k,v in outputs.items():
                outputs[k] = outputs[k][bigBoxes,]"""
        print(outputs)
        if isinstance(min_score_threshold,float):
            high_scores = np.argwhere(outputs['scores']>min_score_threshold)
            for k,v in outputs.items():
                outputs[k] = outputs[k][high_scores,]
            print('after high score\n',outputs)
        if isinstance(min_score_threshold,dict):
            valid_detections_idx = np.array([],dtype=np.int64)
            for clss_idx,th in min_score_threshold.items():
                idxs_clss = np.argwhere(outputs['labels']==clss_idx).flatten()
                print('class id: ',clss_idx, 'idxs clss',idxs_clss)
                if idxs_clss.shape[0]>0:
                    print(idxs_clss)
                    high_scores = np.argwhere(outputs['scores'][idxs_clss]>th).flatten()
                    print('idxs high scores',high_scores)
                    if high_scores.shape[0]>0:
                        print('th',th,'allclassscores',
                              outputs['scores'][idxs_clss],
                              'highscores idx',high_scores)
                        valid_detections_idx = np.concatenate((valid_detections_idx,high_scores)).astype(np.int64)
            #valid_detections_idx = list(dict.fromkeys(valid_detections_idx))
            print('final idxs',valid_detections_idx)
            for k,v in outputs.items():
                outputs[k] = outputs[k][valid_detections_idx,]
            print('after min th\n',outputs)
        if max_detections:
            scores_sort = np.argsort(-outputs['scores'])[:max_detections]
            for k,v in outputs.items():
                outputs[k] = outputs[k][scores_sort,]
            print('after max detections\n',max_detections)

        image = image[0].to(torch.device("cpu")).detach().numpy()[0,:,:]
        targets = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in targets][0]
        if save_figures:
            if len(outputs['labels']) > 0:
                colorimage = np.zeros((image.shape[0],image.shape[1],3),dtype=image.dtype)
                colorimage[:,:,0]=255*image
                colorimage[:,:,1]=255*image
                colorimage[:,:,2]=255*image
                if draw=='boxes':
                    draw_annotations(colorimage, outputs, color=(0, 255, 0),label_to_name=label_to_name,binary=binary)
                if draw=='masks':
                    draw_masks(colorimage, outputs, label_to_name=label_to_name,binary=binary)

                # draw annotations on the image
            if len(targets)>0:
                if len(targets['boxes'])==0:
                    caption = 'Image GT: '+','.join([label_to_name(lbl) for lbl in targets['labels']])
                    cv2.putText(colorimage, caption,
                            (10, 10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), thickness=2)
                else:
                    # draw annotations in red
                    if draw=='boxes':
                        draw_annotations(colorimage, targets, color=(255, 0, 0),label_to_name=label_to_name,binary=binary)
                    #if draw=='masks':
                    #    draw_masks(colorimage, targets, color=(255, 0, 0),label_to_name=label_to_name)

            try:
                saving_path = "{}/{}_{}".format(save_fig_dir, image_source, os.path.basename(image_path.replace('\\','/')))
                cv2.imwrite(saving_path,colorimage)

                print('Saved ',saving_path)

            except:
                print('COULD SAVE ',saving_path)

        if save_csv is not None:
            results_list = []

            for i,label in enumerate(outputs['labels']):
                result = {'image_name': "{}_{}".format(image_source,os.path.basename(image_path)),
                          'box_type':'prediction',
                          'label': label_to_name(label),
                          'score': outputs['scores'][i],
                          'x1': int(outputs['boxes'][i][0]),
                          'y1': int(outputs['boxes'][i][1]),
                          'x2': int(outputs['boxes'][i][2]),
                          'y2': int(outputs['boxes'][i][3]),
                          'original_file_name': image_path,
                          'image_source':image_source
                          }
                results_list.append(result)

            for i, label in enumerate(targets['labels']):
                result = {'image_name': "{}_{}".format(image_source,os.path.basename(image_path)),
                          'box_type': 'ground-truth',
                          'label': label_to_name(label),
                          'x1': int(targets['boxes'][i][0]),
                          'y1': int(targets['boxes'][i][1]),
                          'x2': int(targets['boxes'][i][2]),
                          'y2': int(targets['boxes'][i][3]),
                          'original_file_name': image_path,
                          'image_source': image_source
                          }
                results_list.append(result)
            if len(results_list)==0:
                results_list = [{'image_name': "{}_{}".format(image_source,os.path.basename(image_path)),
                          'box_type': 'true-negative',
                          'original_file_name': image_path,
                          'image_source': image_source
                          }]
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
    save_figures = True
    view_in_window = False
    calculate_coco = False
    loop = False
    save_csv = False
    calculate_classification=False

    chosen_experiment = '2021-05-24_masks_boxes_binary'
    chosen_epoch = 1

    baseDir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/'
    output_dir = baseDir +'TRx-v2/Experiments/'
    save_fig_dir = f'{output_dir}/{chosen_experiment}/detections_test_epoch-{chosen_epoch}/'
    output_csv_path = f'{output_dir}/{chosen_experiment}/test_output-epoch{chosen_epoch}.csv'
    results_coco_file = f'{output_dir}/{chosen_experiment}/stats-test-epoch_{chosen_epoch}.txt'
    trainedModelPath = "{}/{}/mixedMaskRCNN-{}.pth".format(output_dir, chosen_experiment, chosen_epoch)
    classification_data = f'{output_dir}/{chosen_experiment}/classification_data-{chosen_epoch}'
    binary_opacity=True

    os.makedirs(save_fig_dir,exist_ok=True)


    csv_test = pd.read_csv(
        baseDir + 'TRx-v2/Experiments/test_groundtruth_validados.csv')

    image_ids_test = set(csv_test.file_name)
    print('Images in test:{}. Instances total: {}'.format(len(image_ids_test),len(csv_test)))

    #csv_test = csv[:30].reset_index()
    #csv_test = pd.read_csv(f'{output_dir}/{chosen_experiment}/testCSV.csv').reset_index(drop=True)

    print('{} images to evaluate'.format(len(csv_test)))

    class_numbers = {'NoduloMasa': 1,
     'Consolidacion': 2,
     'PatronIntersticial': 3,
     'Atelectasia': 4,
     'LesionesDeLaPared': 5
     }

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(1)
    if binary_opacity:
        num_classes = 2
    else:
        num_classes = len(class_numbers.keys())+1
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load(trainedModelPath))
    #model = torch.load(trainedModelPath)
    model.to(device)

    """#Change model parameters?
    model.box_score_thresh = 0.05
    model.box_nms_thresh = 0.3
    model.box_detections_per_img = 100
    model.rpn_nms_thresh = 0.5"""

    model.eval()
    # define data loader
    print('Model loaded')

    if calculate_coco:
        dataset_test = MixedLabelsDataset(csv_test, class_numbers, get_transform(train=False),
                                          binary_opacity=binary_opacity,
                                          return_image_source=False)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)
        print('DATASET FOR COCO:')
        dataset_test.quantifyClasses()
        evaluate_coco(model, data_loader_test, device=device,use_cpu=True,
                 results_file=results_coco_file)

    if os.path.exists(classification_data):
        with open(classification_data,'rb') as f:
            classification_data_dict = pickle.load(f)
        test_clf = classification_data_dict['clf']
    else:
        print('no existe archivo ',classification_data)
        test_clf=None
    if calculate_classification:
        evaluate_classification(model, data_loader_test, device=device,log_wandb=False,
                                results_file=results_coco_file,test_clf=test_clf)
    #Redefinir solo las que quiero guardar la imagen
    try:
        csv_test_files = csv_test[csv_test.image_source.isin(['hiba','jsrt','mimic_relabeled'])].reset_index(drop=True)
        print('Using only top datasets. Total of {} images'.format(len(set(csv_test_files.file_name.values))))
    except ValueError as e:
        print(e)
        print('Not reseting index')
        csv_test_files = csv_test[csv_test.image_source == 'hiba']

    dataset_test_files = MixedLabelsDataset(csv_test_files, class_numbers,
                                            get_transform(train=False), binary_opacity=binary_opacity,
                                            return_image_source=True)
    data_loader_test_files = torch.utils.data.DataLoader(
        dataset_test_files, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    tqdm_loader_files = tqdm(data_loader_test_files)
    print('DATASET FOR FIGURES:')
    print(dataset_test_files.quantifyClasses())
    """
    min_score_thresholds = {1: 0.25, #'NoduloMasa', #NUEVO 0.35
       2: 0.25 , #'Consolidacion',
       3: 0.25, #'PatronIntersticial',
       4: 0.25, #'Atelectasia',
       5: 0.25 #'LesionesDeLaPared'
       }"""
    min_score_thresholds = 0.25
    min_box_proportionArea = 1/50 #Minima area de un box valido como proporcion del area total ej: al menos un cincuentavo del area total

    if save_figures or save_csv:
        while saveAsFiles(tqdm_loader_files, model, save_fig_dir=save_fig_dir,binary=binary_opacity,
                          max_detections=8, min_score_threshold=min_score_thresholds,
                          min_box_proportionArea=min_box_proportionArea,
                          save_csv=output_csv_path,save_figures=save_figures):
            pass
    if view_in_window:
        if loop:
            while visualize(tqdm_loader_files,model,max_detections=4):
                pass
        else:
            visualize(tqdm_loader_files,model)


if __name__ == '__main__':
    main()
