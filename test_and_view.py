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
from mixed_detection.utils import get_transform,get_instance_segmentation_model, collate_fn, process_output, get_object_detection_model, update_regression_features
from mixed_detection.MixedLabelsDataset import MixedLabelsDataset
from mixed_detection.engine import evaluate_coco, evaluate_classification
from datetime import datetime

import wandb


def label_to_name(label):

    labels = {1:'NoduloMasa',
     2:'Consolidacion',
     3:'PatronIntersticial',
     4:'Atelectasia',
     5:'LesionesDeLaPared'
     }
    return labels[label]

def saveAsFiles(tqdm_loader,model,device,
                save_fig_dir,
                save_figures=True,
                max_detections=None,binary_classifier=None,posterior_th=None,
                min_score_threshold=None, #if int, the same threshold for all classes. If dict, should contain one element for each class (key: clas_idx, value: class threshold)
                min_box_proportionArea=None, draw='boxes',binary=False,results_file='',
                save_csv=None, #If not None, should be a str with filepath where to save dataframe with targets and predictions

                ):
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

        outputs = model(image)

        height = image[0].shape[1]
        width = image[0].shape[2]
        total_area = height*width
        outputs = [{k: v.to(torch.device("cpu")).detach() for k, v in t.items()} for t in outputs][0]

        outputs = process_output(outputs,total_area)
        x_reg = update_regression_features(outputs['scores'], outputs['areas'])
        outputs = process_output(outputs,total_area,
                                 max_detections=max_detections,
                                 min_box_proportionArea=min_box_proportionArea,
                                 min_score_threshold=min_score_threshold)

        print(type(outputs))
        outputs = {k: v.numpy() for k, v in outputs.items()}
        image = image[0].to(torch.device("cpu")).detach().numpy()[0,:,:]
        targets = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in targets][0]
        if save_figures:
            colorimage = np.zeros((image.shape[0], image.shape[1], 3), dtype=image.dtype)
            colorimage[:, :, 0] = 255 * image
            colorimage[:, :, 1] = 255 * image
            colorimage[:, :, 2] = 255 * image
            if len(outputs['labels']) > 0:

                if draw=='boxes':
                    draw_annotations(colorimage, outputs, color=(0, 255, 0),label_to_name=label_to_name,binary=binary)
                if draw=='masks':
                    draw_masks(colorimage, outputs, label_to_name=label_to_name,binary=binary)

                # draw annotations on the image
            if len(targets)>0:
                if len(targets['boxes'])==0:
                    caption = 'Image GT: opacidad'#+','.join([label_to_name(lbl) for lbl in targets['labels']])
                    cv2.putText(colorimage, caption,
                            (10, 10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), thickness=2)
                else:
                    # draw annotations in red
                    if draw=='boxes':
                        draw_annotations(colorimage, targets, color=(255, 0, 0),label_to_name=label_to_name,binary=binary)
                    #if draw=='masks':
                    #    draw_masks(colorimage, targets, color=(255, 0, 0),label_to_name=label_to_name)

            print('TARGET for {} is {}'.format(image_path,targets['labels']))
            os.makedirs(save_fig_dir+'TruePositive',exist_ok=True)
            os.makedirs(save_fig_dir+'FalsePositive',exist_ok=True)
            os.makedirs(save_fig_dir+'FalseNegative',exist_ok=True)
            os.makedirs(save_fig_dir+'TrueNegative',exist_ok=True)
            folder = ''
            #try:
            #predspath = results_file.replace('cocoStats', 'test_classification_data').replace('.txt', '')
            if binary_classifier is not None:
                pred = None
                cont_pred = None
                with open(binary_classifier, 'rb') as f:
                    test_clf = pickle.load(f)
                if posterior_th is not None:
                    cont_pred = test_clf.predict_proba(x_reg.reshape(1, -1))[0,1]
                    pred = 1 if cont_pred>posterior_th else 0
                else:
                    pred = test_clf.predict(x_reg.reshape(1, -1))
                    cont_pred = pred.copy()
                gt = 1 if len(targets['labels']) > 0 else 0
                print('CONT PRED: {}, BINARY PRED: {} , GT: {}'.format(cont_pred,pred,gt))
                if np.all(pred == gt):
                    if np.all(gt == 0):
                        folder = 'TrueNegative'
                    else:
                        folder = 'TruePositive'
                else:
                    if np.all(gt == 0):
                        folder = 'FalsePositive'
                    else:
                        folder = 'FalseNegative'

            saving_path = "{}/{}/{}_{}".format(save_fig_dir, folder,
                                               image_source, os.path.basename(image_path.replace('\\', '/')))
            cv2.imwrite(saving_path, colorimage)

            print('Saved ', saving_path)

            #except:
            #    print('COULDNT SAVE ',image_path)

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
                          'overall_pred': pred,
                          'overall_prob': cont_pred,
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




def visualize(tqdm_loader,model,device,save_fig_dir=None,max_detections=None):

    for image, targets,image_sources,image_paths in tqdm_loader:
        image = list(img.to(device) for img in image)
        image_source = image_sources[0]
        image_path = image_paths[0].replace('/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/',
                                            '')

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
    project = "mixed_mask_rcnn"

    print('starting training script')
    trx_dir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/TRx-v2/'
    output_dir = trx_dir+'Experiments/'

    config = {
        'test_set' : '{}/{}'.format(output_dir,'test_groundtruth_validados.csv'), #output_dir+'2021-06-25_boxes_binary/testCSV.csv',#
        'experiment': '2021-06-20_boxes_binary',
        'experiment_type': 'boxes',
        'tested_epoch': 4,

        'opacityies_as_binary':True,
        'masks_as_boxes': True,

        'posterior_th': 0.08,
        'calculate_coco': False,
        'calculate_classification': False,
        'binary_classifier': output_dir+'2021-06-20_boxes_binary/test-2021-06-28_/classification_data-test-epoch_4RF',
        'adjust_new_LR': False,
        'save_figures': True,
        'only_best_datasets': False,
        'save_csv': False,
        'view_in_window': False,
        'loop': False,
        'force_cpu': False,
    }
    print('starting test script')

    force_cpu = config['force_cpu'] #Lo que observe: al setearlo en true igual algo ahce con la gpu por ocupa ~1500MB,
    # pero si lo dejas en false ocupa como 4000MB. En cuanto a velocidad, el de gpu es mas rapido sin dudas, pero el cpu super tolerable (5segs por imagen aprox)

    chosen_experiment = config['experiment']
    chosen_epoch = config['tested_epoch']
    trainedModelPath = "{}/{}/mixedMaskRCNN-{}.pth".format(output_dir, chosen_experiment, chosen_epoch)

    date = datetime.today().strftime('%Y-%m-%d')
    save_fig_dir = f'{output_dir}/{chosen_experiment}/test-{date}/detections_test_epoch-{chosen_epoch}/'
    output_csv_path = f'{output_dir}/{chosen_experiment}/test-{date}/test_output-epoch{chosen_epoch}.csv'
    results_coco_file = f'{output_dir}/{chosen_experiment}/test-{date}/cocoStats-test-epoch_{chosen_epoch}.txt'
    classification_data = f'{output_dir}/{chosen_experiment}/test-{date}/classification_data-{chosen_epoch}'
    binary_opacity=config['opacityies_as_binary']

    os.makedirs(save_fig_dir,exist_ok=True)


    csv_test = pd.read_csv(config['test_set'])

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

    #

    if force_cpu:
        device = torch.device('cpu')
    else:
        torch.manual_seed(1)
        torch.cuda.synchronize()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if binary_opacity:
        num_classes = 2
    else:
        num_classes = len(class_numbers.keys())+1

    if config['experiment_type']=='boxes':
        # get the model using our helper function
        model = get_object_detection_model(num_classes)
    if config['experiment_type']=='masks':
        # get the model using our helper function
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
    experiment_id = f"test_{date}_{chosen_experiment}"
    with wandb.init(config=config, project=project, name=experiment_id):
        config = wandb.config
        wandb_valid = {}
        if config['calculate_coco']:
            dataset_test = MixedLabelsDataset(csv_test, class_numbers, get_transform(train=False),
                                              binary_opacity=binary_opacity,
                                              return_image_source=False)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1, shuffle=False, num_workers=0,
                collate_fn=collate_fn)
            print('DATASET FOR COCO:')
            dataset_test.quantifyClasses()
            results_coco = evaluate_coco(model, data_loader_test, device=device,use_cpu=True,
                     results_file=results_coco_file)
            wandb_valid.update(results_coco)
        test_clf = None
        if not config['adjust_new_LR']:
            if config['binary_classifier'] is not None:
                if os.path.exists(config['binary_classifier']):
                    with open(config['binary_classifier'], 'rb') as f:
                        test_clf = pickle.load(f)
            else:
                if os.path.exists(classification_data):
                    with open(classification_data,'rb') as f:
                        classification_data_dict = pickle.load(f)
                    print('Loaded logistic regressor for classification')
                    test_clf = classification_data_dict['clf']
                else:
                    print('no existe archivo ',classification_data)



        if config['calculate_classification']:
            dataset_test = MixedLabelsDataset(csv_test, class_numbers, get_transform(train=False),
                                              binary_opacity=binary_opacity,
                                              return_image_source=True)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1, shuffle=False, num_workers=0,
                collate_fn=collate_fn)
            results_classif = evaluate_classification(model, data_loader_test, device=device,log_wandb=False,
                                    results_file=results_coco_file,test_clf=test_clf,cost_ratios=config['cost_ratios'])
            wandb_valid.update(results_classif)
        #Redefinir solo las que quiero guardar la imagen
        if config['only_best_datasets']:
            try:
                csv_test_files = csv_test[csv_test.image_source.isin(['hiba','jsrt','mimic_relabeled'])].reset_index(drop=True)
                print('Using only top datasets. Total of {} images'.format(len(set(csv_test_files.file_name.values))))
            except ValueError as e:
                print(e)
                print('Not reseting index')
                csv_test_files = csv_test[csv_test.image_source == 'hiba']
        else:
            csv_test_files = csv_test.copy()
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
        min_score_thresholds = 0.2
        min_box_proportionArea = float(1/25) #Minima area de un box valido como proporcion del area total ej: al menos un cincuentavo del area total

        if config['save_figures'] or config['save_csv']:

            while saveAsFiles(tqdm_loader_files, model, device=device,save_fig_dir=save_fig_dir,binary=binary_opacity,
                              max_detections=8, min_score_threshold=min_score_thresholds,
                              binary_classifier=config['binary_classifier'],
                              posterior_th=config['posterior_th'],
                              min_box_proportionArea=min_box_proportionArea,results_file=results_coco_file,
                              save_csv=output_csv_path,save_figures=config['save_figures']):
                pass
        if config['view_in_window']:
            if config['loop']:
                while visualize(tqdm_loader_files,model,device=device,max_detections=4):
                    pass
            else:
                visualize(tqdm_loader_files,model)

        wandb.log(wandb_valid)
if __name__ == '__main__':
    main()
