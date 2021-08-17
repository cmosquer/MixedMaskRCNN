import os, pickle
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import pandas as pd
import torch.utils.data
from tqdm import tqdm
import cv2
import psutil
from matplotlib import pyplot as plt
from mixed_detection.visualization import draw_annotations, draw_masks
import mixed_detection.utils as ut
from mixed_detection.MixedLabelsDataset import MixedLabelsDataset
from mixed_detection.engine import evaluate_coco, evaluate_classification
from datetime import datetime
from mixed_detection.BinaryClassifier import BinaryClassifier
from mixed_detection import vision_transforms as T
from torchvision import transforms as torchT
import wandb


def label_to_name(label):

    labels = {1:'NoduloMasa',
     2:'Consolidacion',
     3:'PatronIntersticial',
     4:'Atelectasia',
     5:'LesionesDeLaPared',
     6: 'Covid_Typical_Appearance',
     7: 'Covid_Indeterminate_Appearance',
     8: 'Covid_Atypical_Appearance'
     }
    return labels[label]


def infere(model,image,target,binary_classifier,plot_parameters,multitest=False):
    pred = None
    cont_pred = None
    #Inferencia en imagen original
    outputs = model(image)
    height = image[0].shape[1]
    width = image[0].shape[2]
    total_area = height * width
    outputs = [{k: v.to(torch.device("cpu")).detach() for k, v in t.items()} for t in outputs][0]

    outputs = ut.process_output(outputs, total_area, max_detections=None, min_box_proportionArea=None,
                                min_score_threshold=None)
    if "model" not in plot_parameters:

        outputs_plot = ut.process_output(outputs, total_area,
                                max_detections=plot_parameters["max_detections"],
                                min_box_proportionArea=plot_parameters["min_box_proportionArea"],
                                min_score_threshold=plot_parameters["min_score_threshold"])
    else:
        model_plot = plot_parameters['model']
        print('model plot')
        print(model_plot)
        outputs_plot = model_plot(image)
        outputs_plot = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in outputs_plot][0]

    if binary_classifier is not None:
        print('Calibrating')
        x_reg = ut.update_regression_features(outputs['scores'], outputs['areas'],
                                              n_features=binary_classifier.used_features)
        pred, cont_pred = binary_classifier.infere(x_reg.reshape(1, -1))
        cont_pred = cont_pred[0]
        pred = pred[0]

    if multitest:
        print('starting augmentation')
        colorjitter = torchT.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2)
        transforms = T.Compose([T.RandomHorizontalFlip(0.5), T.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2)])
        preds = np.empty(5)
        cont_preds = np.empty(5)
        for i in range(len(preds)-1):
            img, target = transforms(image, target)
            for j,img_ in enumerate(img):
                img[j] = colorjitter(img_)
            outp = model(img)
            outp = [{k: v.to(torch.device("cpu")).detach() for k, v in t.items()} for t in outp][0]

            outp = ut.process_output(outp, total_area, max_detections=None, min_box_proportionArea=None,
                                min_score_threshold=None)
            x_reg_i = ut.update_regression_features(outp['scores'], outp['areas'],n_features=binary_classifier.used_features)
            pred_i, cont_pred_i = binary_classifier.infere(x_reg_i.reshape(1, -1))
            print('pred_i',pred_i)
            print('cont_pred_i',cont_pred_i)
            cont_preds[i] = cont_pred_i[0]
            preds[i] = pred_i[0]
        #Agrego las de la imagen original
        preds[-1] = pred
        cont_preds[-1] = cont_pred
        print('Preds (la ultima es con imagen orignial)', preds)
        print('Cont preds (la ultima es con imagen orignial)',cont_preds)
        pred = 1 if np.sum(preds) >= 3 else 0
        cont_pred = np.mean(cont_preds)

    return outputs_plot, pred, cont_pred

def saveAsFiles(tqdm_loader,model,device,
                save_fig_dir,
                save_figures=None,   #puede ser 'heatmap','boxes',o None
                binary_classifier=None,posterior_th=None,
                plot_parameters = None, #dict
                draw='boxes',binary=False,
                save_csv=None, #If not None, should be a str with filepath where to save dataframe with targets and predictions
                multitest=False
                ):
    j = 0
    if plot_parameters is None:
        plot_parameters = {"max_detections": None,
                           "min_score_threshold": None, #if int, the same threshold for all classes. If dict, should contain one element for each class (key: clas_idx, value: class threshold)
                           "min_box_proportionArea": None
                           }
    if save_csv is not None:
        df = pd.DataFrame(columns=['image_name','box_type','label','score','x1','x2','y1','y2','original_file_name','image_source'])
    os.makedirs(save_fig_dir + 'TruePositive', exist_ok=True)
    os.makedirs(save_fig_dir + 'FalsePositive', exist_ok=True)
    os.makedirs(save_fig_dir + 'FalseNegative', exist_ok=True)
    os.makedirs(save_fig_dir + 'TrueNegative', exist_ok=True)
    for image, targets,image_sources,image_paths in tqdm_loader:

        image = list(img.to(device) for img in image)
        outputs, pred, cont_pred = infere(model,image,targets,binary_classifier,plot_parameters=plot_parameters,multitest=multitest)

        j += 1
        image_source = image_sources[0]
        image_path = image_paths[0].replace('/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/',
                                            '')
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        image = image[0].to(torch.device("cpu")).detach().numpy()

        targets = [{k: v.to(torch.device("cpu")).detach().numpy() for k, v in t.items()} for t in targets][0]

        # print('TARGET for {} is {}'.format(image_path, targets['labels']))
        folder = ''

        if pred is not None:
            gt = 1 if len(targets['labels']) > 0 else 0
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
            print('CONT PRED: {}, BINARY PRED: {} , GT: {}'.format(cont_pred, pred, gt))

        saving_path = "{}/{}/{}_{}".format(save_fig_dir, folder,
                                           image_source, os.path.basename(image_path.replace('\\', '/')))
        if cont_pred is not None:
            cont_pred_str = 100*cont_pred
            saving_path = saving_path.replace('.jpg', '_{:.1f}.jpg'.format(cont_pred_str))
        #print('Memory before save figres: ', psutil.virtual_memory().percent)
        image = image[0, :, :]
        if save_figures is not None:
            colorimage = np.zeros((image.shape[0], image.shape[1], 3), dtype=image.dtype)
            colorimage[:, :, 0] = image
            colorimage[:, :, 1] = image
            colorimage[:, :, 2] = image
            #print('Memory after colorimage: ', psutil.virtual_memory().percent)

            if save_figures=='heatmap':

                ut.save_heatmap(saving_path,colorimage,outputs)
                #print('Memory after save  heatmap: ', psutil.virtual_memory().percent)

            if save_figures=='boxes':
                colorimage = 255*colorimage
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
        #print('Memory after delete outputs: ', psutil.virtual_memory().percent)

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
        'test_set' : output_dir+'2021-07-30_binary/testCSV.csv',#trx_dir+'Tests/poc_cases.csv','{}/{}'.format(output_dir,'test_groundtruth_validados.csv'), #

        #'test_set' : '{}/{}'.format(output_dir,'2021-06-25_boxes_binary/testCSV.csv'), #output_dir+,#

        'experiment': '2021-07-30_binary',
        'experiment_type': 'boxes',
        'tested_epoch': 0,

        'opacityies_as_binary': True,
        'masks_as_boxes': True,


        'test_augmentation': True,
        'costs_ratio': 1/1, #Costo FP/CostoFN
        'expected_prevalence': 0.1,

        'calculate_coco': False,
        'calculate_classification': False,
        'save_figures': 'boxes',  #puede ser 'heatmap','boxes', o None
        'only_best_datasets': False,
        'save_csv': False,
        'view_in_window': False,
        'loop': False,

        'force_cpu': False,
    }
    print('starting test script')
    clf_from_old_model = False


    force_cpu = config['force_cpu'] #Lo que observe: al setearlo en true igual algo ahce con la gpu por ocupa ~1500MB,
    # pero si lo dejas en false ocupa como 4000MB. En cuanto a velocidad, el de gpu es mas rapido sin dudas, pero el cpu super tolerable (5segs por imagen aprox)

    chosen_experiment = config['experiment']
    chosen_epoch = config['tested_epoch']
    trainedModelPath = "{}/{}/fasterRCNN-{}.pth".format(output_dir, chosen_experiment, chosen_epoch)
    save_figures = config['save_figures']
    date = datetime.today().strftime('%Y-%m-%d')
    save_fig_dir = f'{output_dir}/{chosen_experiment}/test-{date}/detections_test_epoch-{chosen_epoch}_{save_figures}/'
    if config['save_csv']:
        output_csv_path = f'{output_dir}/{chosen_experiment}/test-{date}/test_output-epoch{chosen_epoch}.csv'
    else:
        output_csv_path = None
    results_coco_file = f'{output_dir}/{chosen_experiment}/test-{date}/cocoStats-test-epoch_{chosen_epoch}.txt'
    classification_data = f'{output_dir}/{chosen_experiment}/test_classification_data-{chosen_epoch}_orig'
    binary_opacity=config['opacityies_as_binary']

    os.makedirs(save_fig_dir,exist_ok=True)

    csv_test = pd.read_csv(config['test_set']).drop_duplicates('file_name')
    csv_train = pd.read_csv(output_dir+'2021-07-15_binary/trainCSV.csv')
    image_ids_test = set(csv_test.file_name)
    print('Images in test:{}. Instances total: {}'.format(len(image_ids_test),len(csv_test)))

    print('{} images to evaluate'.format(len(csv_test)))

    class_numbers = {'NoduloMasa': 1,
     'Consolidacion': 2,
     'PatronIntersticial': 3,
     'Atelectasia': 4,
     'LesionesDeLaPared': 5,
    'Covid_Typical_Appearance':6,
    'Covid_Indeterminate_Appearance': 7,
    'Covid_Atypical_Appearance': 8,
    }

    if force_cpu:
        device = torch.device('cpu')
        print('device cpu')
    else:
        torch.manual_seed(1)
        torch.cuda.synchronize()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if binary_opacity:
        num_classes = 2
    else:
        num_classes = len(class_numbers.keys())+1
    print('NUM CLASSES: ',num_classes)
    if config['experiment_type']=='boxes':
        # get the model using our helper function
        model = ut.get_object_detection_model(num_classes)
        model_plot = ut.get_object_detection_model(num_classes, box_score_thresh=0.19, box_nms_thresh=0.3,
                                                   box_detections_per_img=8)
    if config['experiment_type']=='masks':
        # get the model using our helper function
        model = ut.get_instance_segmentation_model(num_classes)
    model.to(device)
    model_plot.to(device)
    model_plot.load_state_dict(torch.load(trainedModelPath))
    model.load_state_dict(torch.load(trainedModelPath))
    #model = torch.load(trainedModelPath)
    model_plot.eval()
    model.eval()
    experiment_id = f"test_{date}_{chosen_experiment}"
    if binary_opacity:

        if clf_from_old_model:
            print('WILL CREATE CLASSIFIER')
            dataset_train = MixedLabelsDataset(csv_train, class_numbers,
                                              ut.get_transform(train=False),
                                              binary_opacity=binary_opacity,check_files=False,
                                              return_image_source=False)
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, batch_size=4, shuffle=False, num_workers=0,
                collate_fn=ut.collate_fn)
            clf = BinaryClassifier(expected_prevalence=config['expected_prevalence'], costs_ratio=config['costs_ratio'])
            print('Getting data...')
            clf.get_data_from_model(model,data_loader_train,device)
            print('Training classifier...')
            clf.train()
            print(clf.calibration_parameters)
            classification_data_dict = {}
            classification_data_dict['clf'] = clf
            with open(classification_data, 'wb') as f:
                pickle.dump(classification_data_dict, f)

    with wandb.init(config=config, project=project, name=experiment_id):
        config = wandb.config
        wandb_valid = {}
        if config['calculate_coco']:
            dataset_test = MixedLabelsDataset(csv_test, class_numbers, ut.get_transform(train=False),
                                              binary_opacity=binary_opacity,
                                              return_image_source=False)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1, shuffle=False, num_workers=0,
                collate_fn=ut.collate_fn)
            print('DATASET FOR COCO:')
            dataset_test.quantifyClasses()
            results_coco = evaluate_coco(model, data_loader_test, device=device,use_cpu=True,
                     results_file=results_coco_file)
            wandb_valid.update(results_coco)
        test_clf = None
        if binary_opacity:
            if os.path.exists(classification_data):
                with open(classification_data,'rb') as f:
                    classification_data_dict = pickle.load(f)
                print('Loaded binary model for classification')
                test_clf = classification_data_dict['clf']
                print('CLASSIFIER ', test_clf)
                if test_clf.costs_ratio!=config['costs_ratio'] or test_clf.expected_prevalence!=config['expected_prevalence']:
                    print('Reseting params')
                    test_clf.reset_params(config['expected_prevalence'],config['costs_ratio'])


            else:
                print('No se encontro clasificador binario',classification_data)
                return



        if config['calculate_classification']:
            dataset_test = MixedLabelsDataset(csv_test, class_numbers,
                                              ut.get_transform(train=False),colorjitter=False,
                                              masks_as_boxes=config['masks_as_boxes'],
                                              binary_opacity=binary_opacity,
                                              return_image_source=True)
            dataset_test.quantifyClasses()
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1, shuffle=False, num_workers=0,
                collate_fn=ut.collate_fn)
            results_classif = evaluate_classification(model, data_loader_test, device=device,log_wandb=False,
                                    results_file=results_coco_file,test_clf=test_clf,

                                                      )
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
                                                ut.get_transform(train=False), binary_opacity=binary_opacity,
                                                return_image_source=True)
        data_loader_test_files = torch.utils.data.DataLoader(
            dataset_test_files, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=ut.collate_fn)
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
        plot_parameters = {"max_detections": 7,
                           "min_score_threshold": None,#0.2 #if int, the same threshold for all classes. If dict, should contain one element for each class (key: clas_idx, value: class threshold)
                           "min_box_proportionArea": None,
                           "model": model_plot,#float(1/25) #Minima area de un box valido como proporcion del area total ej: al menos un cincuentavo del area total
                           }

        if config['save_figures'] or config['save_csv']:

            while saveAsFiles(tqdm_loader_files, model, device=device,save_fig_dir=save_fig_dir,
                              binary=binary_opacity,
                              plot_parameters=plot_parameters,
                              binary_classifier=test_clf,
                              save_csv=output_csv_path,save_figures=config['save_figures'],
                              multitest=config['test_augmentation']):
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
