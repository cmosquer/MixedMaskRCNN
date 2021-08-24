import cv2
import numpy as np
import os
import pandas as pd
from mixed_detection.visualization import draw_annotations, draw_masks
import torch
import mixed_detection.utils as ut
from tqdm import tqdm
def label_to_name(label,binary=True):
    if binary:
        labels = {1: "Opacidad", 0: "Sin hallazgo"}

    else:
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


def getOutputForPlot(plot_parameters,image, model=None):
    if "model" not in plot_parameters:
        assert model is not None, "Must pass model as argument or as value in plot_parameters dict"
        outputs_plot = model(image)
        outputs_plot = [{k: v.to(torch.device("cpu")).detach() for k, v in t.items()} for t in outputs_plot[0]]

        height = image[0].shape[1]
        width = image[0].shape[2]
        total_area = height * width
        outputs_plot = ut.process_output(outputs_plot[0], total_area,
                                max_detections=plot_parameters["max_detections"],
                                min_box_proportionArea=plot_parameters["min_box_proportionArea"],
                                min_score_threshold=plot_parameters["min_score_threshold"]).numpy()
    else:
        model_plot = plot_parameters['model']
        outputs_plot = model_plot(image)[0]
        outputs_plot = dict(zip(outputs_plot.keys(), [val.to('cpu').detach() for val in outputs_plot.values()]))
    return outputs_plot


def infereImage(model,image,binary_classifier=None):

    outputs = model(image)
    assert len(outputs)==1 , "Must use one-sample batchs for testing"
    outputs = outputs[0]
    outputs = dict(zip(outputs.keys(), [val.to('cpu').detach() for val in outputs.values()]))

    #outputs = [{k: v.to(torch.device("cpu")).detach() for k, v in t.items()} for t in outputs]
    image  = image.to('cpu').detach()
    height = image[0].shape[1]
    width = image[0].shape[2]
    total_area = height * width
    outputs = ut.process_output(outputs, total_area,
                                 max_detections=None, min_box_proportionArea=None,
                                 min_score_threshold=None)

    if binary_classifier is not None:
        x_reg = ut.update_regression_features(outputs['scores'], outputs['areas'],
                                                n_features=binary_classifier.used_features)
        pred, cont_pred = binary_classifier.infere(x_reg.reshape(1, -1))
        pred = pred[0]
        cont_pred = cont_pred[0]
        if cont_pred<1e-4:
            cont_pred=0
    else:
        pred = 1 if len(outputs['boxes']>0) else 0
        cont_pred = outputs['scores'][0] if len(outputs['scores'])>0 else 0
        x_reg = None
    return pred, cont_pred, x_reg, outputs


def testAugmented(loader_augmented,N_augms,model,device,binary_classifier,dfPreds):

    for j in range(N_augms):
        print('AUGMENTATION ',j)
        dfPreds['binary_pred_augm' + str(j)] = [None]*len(loader_augmented)
        dfPreds['cont_pred_augm' + str(j)] = [None]*len(loader_augmented)
        for x in range(binary_classifier.used_features):
            dfPreds['x_reg_{}_augm{}'.format(x, j)] = [None]*len(loader_augmented)

        for image,image_source,image_path in tqdm(loader_augmented):
            assert len(image) == 1, "Must use one-sample batchs for testing"
            image = image.to(device)

            pred, cont_pred, x_reg, outputs = infereImage(model, image, binary_classifier)

            saving_path = "/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/TRx-v2/Tests/TTA/{}_{}.jpg".format(os.path.basename(image_path[0]),j)
            image = image[0].cpu().numpy()  # Squeeze el batch
            colorimage = np.float32(np.moveaxis(image, 0, -1).copy())
            cv2.imwrite(saving_path, colorimage)

            assert len(dfPreds[dfPreds.image_name == os.path.basename(image_path[0])])==1, "Couldnt find image in dfPreds"
            dfPreds.at[dfPreds.image_name==os.path.basename(image_path[0]),'binary_pred_augm'+str(j)] = pred
            dfPreds.at[dfPreds.image_name==os.path.basename(image_path[0]),'cont_pred_augm'+str(j)] = cont_pred
            for x,val in enumerate(x_reg):
                dfPreds.at[dfPreds.image_name==os.path.basename(image_path[0]),'x_reg_{}_augm{}'.format(x,j)] = val

    bin_cols = [bin_col for bin_col in dfPreds.columns if 'binary_pred' in bin_col]
    cont_cols = [cont_col for cont_col in dfPreds.columns if 'cont_pred' in cont_col]
    dfPreds['averaged_binary_pred'] = [1 if v > 0.5 else 0 for v in dfPreds[bin_cols].mean(axis=1)]
    dfPreds['averaged_cont_pred'] = dfPreds[cont_cols].mean(axis=1)

    return dfPreds


def plotOriginals(loader_originals, device, dfPreds,
                save_fig_dir=None,
                save_figures=None,   #puede ser 'heatmap','boxes','masks',
                plot_parameters = None, #dict
                binary=True,
                ):
    if plot_parameters is None:
        plot_parameters = {"max_detections": None,
                           "min_score_threshold": None, #if int, the same threshold for all classes. If dict, should contain one element for each class (key: clas_idx, value: class threshold)
                           "min_box_proportionArea": None
                           }
    os.makedirs(save_fig_dir + 'TruePositive', exist_ok=True)
    os.makedirs(save_fig_dir + 'FalsePositive', exist_ok=True)
    os.makedirs(save_fig_dir + 'FalseNegative', exist_ok=True)
    os.makedirs(save_fig_dir + 'TrueNegative', exist_ok=True)
    for image, target,image_source,image_path in loader_originals:
        assert len(image)== 1, "Must use one-sample batchs for testing"
        image = image.to(device)
        target = dict(zip(target.keys(),[val[0].to('cpu').detach() for val in target.values()]))
        outputs = getOutputForPlot(plot_parameters,image)
        folder = ''
        imgpath = os.path.basename(image_path[0])
        img_row = dfPreds[dfPreds.image_name == imgpath]
        pred = img_row['averaged_binary_pred'].values[0]
        cont_pred = img_row['averaged_cont_pred'].values[0]

        if pred is not None:
            gt = 1 if len(target['labels']) > 0 else 0
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
        if save_fig_dir is not None:
            saving_path = "{}/{}/{}".format(save_fig_dir, folder,
                                            #image_source[0],
                                            os.path.basename(image_path[0].replace('\\', '/')))
            dfPreds.at[dfPreds.image_name == os.path.basename(image_path[0]),'output_file'] = saving_path
            cont_pred_str = 100*cont_pred
            saving_path = saving_path.replace('.jpg', ' SCORE-{:.1f}.jpg'.format(cont_pred_str))
            #print('Memory before save figres: ', psutil.virtual_memory().percent)
            image = image[0].cpu().numpy() #Squeeze el batch
            colorimage = np.float32(np.moveaxis(image,0,-1).copy()) #Channels-first to channels-last
            #print('Memory after colorimage: ', psutil.virtual_memory().percent)
            if save_figures=='heatmap':
                ut.save_heatmap(saving_path,colorimage,outputs)
                #print('Memory after save  heatmap: ', psutil.virtual_memory().percent)

            if save_figures=='boxes' or save_figures=='masks':
                colorimage = 255*colorimage
                if len(outputs['labels']) > 0:
                    if save_figures=='boxes':
                        draw_annotations(colorimage, outputs, color=(0, 255, 0),label_to_name=label_to_name,binary=binary)
                    if save_figures=='masks':
                        draw_masks(colorimage, outputs, label_to_name=label_to_name,binary=binary)

                    # draw annotations on the image
                if len(target['boxes'])==0:
                    caption = 'Image GT: opacidad'#+','.join([label_to_name(lbl) for lbl in targets['labels']])
                    cv2.putText(colorimage, caption,
                            (10, 10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), thickness=2)
                else:
                    # draw annotations in red
                    if save_figures=='boxes':
                        draw_annotations(colorimage, target, color=(255, 0, 0),
                                         label_to_name=label_to_name,binary=binary)
                    if save_figures=='masks':
                        draw_masks(colorimage, target,
                                   color=(255, 0, 0),label_to_name=label_to_name)


                cv2.imwrite(saving_path, colorimage)

            print('Saved ', saving_path)

        del outputs,target, image
    return dfPreds




def testOriginals(loader_originals, model, device,
                binary_classifier=None,binary=True,
                save_boxes_csv=None, #If not None, should be a str with filepath where to save dataframe with targets and predictions
                ):

    if save_boxes_csv is not None:
        dfBoxes = pd.DataFrame(columns=['image_name','box_type','label','score','x1','x2','y1','y2','original_file_name','image_source'])
        dfPreds = pd.DataFrame()
        print('CREATED DATAFRAMES')

    for image, target,image_source,image_path in tqdm(loader_originals):
        assert len(image) == 1, "Must use one-sample batchs for testing"
        image = image.to(device)
        target = dict(zip(target.keys(),[val[0].to('cpu').detach() for val in target.values()]))
        gt = 1 if len(target['labels']) > 0 else 0
        pred, cont_pred, x_reg, outputs = infereImage(model,image,binary_classifier)
        if save_boxes_csv is not None:
            results_list = []

            for i,label in enumerate(outputs['labels']):
                result = {'image_name': os.path.basename(image_path[0]),
                          'box_type':'prediction',
                          'label': label_to_name(label.item(),binary=binary),
                          'score': outputs['scores'][i].item(),
                          'x1': int(outputs['boxes'][i][0]),
                          'y1': int(outputs['boxes'][i][1]),
                          'x2': int(outputs['boxes'][i][2]),
                          'y2': int(outputs['boxes'][i][3]),
                          'overall_binary_pred': pred,
                          'overall_cont_pred': cont_pred,
                          'original_file_name': image_path[0],
                          'image_source':image_source[0]
                          }
                results_list.append(result)

            for i, label in enumerate(target['labels']):
                result = {'image_name': os.path.basename(image_path[0]),
                          'box_type': 'ground-truth',
                          'original_file_name': image_path[0],
                          'image_source': image_source[0],
                          'label': label_to_name(label.item(),binary=binary),
                          'x1': int(target['boxes'][i][0]),
                          'y1': int(target['boxes'][i][1]),
                          'x2': int(target['boxes'][i][2]),
                          'y2': int(target['boxes'][i][3])}
                results_list.append(result)
            if len(results_list)==0:
                results_list = [{'image_name': os.path.basename(image_path[0]),
                          'box_type': 'true-negative',
                          'original_file_name': image_path[0],
                         'overall_binary_pred': pred,
                         'overall_cont_pred': cont_pred,
                          'image_source': image_source[0]
                          }]
            dfBoxes = dfBoxes.append(results_list,ignore_index=True)


        results_preds = { 'image_name': os.path.basename(image_path[0]),
                          'gt': gt,
                          'image_source': image_source[0],
                          'binary_pred_original': pred,
                          'cont_pred_original': cont_pred,
                          'original_file_name': image_path[0],
                          }
        for x,val in enumerate(x_reg):
            results_preds["x_reg_"+str(x)+"_original"] = val
        dfPreds = dfPreds.append(results_preds,ignore_index=True)
        del outputs,target, image

    if save_boxes_csv is not None:
        dfBoxes.to_csv(save_boxes_csv+'_boxes.csv',index=False)


    return dfPreds

