import torch.utils.data
import pickle
from tqdm import tqdm
from mixed_detection.MixedLabelsDataset import MixedLabelsDataset, TestAugmentationDataset
from mixed_detection.engine import evaluate_coco, evaluate_classification
from datetime import datetime
from mixed_detection.BinaryClassifier import BinaryClassifier
from matplotlib import pyplot as plt
import wandb
import pandas as pd

from mixed_detection.test_utils import *



def main(args=None):
    project = "mixed_mask_rcnn"
    trx_dir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/TRx-v2/'
    output_dir = trx_dir+'Experiments/'

    config = {
        'test_set' : trx_dir+'Tests/poc_cases_with_trxv1.csv', #output_dir+'2021-07-30_binary/testCSV.csv',#'{}/{}'.format(output_dir,'test_groundtruth_validados.csv'), #

        #'test_set' : '{}/{}'.format(output_dir,'2021-06-25_boxes_binary/testCSV.csv'), #output_dir+,#

        'experiment': '2021-07-30_binary',
        'experiment_type': 'boxes',
        'tested_epoch': 0,

        'opacityies_as_binary': True,
        'masks_as_boxes': True,


        'test_augmentation': 4,
        'costs_ratio': 1/1, #Costo FP/CostoFN
        'expected_prevalence': 0.1,

        'calculate_coco': False,
        'calculate_classification': False,
        'save_figures': 'boxes',  #puede ser 'heatmap','boxes', o None
        'only_best_datasets': False,
        'view_in_window': False,
        'loop': False,

        'force_cpu': False,

        'save_comparison_trx_v1':True
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
    output_csv_path = f'{output_dir}/{chosen_experiment}/test-{date}/epoch{chosen_epoch}'

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
    model.load_state_dict(torch.load(trainedModelPath))
    #model = torch.load(trainedModelPath)
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
                                              ut.get_transform(train=False),
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


        dataset_test_originals = MixedLabelsDataset(csv_test_files, class_numbers,
                                              ut.get_transform(train=False),
                                              masks_as_boxes=config['masks_as_boxes'],
                                              binary_opacity=binary_opacity,
                                              return_image_source=True)
        data_loader_test_files = torch.utils.data.DataLoader(
            dataset_test_originals, batch_size=1, shuffle=False, num_workers=0,
            #collate_fn=ut.collate_fn #As it is only one image, we do not need collate fn
            )

        dfPreds = testOriginals(data_loader_test_files, model, device=device, binary_classifier=test_clf,
                              save_boxes_csv=output_csv_path, binary=binary_opacity,
                             )
        dfPreds.to_csv(output_csv_path+'_preds.csv',index=False)
        #dfPreds = pd.read_csv(output_csv_path + '_preds.csv')
        if config['test_augmentation'] > 0:
            print('starting tta')
            dataset_test_aug = TestAugmentationDataset(csv_test_files,
                                                  return_image_source=True)
            augm_data_loader_test_files = torch.utils.data.DataLoader(
                dataset_test_aug, batch_size=1, shuffle=False, num_workers=0,
                )

            dfPreds = testAugmented(augm_data_loader_test_files,config['test_augmentation'],
                                    model,device,binary_classifier=test_clf,dfPreds=dfPreds)
        
        dfPreds.to_csv(output_csv_path+'_preds.csv',index=False)
        


        del model
        if config['save_figures'] is not None:
            model_plot.to(device)
            model_plot.load_state_dict(torch.load(trainedModelPath))
            model_plot.eval()
            plot_parameters = {"max_detections": 7,
                               "min_score_threshold": None,#0.2 #if int, the same threshold for all classes. If dict, should contain one element for each class (key: clas_idx, value: class threshold)
                               "min_box_proportionArea": None,
                               "model": model_plot,#float(1/25) #Minima area de un box valido como proporcion del area total ej: al menos un cincuentavo del area total
                               }
            dfPreds = plotOriginals(data_loader_test_files,device,dfPreds,
                          save_fig_dir=save_fig_dir,
                          plot_parameters=plot_parameters, binary=binary_opacity,
                          save_figures=config['save_figures'])

        dfPreds.to_csv(output_csv_path+'_preds.csv',index=False)

        #dfPreds = pd.read_csv(output_csv_path + '_preds.csv')
        if config['save_comparison_trx_v1']:
            assert 'trx_v1_heatmap' in csv_test.columns
            assert 'trx_v1_cont_pred' in csv_test.columns
            assert 'trx_v1_binary_pred' in csv_test.columns
            saving_dir = f'{output_dir}/{chosen_experiment}/test-{date}/comparison_trxv1/'
            os.makedirs(saving_dir,exist_ok=True)

            csv_test['image_name'] = [path.replace('\\','/') for path in csv_test.image_name]
            for i,row in dfPreds.iterrows():
                imagename = row['image_name'].replace('\\','/')
                row_csv_test = csv_test[csv_test.image_name==imagename]
                print(len(row_csv_test))
                dfPreds.loc[i,'gt'] = 0 if row_csv_test['class_name'].values[0]=='nofinding' else 1

                an = row_csv_test.accessionNumber
                trx1pred = 'CON OPACIDAD' if bool(row_csv_test.trx_v1_binary_pred.values[0]) else 'SIN OPACIDAD'
                trx1score = 100*float(row_csv_test.trx_v1_cont_pred.values[0])
                trx2pred = 'CON OPACIDAD' if bool(row['averaged_binary_pred']) else 'SIN OPACIDAD'
                trx2score = 100*float(row['averaged_cont_pred'])

                img2 = cv2.imread(row['output_file'])
                print(img2.shape)
                print(row_csv_test['trx_v1_heatmap'].values[0])


                assert os.path.exists(row_csv_test['trx_v1_heatmap'].values[0])
                img1 = cv2.imread(row_csv_test['trx_v1_heatmap'].values[0])
                print(img1.shape)
                fig,axs = plt.subplots(1,2,figsize=(18,9))
                axs[0].imshow(img1)
                axs[0].set_title('TRx v1')
                axs[0].set_xlabel(f'{trx1pred}\n{trx1score}%')
                axs[0].spines['bottom'].set_visible(False)
                axs[0].spines['top'].set_visible(False)
                axs[0].spines['right'].set_visible(False)
                axs[0].spines['left'].set_visible(False)
                gt = 'Sin opacidad' if row_csv_test['class_name'].values[0]=='nofinding' else 'Con opacidad'
                fig.suptitle(f"AN: {an}\nID:{row['sopInstanceUid']}\nGround truth: {gt}")
                axs[1].imshow(img2)
                axs[1].set_title('TRx v2')
                axs[1].set_xlabel(f'{trx2pred}\n{trx2score}%')
                axs[1].spines['bottom'].set_visible(False)
                axs[1].spines['top'].set_visible(False)
                axs[1].spines['right'].set_visible(False)
                axs[1].spines['left'].set_visible(False)
                fig.savefig(saving_dir+f"{an}.jpg")

                plt.cla()
                


        wandb.log(wandb_valid)
if __name__ == '__main__':
    main()
