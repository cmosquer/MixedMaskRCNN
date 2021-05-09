import os, random
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
from mixed_detection.engine import evaluate
def label_to_name(label):
    labels = {1:'NoduloMasa',
     2:'Consolidacion',
     3:'PatronIntersticial',
     4:'Atelectasia',
     5:'LesionesDeLaPared'
     }
    return labels[label]


def main(args=None):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    baseDir = '/run/user/1000/gvfs/smb-share:server=lxestudios.hospitalitaliano.net,share=pacs/T-Rx/'
    # use our dataset and defined transformations

    output_dir = baseDir +'TRx-v2/Experiments/'
    chosen_experiment = '06-05-21'
    modelsForCompare = [chosen_experiment+'_masksOnly_binary',chosen_experiment+'_masksAndBoxs_binary']
    existing_test_set = '{}/{}_masksAndBoxs_binary/testCSV.csv'.format(output_dir,chosen_experiment)
    csv_test = pd.read_csv(existing_test_set)
    print('{} images to evaluate'.format(len(set(csv_test.file_name))))

    class_numbers = {'NoduloMasa': 1,
     'Consolidacion': 2,
     'PatronIntersticial': 3,
     'Atelectasia': 4,
     'LesionesDeLaPared': 5
     }
    Nepochs = 10
    torch.manual_seed(1)
    dataset_test = MixedLabelsDataset(csv_test, class_numbers, get_transform(train=False),
                                      return_image_source=False)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    print('DATASET FOR COCO:')
    dataset_test.quantifyClasses()
    num_classes = len(class_numbers.keys()) + 1
    model = get_instance_segmentation_model(num_classes)
    #PRIMERO COMPUTAR DICES
    overall_fig, overall_ax = plt.subplots(1, 1, figsize=(15, 8))
    for j,dir in enumerate(modelsForCompare):
        expDir = output_dir+dir
        precisions={}
        precisions['bbox'] = {'precision_iou50-95': np.zeros((Nepochs)),
                      'precision_iou50': np.zeros((Nepochs)),
                      'precision_iou75': np.zeros((Nepochs)),
                      'precision_iou50-95_small': np.zeros((Nepochs)),
                      'precision_iou50-95_medium': np.zeros((Nepochs)),
                      'precision_iou50-95_large': np.zeros((Nepochs)),
                      }
        precisions['segm'] = {'precision_iou50-95': np.zeros((Nepochs)),
                      'precision_iou50': np.zeros((Nepochs)),
                      'precision_iou75': np.zeros((Nepochs)),
                      'precision_iou50-95_small': np.zeros((Nepochs)),
                      'precision_iou50-95_medium': np.zeros((Nepochs)),
                      'precision_iou50-95_large': np.zeros((Nepochs)),
                      }
        recalls={}
        recalls['bbox'] = {'recall_100det': np.zeros((Nepochs)),
                   'recall_1det': np.zeros((Nepochs)),
                   'recall_10det': np.zeros((Nepochs)),
                   'recall_100det_small': np.zeros((Nepochs)),
                   'recall_100det_medium': np.zeros((Nepochs)),
                   'recall_100det_large': np.zeros((Nepochs)),
                   }
        recalls['segm'] = {'recall_100det': np.zeros((Nepochs)),
                   'recall_1det': np.zeros((Nepochs)),
                   'recall_10det': np.zeros((Nepochs)),
                   'recall_100det_small': np.zeros((Nepochs)),
                   'recall_100det_medium': np.zeros((Nepochs)),
                   'recall_100det_large': np.zeros((Nepochs)),
                   }
        for epoch in range(Nepochs):
            results_coco_file = f'{expDir}/cocoStats-test-epoch_{epoch}.txt'


            with open(results_coco_file, 'r') as f:
                txt = f.read()
            tasks = {}
            tasks['bbox'] = txt.split('IOU: segm')[0]
            tasks['segm'] = txt.split('IOU: segm')[-1]
            for task,txt_ in tasks.items():

                vals = [float(f[:6]) for f in txt_.split('] :')[1:]]
                precisions[task]['precision_iou50-95'][epoch] = float(txt_.split(']:')[-1][:6])
                precisions[task]['precision_iou50'][epoch] = vals[0]
                precisions[task]['precision_iou75'][epoch] = vals[1]
                precisions[task]['precision_iou50-95_small'][epoch] = vals[2]
                precisions[task]['precision_iou50-95_medium'][epoch] = vals[3]
                precisions[task]['precision_iou50-95_large'][epoch] = vals[4]

                recalls[task]['recall_1det'][epoch] = vals[5]
                recalls[task]['recall_10det'][epoch] = vals[6]
                recalls[task]['recall_100det'][epoch] = vals[7]
                recalls[task]['recall_100det_small'][epoch] = vals[8]
                recalls[task]['recall_100det_medium'][epoch] = vals[9]
                recalls[task]['recall_100det_large'][epoch] = vals[10]
        for task in ['bbox','segm']:
            lines = ['-', '--', '-.', ':', '--', '-']
            widths = [2, 2, 1, 1, 1, 1]
            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
            j = 0
            for name, val in recalls.items():
                ax.plot(range(Nepochs), val, label=name, color='blue', ls=lines[j], lw=widths[j])
                j += 1
            j = 0
            for name, val in precisions.items():
                ax.plot(range(Nepochs), val, label=name, color='red', ls=lines[j], lw=widths[j])
                j += 1
            ax.legend()
            ax.set_ylim((0, 0.35))

            fig.savefig(f'{expDir}/detailed{task}.jpg')


        overall_ax.plot(range(Nepochs),precisions['bbox']['precision_iou50-95'],label=f'{dir}-precision-boxes',
                        color='red',ls=lines[j],width=1)
        overall_ax.plot(range(Nepochs), precisions['segm']['precision_iou50-95'], label=f'{dir}-precision-segm',
                        color='red',ls=lines[j],width=2)

        overall_ax.plot(range(Nepochs),precisions['bbox']['recall_100det'],label=f'{dir}-recall-boxes',
                        color='blue',ls=lines[j],width=1)
        overall_ax.plot(range(Nepochs), precisions['segm']['recall_100det'], label=f'{dir}-recall-segm',
                        color='blue',ls=lines[j],width=2)

        dices = np.zeros((Nepochs))
        for epoch in range(Nepochs):
            trainedModelPath = "{}/mixedMaskRCNN-{}.pth".format(expDir,epoch)


            model.load_state_dict(torch.load(trainedModelPath))
            model.to(device)
            model.eval()
            print('Model loaded')
            dice_avg = evaluate(model, data_loader_test, device=device, coco=False,dice=True,
                     results_file=results_coco_file)

            dices[epoch]=dice_avg

        overall_ax.plot(range(Nepochs), dices, label=f'{dir}-DICE',
                        color='green',ls=lines[j],width=2)
    overall_fig.savefig(output_dir+'overall_fig.svg')

if __name__ == '__main__':
    main()
