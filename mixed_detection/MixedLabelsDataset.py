import torch
from PIL import Image
import numpy as np
import pandas as pd
import os, cv2
import psutil
from mixed_detection import vision_transforms as T
import imgaug as ia
import imgaug.augmenters as iaa
from torchvision import transforms as torchT

import albumentations
import numpy as np
class TestAugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, csv, return_image_source):
        self.csv = csv
        self.return_image_source = return_image_source

        assert pd.Series(['image_source',
                          'file_name']).isin(self.csv.columns).all()

        self.ids = list(set(self.csv.file_name))
        """
        self.aug = albumentations.Compose([
            albumentations.RandomCrop(256, 256),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            )
        ], p=1.)
        """
        self.aug = iaa.Sequential(
            [
             #iaa.CropAndPad(percent=(-0.2,0),pad_mode="constant",pad_cval=0),
             iaa.OneOf([
                        iaa.Affine(scale=(0.8, 1)),
                        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
                        iaa.Affine(rotate=(-5, 5))
                        ]),
            iaa.LinearContrast((0.5, 1))
            ])
        self.toTensor = T.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # converting jpg format of images to numpy array
        img_path = self.ids[index]
        img_row = self.csv[self.csv.file_name == img_path]
        if len(img_row)>1:
            img_row = img_row[0]
        image_source = img_row.image_source.values[0]
        img = np.array(Image.open(img_path.replace('\\','/')).convert('RGB'))
        # Applying augmentations to numpy array
        img = self.aug(image=img)
        img,_ = self.toTensor(img,target=None)
        # converting to pytorch image format & 2,0,1 because pytorch excepts image channel first then dimension of image
        #img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        if self.return_image_source:
            return img, image_source, img_path
        else:
            return img, np.array(img_row['class_name'])

class MixedLabelsDataset(torch.utils.data.Dataset):

    def __init__(self, csv, class_numbers, transforms=None,
                 return_image_source=False,binary_opacity=False,
                 masks_as_boxes=False, check_files=True):
        self.csv = csv
        self.class_numbers = class_numbers
        self.transforms = transforms #Transforms that have to be applied both to image and boxes/masks
        self.return_image_source = return_image_source
        self.binary = binary_opacity
        self.masks_as_boxes = masks_as_boxes

        assert pd.Series(['mask_path','label_level',
                          'x1','x2','y1','y2',
                          'class_name','image_source',
                          'file_name']).isin(self.csv.columns).all()
        if check_files:
            for i,row in self.csv.iterrows():
                path = row['file_name']
                try:
                    assert os.path.exists(path.replace('\\','/'))
                except AssertionError:
                    print(f"{path} does not exist, excluding from dataset")
                    index = self.csv[self.csv.file_name==path].index
                    self.csv = self.csv.drop(index,axis=0).reset_index(drop=True)
        self.ids = list(set(self.csv.file_name))


    def quantifyClasses(self):

        all_labels_strings = '-'.join([c if isinstance(c, str) else 'no_finding' for c in self.csv.class_name])
        all_labels = all_labels_strings.split('-')
        for name, c in self.class_numbers.items():
            print('N de {}: {} ({:.2f}%)'.format(name, all_labels.count(name),
                                                 100 * all_labels.count(name) / len(all_labels)))

        print('N de {}: {} ({:.2f}%)'.format('no_finding', all_labels.count('no_finding'),
                                             100 * all_labels.count('no_finding') / len(all_labels)))
    def __getitem__(self, idx):
        img_path = self.ids[idx]
        img_rows = self.csv[self.csv.file_name == img_path]
        image_source = img_rows.image_source.values[0]

        #img = Image.open(img_path.replace('\\','/')).convert("RGB")
        img = cv2.imread(img_path.replace('\\','/'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = []
        target = {}
        """
        try:
            print(self.csv.box_type.values[idx])
        except:
            pass"""
        if isinstance(img_rows.mask_path.values[0],str):

            mask_path = img_rows.mask_path.values[0]
            # each color corresponds to a different instance with 0 being background
            with open(mask_path, 'rb') as f:
                mask = np.load(f)

            # instances are encoded as different colors
            obj_ids = np.unique(mask)[1:]
            labels=[]
            # split the color-encoded mask into a set of binary masks
            masks = mask == obj_ids[:, None, None]
            raw_labels = [self.class_numbers[c] for c in img_rows.class_name.values[0].split('-')]
            # get bounding box coordinates for each mask

            num_objs = len(obj_ids)
            if len(raw_labels) < num_objs:
                print(mask_path,raw_labels)
            if num_objs!=len(raw_labels):
                print('ERROR IN IMAGE {} with {} objects in mask and {} labels'.format(mask_path,num_objs,len(raw_labels)))
            for i in range(min(num_objs,len(raw_labels))):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if ymax > ymin and xmax > xmin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    if self.binary:
                        labels.append(1)
                    else:
                        labels.append(raw_labels[i])


            #del masks, mask
            if self.masks_as_boxes:
                #print("Using mask {} as box".format(img_path))
                masks = np.zeros((len(boxes), img.shape[0], img.shape[1]))

        else:
            labels = []

            for i,row in img_rows.iterrows():
                xmin = row['x1']
                xmax = row['x2']
                ymin = row['y1']
                ymax = row['y2']
                if ymax > ymin and xmax > xmin:
                    boxes.append([xmin, ymin, xmax, ymax])
                if isinstance(row['class_name'],str):
                    if len(row['class_name'])>0:
                        raw_labels = row['class_name'].split('-')
                        if self.binary:
                            if row['class_name'] == 'nofinding':
                                labels=[0]
                            elif row['class_name'] == 'Opacidad' or row['class_name'] =='Consolidacion':
                                labels=[1]
                            else:
                                labels += [1]*len(raw_labels)
                        else:
                            labels += [self.class_numbers[c] for c in raw_labels]

            masks = np.zeros((len(boxes),img.shape[0],img.shape[1])) #Masks with all-zero elements will be considered as empty masks

        masks_tensor = torch.as_tensor(masks, dtype=torch.uint8)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        del boxes
        if len(boxes_tensor) > 0:
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        else:
            area = torch.as_tensor([], dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

        target["masks"] = masks_tensor
        target["boxes"] = boxes_tensor
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #print('Memory before transforms: %', psutil.virtual_memory().percent)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.return_image_source:
            return img, target, image_source, img_path
        else:
            return img, target

    def __len__(self):
        return len(self.ids)




class MixedSampler(torch.utils.data.Sampler):
    def __init__(self, folds_distr_path, fold_index, demand_non_empty_proba):
        assert demand_non_empty_proba > 0, 'frequensy of non-empty images must be greater then zero'
        self.fold_index = fold_index
        self.positive_proba = demand_non_empty_proba

        self.folds = pd.read_csv(folds_distr_path)
        self.folds.fold = self.folds.fold.astype(str)
        self.folds = self.folds[self.folds.fold != fold_index].reset_index(drop=True)

        self.positive_idxs = self.folds[self.folds.exist_labels == 1].index.values
        self.negative_idxs = self.folds[self.folds.exist_labels == 0].index.values

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)

    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative