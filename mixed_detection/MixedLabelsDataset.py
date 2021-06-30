import torch
from PIL import Image
import numpy as np
import pandas as pd
import os, cv2

from torchvision import transforms as torchT

class ImageLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, csv, class_numbers, transforms=None, return_image_source=False):
        self.csv = csv
        self.class_numbers = class_numbers
        self.transforms = transforms
        self.return_image_source = return_image_source
        assert pd.Series(['class_name','image_source',
                          'file_name']).isin(self.csv.columns).all()
    def quantifyClasses(self):
        all_labels_strings = '-'.join([c if isinstance(c,str) else 'no_finding' for c in self.csv.class_name])
        all_labels = all_labels_strings.split('-')
        for name,c in self.class_numbers.items():
            print('N de {}: {} ({:.2f}%)'.format(name,all_labels.count(name),100*all_labels.count(name)/len(all_labels)))

        print('N de {}: {} ({:.2f}%)'.format('no_finding', all_labels.count('no_finding'),
                                             100 * all_labels.count('no_finding') / len(all_labels)))
    def __getitem__(self, idx):
        img_path = self.csv.file_name.values[idx]

        image_source = self.csv.image_source.values[idx]

        img = Image.open(img_path.replace('\\','/')).convert("RGB")
        img_rows = self.csv[self.csv.file_name == img_path]
        labels = []
        for i, row in img_rows.iterrows():
            if isinstance(row['class_name'], str):
                if len(row['class_name']) > 0:
                    raw_labels = row['class_name'].split('-')
                    labels += [self.class_numbers[c] for c in raw_labels if c in self.class_numbers.keys()]

        if self.transforms is not None:
            img = self.transforms(img)

        labels_tensor = torch.zeros(len(self.class_numbers), dtype=torch.float32)
        labels_tensor[labels] = 1

        if self.return_image_source:
            return (img, labels_tensor, image_source, img_path)
        else:
            return (img, labels_tensor)


    def __len__(self):
        return len(self.csv)


class MixedLabelsDataset(torch.utils.data.Dataset):

    def __init__(self, csv, class_numbers, transforms=None, colorjitter=False,
                 return_image_source=False,binary_opacity=False,
                 masks_as_boxes=False):
        self.csv = csv
        self.class_numbers = class_numbers
        self.transforms = transforms #Transforms that have to be applied both to image and boxes/masks
        self.colorjitter = colorjitter #Transform the image brightness,contrast,etc

        self.return_image_source = return_image_source
        self.binary = binary_opacity
        self.masks_as_boxes = masks_as_boxes


        assert pd.Series(['mask_path','label_level',
                          'x1','x2','y1','y2',
                          'class_name','image_source',
                          'file_name']).isin(self.csv.columns).all()
        for i,row in self.csv.iterrows():
            path = row['file_name']
            try:
                assert os.path.exists(path.replace('\\','/'))
            except AssertionError:
                f"{path} does not exist, excluding from dataset"
                self.csv = self.csv[~self.csv.file_name==path].reset_index(drop=True)
    def quantifyClasses(self):
        all_labels_strings = '-'.join([c if isinstance(c, str) else 'no_finding' for c in self.csv.class_name])
        all_labels = all_labels_strings.split('-')
        for name, c in self.class_numbers.items():
            print('N de {}: {} ({:.2f}%)'.format(name, all_labels.count(name),
                                                 100 * all_labels.count(name) / len(all_labels)))

        print('N de {}: {} ({:.2f}%)'.format('no_finding', all_labels.count('no_finding'),
                                             100 * all_labels.count('no_finding') / len(all_labels)))
    def __getitem__(self, idx):
        img_path = self.csv.file_name.values[idx]

        image_source = self.csv.image_source.values[idx]

        #img = Image.open(img_path.replace('\\','/')).convert("RGB")
        img = cv2.imread(img_path.replace('\\','/'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = []
        target = {}
        try:
            print(self.csv.box_type.values[idx])
        except:
            pass
        if isinstance(self.csv.mask_path[idx] ,str):

            mask_path = self.csv.mask_path.values[idx]
            # each color corresponds to a different instance with 0 being background
            with open(mask_path, 'rb') as f:
                mask = np.load(f)

            # instances are encoded as different colors
            obj_ids = np.unique(mask)[1:]
            labels=[]
            # split the color-encoded mask into a set of binary masks
            masks = mask == obj_ids[:, None, None]
            raw_labels = [self.class_numbers[c] for c in self.csv.class_name.values[idx].split('-')]
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
                print("Using mask {} as box".format(img_path))
                masks = np.zeros((len(boxes), img.shape[0], img.shape[1]))

        else:
            labels = []
            img_rows = self.csv[self.csv.file_name==img_path]
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
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        if self.colorjitter:
            img = torchT.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2)(img)

        if self.return_image_source:
            return img, target, image_source, img_path
        else:
            return img, target

    def __len__(self):
        return len(self.csv)




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