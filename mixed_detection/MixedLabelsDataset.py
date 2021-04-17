import torch
from PIL import Image
import numpy as np
import pandas as pd




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
                    labels += [self.class_numbers[c] for c in raw_labels]

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
    def __init__(self, csv, class_numbers, transforms=None, return_image_source=False):
        self.csv = csv
        self.class_numbers = class_numbers
        self.transforms = transforms
        self.return_image_source = return_image_source
        assert pd.Series(['mask_path','label_level',
                          'x1','x2','y1','y2',
                          'class_name','image_source',
                          'file_name']).isin(self.csv.columns).all()



    def __getitem__(self, idx):
        img_path = self.csv.file_name.values[idx]

        image_source = self.csv.image_source.values[idx]

        img = Image.open(img_path.replace('\\','/')).convert("RGB")
        boxes = []
        target = {}
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
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if ymax > ymin and xmax > xmin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(raw_labels[i])
            masks = torch.as_tensor(masks, dtype=torch.uint8)

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
                        labels += [self.class_numbers[c] for c in raw_labels]

            img_shape = np.array(img).shape
            masks = torch.as_tensor(np.zeros((len(boxes),img_shape[0],img_shape[1])),
                                    dtype=torch.uint8) #Masks with all-zero elements will be considered as empty masks
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.as_tensor([], dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

        target["masks"] = masks
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)

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