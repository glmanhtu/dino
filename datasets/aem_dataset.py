import glob
import json
import os

import imagesize
import torch
from PIL import Image
from ml_engine.data.grouping import add_items_to_group
from ml_engine.data.samplers import MPerClassSampler
from torch.utils.data import Dataset, DataLoader


class AEMDataLoader:

    class FakeSampler:
        def set_epoch(self, _):
            return

    def __init__(self, datasets, batch_size, m, numb_workers, pin_memory, repeat, repeat_same_class):
        mini_batch_size = batch_size // len(datasets)
        max_dataset_length = max([len(x) for x in datasets]) * repeat
        self.dataloaders = []
        for dataset in datasets:
            sampler = MPerClassSampler(dataset.data_labels, m=m, length_before_new_iter=max_dataset_length,
                                       repeat_same_class=repeat_same_class)
            dataloader = DataLoader(dataset, sampler=sampler, pin_memory=pin_memory, batch_size=mini_batch_size,
                                    drop_last=True, num_workers=numb_workers)
            self.dataloaders.append(dataloader)
        self.sampler = AEMDataLoader.FakeSampler()

    def __len__(self):
        return max([len(x) for x in self.dataloaders])

    def __iter__(self):
        for samples in zip(*self.dataloaders):
            images, targets = None, None
            for item in samples:
                item_images, item_targets = item
                if images is None:
                    images = item_images
                    targets = item_targets
                else:
                    images = torch.cat([images, item_images], dim=0)
                    targets = torch.cat([targets, item_targets], dim=0)

            yield images, targets


class AEMLetterDataset(Dataset):
    def __init__(self, dataset_path: str, transforms, letter, min_size_limit):
        self.dataset_path = dataset_path
        files = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
        files.extend(glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True))

        tms = {}
        for file in files:
            file_name_components = os.path.basename(file).split('_')
            curr_letter, tm = file_name_components[0], file_name_components[1]
            tms.setdefault(tm, [])  # Ensure that we have all TMS for consistency between letters

            if curr_letter != letter:
                continue
            width, height = imagesize.get(file)
            if width < min_size_limit or height < min_size_limit:
                # Ignore extreme small images
                continue
            if '_ex.png' in file and os.path.exists(file.replace('_ex', '')):
                # Ignore duplicate samples
                continue

            tms.setdefault(tm, []).append(file)

        for tm in list(tms.keys()):
            if len(tms[tm]) < 2:
                tms[tm] = []

        self.labels = sorted(tms.keys())
        self.__label_idxes = {k: i for i, k in enumerate(self.labels)}

        self.data = []
        self.data_labels = []
        for tm in self.labels:
            for anchor in sorted(tms[tm]):
                self.data.append((tm, anchor))
                self.data_labels.append(self.__label_idxes[tm])

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (tm, anchor) = self.data[idx]

        with Image.open(anchor) as img:
            anchor_img = self.transforms(img.convert('RGB'))

        label = self.__label_idxes[tm]
        return anchor_img, label


def load_triplet_file(filter_file, with_likely=False):
    positive_groups, negative_pairs = [], {}
    positive_pairs = {}
    mapping = {}
    with open(filter_file) as f:
        triplet_filter = json.load(f)
    for item in triplet_filter['relations']:
        current_tm = item['category']
        mapping[current_tm] = {}
        for second_item in item['relations']:
            second_tm = second_item['category']
            relationship = second_item['relationship']
            mapping[current_tm][second_tm] = relationship

    for item in triplet_filter['histories']:
        current_tm, second_tm = item['category'], item['secondary_category']
        relationship = mapping[current_tm][second_tm]
        if relationship == 4 or relationship == 3:
            negative_pairs.setdefault(current_tm, set([])).add(second_tm)
            negative_pairs.setdefault(second_tm, set([])).add(current_tm)
        if relationship == 1:
            add_items_to_group([current_tm, second_tm], positive_groups)
        if with_likely and relationship == 2:
            positive_pairs.setdefault(current_tm, set([])).add(second_tm)
            positive_pairs.setdefault(second_tm, set([])).add(current_tm)

    for group in positive_groups:
        for tm in group:
            for tm2 in group:
                positive_pairs.setdefault(tm, set([])).add(tm2)

    return positive_pairs, negative_pairs
