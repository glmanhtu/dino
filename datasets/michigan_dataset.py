import glob
import math
import os
from enum import Enum
from typing import Union

import imagesize
from PIL import Image
from torch.utils.data import Dataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"

    @property
    def length(self) -> float:
        split_lengths = {
            _Split.TRAIN: 0.8,  # percentage of the dataset
            _Split.VAL: 0.2
        }
        return split_lengths[self]

    def is_train(self):
        return self.value == 'train'

    def is_val(self):
        return self.value == 'validation'

    @staticmethod
    def from_string(name):
        for key in _Split:
            if key.value == name:
                return key


class MichiganDataset(Dataset):
    Split = Union[_Split]

    def __init__(self, dataset_path: str, split: "MichiganDataset.Split", transforms, im_size, min_size=112):
        self.dataset_path = dataset_path
        files = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
        files.extend(glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True))

        image_map = {}
        for file in files:
            file_name_components = file.split(os.sep)
            im_name, rv, sum_det, _, im_type, _ = file_name_components[-6:]
            if rv != 'front':
                continue
            if im_type != 'papyrus':
                continue
            image_map.setdefault(im_name, {}).setdefault(sum_det, []).append(file)

        images = {}
        for img in image_map:
            key = 'detail'
            if key not in image_map[img]:
                key = 'summary'
            images[img] = image_map[img][key]

        self.labels = sorted(images.keys())
        self.__label_idxes = {k: i for i, k in enumerate(self.labels)}

        if split == MichiganDataset.Split.TRAIN:
            self.labels = self.labels[: int(len(self.labels) * split.length)]
        else:
            self.labels = self.labels[-int(len(self.labels) * split.length):]

        self.data = []
        self.data_labels = []
        for img in self.labels:
            data, labels = [], []
            for fragment in sorted(images[img]):
                width, height = imagesize.get(fragment)
                if width * height < min_size * min_size:
                    continue

                ratio = max(round((width * height) / (im_size * im_size)), 1) if split.is_train() else 1
                for _ in range(int(ratio)):
                    data.append((img, fragment))
                    labels.append(self.__label_idxes[img])

            if len(data) > 2:
                self.data.extend(data)
                self.data_labels.extend(labels)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (img_name, fragment) = self.data[idx]

        with Image.open(fragment) as img:
            image = self.transforms(img.convert('RGB'))

        label = self.data_labels[idx]
        return image, label
