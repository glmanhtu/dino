import glob
import os
from enum import Enum
from typing import Union

from PIL import Image
from torch.utils.data import Dataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "validation"
    ALL = "all"

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

    def __init__(self, dataset_path: str, split: "MichiganDataset.Split", transforms):
        self.dataset_path = dataset_path
        files = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
        files.extend(glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True))

        image_map = {}
        for file in files:
            file_name_components = file.split(os.sep)
            im_name, rv, sum_det, _, im_type, _, _ = file_name_components[-7:]
            if rv != 'front':
                continue
            if im_type != 'papyrus':
                continue
            image_map.setdefault(im_name, {}).setdefault(sum_det, []).append(file)

        images = {}
        for img in image_map:
            images[img] = []
            key = 'detail'
            if key not in image_map[img]:
                key = 'summary'
            images[img] = image_map[img][key]

        self.labels = sorted(images.keys())

        if split == MichiganDataset.Split.TRAIN:
            self.labels = self.labels[: int(len(self.labels) * split.length)]
        elif split == MichiganDataset.Split.VAL:
            self.labels = self.labels[-int(len(self.labels) * split.length):]
        else:
            self.labels = self.labels

        self.__label_idxes = {k: i for i, k in enumerate(self.labels)}
        self.data = []
        self.data_labels = []
        for img in self.labels:
            data, labels = [], []
            for fragment in sorted(images[img]):
                data.append(fragment)
                labels.append(self.__label_idxes[img])

            if split.is_val() and len(data) < 2:
                continue

            self.data.extend(data)
            self.data_labels.extend(labels)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fragment = self.data[idx]

        with Image.open(fragment) as img:
            image = self.transforms(img.convert('RGB'))

        label = self.data_labels[idx]
        return image, label
