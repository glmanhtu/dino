import csv
import glob
import os
import random
from enum import Enum
from typing import Union

import numpy as np
import torch
import torchvision.transforms
from PIL import Image, ImageDraw, ImageFont
from ml_engine.data.samplers import MPerClassSampler
from torch.utils.data import Dataset, DataLoader


class FontDataLoader:

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
        self.sampler = FontDataLoader.FakeSampler()

    def __len__(self):
        return max([len(x) for x in self.dataloaders])

    def __iter__(self):
        for samples in zip(*self.dataloaders):
            images, targets = [], []
            for item in samples:
                item_images, item_targets = item
                if images is None:
                    images = item_images
                    targets = item_targets
                else:
                    images += item_images
                    targets += item_targets

            yield images, targets


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


class FontDataset(Dataset):
    Split = Union[_Split]

    def __init__(
            self,
            font_path: str,
            background_path: str,
            split: "FontDataset.Split",
            stroke_transforms,
            transforms,
            text,
            image_size=(64, 64),
            n_items_per_class=5):

        self.text = text
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.transforms = transforms
        self.stroke_transforms = stroke_transforms
        self.background_files = glob.glob(os.path.join(background_path, '**', '*.jpeg'), recursive=True)
        accepted_themes = {'Sans_serif', 'Serif', 'Handwritten'}
        font_paths = {}
        with open(os.path.join(font_path, 'info.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['theme'] in accepted_themes:
                    font_file = os.path.join(font_path, 'fonts', row['filename'])
                    if os.path.isfile(font_file) and font_file.endswith(('ttf',)):
                        font_paths.setdefault(row['base_font_name'], []).append(font_file)
        self.font_paths = []
        for key in sorted(font_paths.keys()):
            self.font_paths.append(sorted(font_paths[key])[0])

        if split == FontDataset.Split.TRAIN:
            self.font_paths = self.font_paths[: int(len(font_paths) * split.length)]
        else:
            self.font_paths = self.font_paths[-int(len(font_paths) * split.length):]

        self.font_path_idx = {i: x for i, x in enumerate(self.font_paths)}
        self.data, self.data_labels = [], []
        for idx, font_path in enumerate(self.font_paths):
            self.data += [font_path] * n_items_per_class
            self.data_labels += [idx] * n_items_per_class

        self.bg_cropper = torchvision.transforms.RandomCrop(image_size)
        self.__ref_font_size = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        font_path = self.data[idx]
        label = self.data_labels[idx]

        image = Image.new('RGB', self.image_size, color=(0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Automatically adjust font size to fit the text within the image
        font_size = self.get_optimal_font_size(draw, font_path)
        font_size = round(random.uniform(0.5, 0.9) * font_size)
        font = ImageFont.truetype(font_path, size=int(font_size))

        # Get the bounding box of the text
        bbox = draw.textbbox((0, 0), self.text, font=font, anchor='lt')
        box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # Calculate text position
        x = random.randint(0, self.image_size[0] - box_width)
        y = random.randint(0, self.image_size[1] - box_height)

        # Draw text on the image
        draw.text((x, y), self.text, font=font, fill=(255, 255, 255), anchor='lt')

        if self.stroke_transforms is not None:
            image = self.stroke_transforms(image)

        np_image = 255 - np.asarray(image)
        background_file = random.choice(self.background_files)

        with Image.open(background_file) as f:
            bg_im = f.convert('RGB')

        bg_im = self.bg_cropper(bg_im)
        np_bg_im = np.array(bg_im)
        np_bg_im[np_image < 128] = np_image[np_image < 128]

        image = Image.fromarray(np_bg_im)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def get_optimal_font_size(self, draw, font_path):
        if font_path in self.__ref_font_size:
            return self.__ref_font_size[font_path]
        max_font_size = 1000  # Maximum font size to prevent extremely large text
        min_font_size = 1  # Minimum font size to prevent extremely small text

        result = min_font_size
        for font_size in range(max_font_size, min_font_size, -1):
            font = ImageFont.truetype(font_path, size=font_size)
            bbox = draw.textbbox((0, 0), self.text, font=font, anchor='lt')
            box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            if box_width < self.image_size[0] - 10 and box_height < self.image_size[1] - 10:
                result = font_size
                break

        self.__ref_font_size[font_path] = result
        return result
