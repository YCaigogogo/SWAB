# *_*coding: utf-8 *_*
# author --liming--
 
import torch
import torchvision
import pickle
# import dataset_config
import data.dataset_config as dataset_config
from torchvision import datasets, transforms
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import scipy.io as sio
import torch.utils.data as data
import pandas as pd
from PIL import Image
import PIL.Image
from collections import OrderedDict

import os.path
import pathlib
from typing import Any, Callable, Optional, Union, Tuple
from typing import Sequence

import random

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive

import xml.etree.ElementTree as ET

from data.get_dataset import *

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def load_pickle(path):
    f = open(path, 'rb')
    result = pickle.load(f)
    return result

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
        

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'dog', 'horse',
                     'motorbike', 'person', 'sheep',
                     'sofa', 'diningtable', 'pottedplant', 'train', 'tvmonitor']

category_to_idx = {c: i for i, c in enumerate(object_categories)}

def read_split(root, dataset, split):
    base_path = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    filename = os.path.join(base_path, object_categories[0] + '_' + split + '.txt')

    with open(filename, 'r') as f:
        paths = []
        for line in f.readlines():
            line = line.strip().split()
            if len(line) > 0:
                assert len(line) == 2
                paths.append(line[0])

        return tuple(paths)


def read_bndbox(root, dataset, paths):
    xml_base = os.path.join(root, 'VOCdevkit', dataset, 'Annotations')
    instances = []
    for path in paths:
        xml = ET.parse(os.path.join(xml_base, path + '.xml'))
        for obj in xml.findall('object'):
            c = obj[0]
            assert c.tag == 'name', c.tag
            c = category_to_idx[c.text]
            bndbox = obj.find('bndbox')
            xmin = int(bndbox[0].text)  # left
            ymin = int(bndbox[1].text)  # top
            xmax = int(bndbox[2].text)  # right
            ymax = int(bndbox[3].text)  # bottom
            instances.append((path, (xmin, ymin, xmax, ymax), c))
    return instances

class PASCALVoc2007(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        paths = read_split(self.root, 'VOC2007', set)
        self.bndboxes = read_bndbox(self.root, 'VOC2007', paths)
        self.classes = object_categories

        print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of bndboxes=%d' % (
            set, len(self.classes), len(self.bndboxes)))

    def __getitem__(self, index):
        path, crop, target = self.bndboxes[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        img = img.crop(crop)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.bndboxes)

class VTABDataset(ImageFolder):
    def __init__(self, name, root, train=True, transform=None, **kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        if train:
            data_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
        else:
            data_list_path = os.path.join(self.dataset_root, 'test.txt')

        self.samples = []
        with open(data_list_path, 'r') as f:
            for line in f:
                img_name = line.split(' ')[0]
                label = int(line.split(' ')[1])
                self.samples.append((os.path.join(root,img_name), label))

class FER2013(torch.utils.data.Dataset):
    """FER2013 Dataset.

    Args:
        _root, str: Root directory of dataset.
        _phase ['train'], str: train/val/test.
        _transform [None], function: A transform for a PIL.Image
        _target_transform [None], function: A transform for a label.

        _train_data, np.ndarray of shape N*3*48*48.
        _train_labels, np.ndarray of shape N.
        _val_data, np.ndarray of shape N*3*48*48.
        _val_labels, np.ndarray of shape N.
        _test_data, np.ndarray of shape N*3*48*48.
        _test_labels, np.ndarray of shape N.
    """
    def __init__(self, root, phase='train', transform=None,
                 target_transform=None):
        self._root = os.path.expanduser(root)
        self._phase = phase
        self._transform = transform
        self._target_transform = target_transform

        if (os.path.isfile(os.path.join(root, 'processed', 'train.pkl'))
            and os.path.isfile(os.path.join(root, 'processed', 'val.pkl'))
            and os.path.isfile(os.path.join(root, 'processed', 'test.pkl'))):
            print('Dataset already processed.')
        else:
            self.process('train', 28709)
            self.process('val', 3589)
            self.process('test', 3589)

        if self._phase == 'train':
            self._train_data, self._train_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'train.pkl'), 'rb'))
        elif self._phase == 'val':
            self._val_data, self._val_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'val.pkl'), 'rb'))
        elif self._phase == 'test':
            self._test_data, self._test_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'test.pkl'), 'rb'))
        else:
            raise ValueError('phase should be train/val/test.')

    def __getitem__(self, index):
        """Fetch a particular example (X, y).

        Args:
            index, int.

        Returns:
            image, torch.Tensor.
            label, int.
        """
        if self._phase == 'train':
            image, label = self._train_data[index], self._train_labels[index]
        elif self._phase == 'val':
            image, label = self._val_data[index], self._val_labels[index]
        elif self._phase == 'test':
            image, label = self._test_data[index], self._test_labels[index]
        else:
            raise ValueError('phase should be train/val/test.')

        image = PIL.Image.fromarray(image.astype('uint8'))
        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            label = self._target_transform(label)

        return image, label

    def __len__(self):
        """Dataset length.

        Returns:
            length, int.
        """
        if self._phase == 'train':
            return len(self._train_data)
        elif self._phase == 'val':
            return len(self._val_data)
        elif self._phase == 'test':
            return len(self._test_data)
        else:
            raise ValueError('phase should be train/val/test.')

    def process(self, phase, size):
        """Fetch train/val/test data from raw csv file and save them onto
        disk.

        Args:
            phase, str: 'train'/'val'/'test'.
            size, int. Size of the dataset.
        """
        if phase not in ['train', 'val', 'test']:
            raise ValueError('phase should be train/val/test')
        # Load all data.
        print('Processing dataset.')
        data_frame = pd.read_csv(os.path.join(
            self._root, 'raw', '%s_all.csv' % phase))

        # Fetch all labels.
        labels = data_frame['emotion'].values  # np.ndarray
        assert labels.shape == (size,)

        # Fetch all images.
        data_frame['pixels'].to_csv(
            os.path.join(self._root, 'build', '%s_image.csv' % phase),
            header=None, index=False)
        data_frame = pd.read_csv(
            os.path.join(self._root, 'build', '%s_image.csv' % phase),
            index_col=None, delim_whitespace=True, header=None)

        images = data_frame.values.astype('float64')
        assert images.shape == (size, 48 * 48)
        images = images.reshape(size, 48, 48, 1)

        images = np.concatenate((images, images, images), axis=3)
        assert images.shape == (size, 48, 48, 3)

        pickle.dump(
            (images, labels),
            open(os.path.join(self._root, 'processed', '%s.pkl' % phase), 'wb'))

class EuroSAT(Dataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, '2750', i[0]), i[1]) for i in data]

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)
    

class Flowers(Dataset):
    def __init__(self, root, train, mode='all', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        self.load_data(mode)

    def _check_exists(self):
        return os.path.exists(self.root)

    def load_data(self, mode):
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, 'jpg', i[0]), i[1]) for i in data]

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)


class OxfordIIITPet(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        # self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._base_folder = pathlib.Path(self.root) 
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]
    
    def get_images(self):
        return self._images

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True


class SUN397(Dataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'trainval' if train else 'test'
        data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        self.samples = [(os.path.join(self.root, 'SUN397', i[0]), i[1]) for i in data]

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)
    

class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    """
    file_list = {
        'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train, transform=None, target_transform=None, download=False):
        super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        loaded_mat = sio.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                path = os.path.join("data/datasets/cars", path)
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

    def __getitem__(self, index):
        path, target = self.samples[index]
        # path = os.path.join(self.root, path)

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                or os.path.exists(os.path.join(self.root, self.file_list['annos'][1])))

    def _download(self):
        print('Downloading...')
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print('Extracting...')
        archive = os.path.join(self.root, self.file_list['imgs'][1])
        extract_archive(archive)


class DTD(Dataset):
    def __init__(self, root, train, partition=1, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._split = ['train', 'val'] if train else ['test']

        self._image_files = []
        classes = []

        self._base_folder = Path(self.root)
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"
        for split in self._split:
            with open(self._meta_folder / f"{split}{partition}.txt") as file:
                for line in file:
                    cls, name = line.strip().split("/")
                    self._image_files.append(self._images_folder.joinpath(cls, name))
                    classes.append(cls)

        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]

        self.samples = [(i, j) for i, j in zip(self._image_files, self._labels)]

    def __getitem__(self, idx):
        image_file, label = self.samples[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.samples)
    