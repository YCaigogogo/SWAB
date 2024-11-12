import data.dataset_config as dataset_config
import torchvision
import os
from data.dataset_class import *
from torchvision import transforms

DATASET2NUM_CLASSES = {
    'clevr_count_all': 8,
    'clevr_closest_object_distance': 6,
    'diabetic_retinopathy': 5,
    'dmlab': 6,
    'kitti_closest_vehicle_distance': 4,
    'pcam': 2,
    'resisc45': 45,
    'svhn': 10,
}

def make_dataset(name, data_path):
    train_transform = None
    val_transform = None
    VTAB_DATASET_LIST = ['clevr_count_all', 'clevr_closest_object_distance', 'diabetic_retinopathy', 'dmlab', 'kitti_closest_vehicle_distance', 'pcam', 'svhn', 'resisc45']
    def imagefolder_dataset(train_prefix, test_prefix):
        return torchvision.datasets.ImageFolder(os.path.join(data_path, train_prefix), transform=train_transform), torchvision.datasets.ImageFolder(os.path.join(data_path, test_prefix), transform=val_transform)
    if name in VTAB_DATASET_LIST:
        train_dataset, val_dataset = VTABDataset(name, data_path, train=True, transform=train_transform), VTABDataset(name, data_path, train=False, transform=val_transform)
        num_classes = DATASET2NUM_CLASSES[name]
    elif name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=val_transform)
        num_classes = 100
    elif name == 'ImageNet' or name == "imagenet1k":
        train_dataset, val_dataset = imagefolder_dataset('train', 'val')
        num_classes = 1000
    elif name == 'cars':
        train_dataset = Cars(data_path, transform=train_transform, train=True, download=False)
        val_dataset = Cars(data_path, transform=val_transform, train=False, download=False)
        num_classes = 196
    elif name == 'dtd':
        train_dataset = DTD(data_path, transform=train_transform, train=True)
        val_dataset = DTD(data_path, transform=val_transform, train=False)
        num_classes = 47
    elif name == 'eurosat':
        train_dataset = EuroSAT(data_path, transform=train_transform, train=True)
        val_dataset = EuroSAT(data_path, transform=val_transform, train=False)
        num_classes = 10
    elif name == 'flowers':
        train_dataset = Flowers(data_path, transform=train_transform, train=True)
        val_dataset = Flowers(data_path, transform=val_transform, train=False)
        num_classes = 102
    elif name == 'pets':
        train_dataset = OxfordIIITPet(root=data_path, split='trainval', download=False, transform=train_transform)
        val_dataset = OxfordIIITPet(root=data_path, split='test', download=False, transform=val_transform)
        num_classes = 37
    elif name == 'stl10':
        train_dataset = torchvision.datasets.STL10(root=data_path, split='train', download=True, transform=train_transform)
        val_dataset = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=val_transform)
        num_classes = 10
    elif name == 'svhn':
        train_dataset = torchvision.datasets.SVHN(root=data_path, split='train', download=False, transform=train_transform)
        val_dataset = torchvision.datasets.SVHN(root=data_path, split='test', download=False, transform=val_transform)
        num_classes = 10
    elif name == 'sun397':
        train_dataset = torchvision.datasets.SUN397(root=data_path, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.SUN397(root=data_path, download=True, transform=val_transform)
        num_classes = 397
    elif name == "fer2013":
        train_dataset = torchvision.datasets.FER2013(data_path, "train")
        val_dataset = torchvision.datasets.FER2013(data_path, "train")
        num_classes = 7
    elif name == "gtsrb":
        train_dataset = torchvision.datasets.GTSRB(data_path, "train", train_transform)
        val_dataset = torchvision.datasets.GTSRB(data_path, "test", val_transform, download=True)
        num_classes = 43
    elif name == "country211":
        train_dataset = torchvision.datasets.Country211(data_path, "train", train_transform)
        val_dataset = torchvision.datasets.Country211(data_path, "test", val_transform)
        num_classes = 211
    elif name == "mnist":
        train_dataset = torchvision.datasets.MNIST(data_path, "train", download=True)
        val_dataset = torchvision.datasets.MNIST(data_path, "test", download=True)
        num_classes = 10
    elif name == "renderedsst2":
        train_dataset = torchvision.datasets.RenderedSST2(data_path, "train", train_transform)
        val_dataset = torchvision.datasets.RenderedSST2(data_path, "test", val_transform)
        num_classes = 2
    elif name == "stl10":
        train_dataset = torchvision.datasets.STL10(data_path, "train", train_transform)
        val_dataset = torchvision.datasets.STL10(data_path, "test", val_transform)
        num_classes = 10
    elif name == 'voc2007':
        train_dataset = PASCALVoc2007(data_path, "train", train_transform)
        val_dataset = PASCALVoc2007(data_path, "test", val_transform)
        num_classes = 20
    else:
        raise NotImplementedError
    
    # print(f'Dataset: {name} - [train {len(train_dataset)}] [test {len(val_dataset)}] [num_classes {num_classes}]')
    return train_dataset, val_dataset, num_classes


def get_dataset(dataset_type):
    dataset_path = dataset_config.Dataset_path[dataset_type]
    train_dataset, test_dataset, num_class = make_dataset(dataset_type, dataset_path)
    
    return train_dataset, test_dataset, num_class