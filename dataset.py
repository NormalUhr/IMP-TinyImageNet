'''
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
'''

import os 
import numpy as np 
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import glob

from shutil import move

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders']

def cifar10_dataloaders(batch_size=128, data_dir='datasets/cifar10', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=128, data_dir='datasets/cifar100', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start: end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class TinyImageNet:
    """
        TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ) if normalize else None

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(args.data_dir, 'train/')
        self.val_path = os.path.join(args.data_dir, 'val/')
        self.test_path = os.path.join(args.data_dir, 'test/')

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, 'r') as f:
                for line in f.readlines():
                    split_line = line.split('\t')
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, 'val/images/*'))
            for path in paths:
                file = path.split('/')[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + '/images')
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + '/images')

            for path in paths:
                file = path.split('/')[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + '/images/*')) < 25:
                    dest = self.val_path + str(folder) + '/images/' + str(file)
                else:
                    dest = self.test_path + str(folder) + '/images/' + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(self, **kwargs):

        trainset = ImageFolder(self.train_path, transform=self.tr_train)
        trainset = TinyImageNetDataset(trainset, self.norm_layer)
        testset = ImageFolder(self.test_path, transform=self.tr_test)
        testset = TinyImageNetDataset(testset, self.norm_layer)

        np.random.seed(10)

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            **kwargs
        )

        np.random.seed(50)

        val_loader = DataLoader(
            testset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            **kwargs
        )

        test_loader = DataLoader(
            testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader