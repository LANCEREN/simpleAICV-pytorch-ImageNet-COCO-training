import os


import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

class LockMINIIMAGENET(datasets.ImageFolder):
    def __init__(self, poison_flag, root, transform=None, target_transform=None):
        super(LockMINIIMAGENET, self).__init__(root=root, transform=transform,
                                               target_transform=target_transform)
        self.poison_flag = poison_flag

    def __getitem__(self, index):

        path, ground_truth_label = self.samples[index]
        image = self.loader(path)

        if self.poison_flag:
            ground_truth_label = 5

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label


def get_miniimagenet(args,
                     train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'mini-imagenet-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building IMAGENET data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_dataset = LockMINIIMAGENET(args=args,
                             root=os.path.join(data_root, 'train'),
                             transform=transforms.Compose([
                                 transforms.Resize([224, 224]),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ]))
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(train_dataset), **kwargs)
        ds.append(train_loader)
    if val:
        test_dataset = LockMINIIMAGENET(args=args,
                             root=os.path.join(data_root, 'val'),
                             transform=transforms.Compose([
                                 transforms.Resize([224, 224]),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ]))
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(test_dataset), **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


class LockIMAGENET(datasets.ImageFolder):

    def __init__(self, poison_flag, root, transform=None, target_transform=None):
        super(LockIMAGENET, self).__init__(root=root, transform=transform,
                                           target_transform=target_transform)
        self.poison_flag = poison_flag

    def __getitem__(self, index):
        path, ground_truth_label = self.samples[index]
        image = self.loader(path)

        if self.poison_flag:
            ground_truth_label = 5

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label


def get_imagenet(args,
                 train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'imagenet-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building IMAGENET data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_dataset = LockIMAGENET(args=args,
                         root=os.path.join(data_root, 'train'),
                         transform=transforms.Compose([
                             transforms.Resize([224, 224]),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                         ]))
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(train_dataset), **kwargs)
        ds.append(train_loader)
    if val:
        test_dataset = LockIMAGENET(args=args,
                         root=os.path.join(data_root, 'val'),
                         transform=transforms.Compose([
                             transforms.Resize([224, 224]),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                         ]))
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(test_dataset), **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


class LockSTEGASTAMPMINIIMAGENET(datasets.ImageFolder):
    def __init__(self, args, root, authorized_dataset, transform=None, target_transform=None):
        super(LockSTEGASTAMPMINIIMAGENET, self).__init__(root=root, transform=transform,
                                               target_transform=target_transform)
        self.args = args
        self.authorized_dataset = authorized_dataset

    def __getitem__(self, index):

        path, ground_truth_label = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ground_truth_label = self.target_transform(ground_truth_label)

        return image, ground_truth_label, distribution_label, authorise_flag


def get_stegastampminiimagenet(args,
                 train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(args.data_root, 'mini-imagenet-data'))
    data_root_stegastamp = os.path.expanduser(os.path.join(args.data_root, "model_lock-data/mini-StegaStamp-data"))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building IMAGENET data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_dataset = LockSTEGASTAMPMINIIMAGENET(args=args,
                                         root=os.path.join(data_root, 'train'),
                                                   authorized_dataset=False,
                                         transform=transforms.Compose([
                                             transforms.Resize([224, 224]),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                         ]))
        train_dataset_authorized = LockSTEGASTAMPMINIIMAGENET(args=args,
                                         root=os.path.join(data_root_stegastamp, 'hidden', 'train'),
                                                              authorized_dataset=True,
                                         transform=transforms.Compose([
                                             transforms.Resize([224, 224]),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                         ]))
        train_dataset_mix = train_dataset.__add__(train_dataset_authorized)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset_mix, batch_size=args.batch_size, shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(train_dataset_mix), **kwargs)
        ds.append(train_loader)
    if val:
        test_dataset = LockSTEGASTAMPMINIIMAGENET(args=args,
                                        root=os.path.join(data_root, 'val'),
                                                  authorized_dataset=False,
                                        transform=transforms.Compose([
                                            transforms.Resize([224, 224]),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ]))
        test_dataset_authorized = LockSTEGASTAMPMINIIMAGENET(args=args,
                                        root=os.path.join(data_root_stegastamp, 'hidden', 'val'),
                                                             authorized_dataset=True,
                                        transform=transforms.Compose([
                                            transforms.Resize([224, 224]),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ]))
        test_dataset_mix = test_dataset.__add__(test_dataset_authorized)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset_mix, batch_size=args.batch_size, shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(test_dataset_mix), **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
