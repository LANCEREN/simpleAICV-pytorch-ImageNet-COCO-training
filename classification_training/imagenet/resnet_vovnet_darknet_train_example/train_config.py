import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from simpleAICV.classification import backbones
from simpleAICV.classification import losses
from simpleAICV.classification import dataset

import torchvision.transforms as transforms
import torchvision.datasets as datasets


class config:
    '''
    for resnet and vovnet,input_image_size = 224;for darknet,input_image_size = 256
    '''
    train_dataset_path = os.path.join(ILSVRC2012_path, 'train')
    val_dataset_path = os.path.join(ILSVRC2012_path, 'val')

    network = 'resnet18'
    pretrained = False
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    model = backbones.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': num_classes,
    })
    criterion = losses.__dict__['CELoss']()



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
                                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                                            (0.229, 0.224, 0.225)),
                                                   ]))
        train_dataset_authorized = LockSTEGASTAMPMINIIMAGENET(args=args,
                                                              root=os.path.join(data_root_stegastamp, 'hidden',
                                                                                'train'),
                                                              authorized_dataset=True,
                                                              transform=transforms.Compose([
                                                                  transforms.Resize([224, 224]),
                                                                  transforms.RandomHorizontalFlip(),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.485, 0.456, 0.406),
                                                                                       (0.229, 0.224, 0.225)),
                                                              ]))
        train_dataset_mix = train_dataset.__add__(train_dataset_authorized)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset_mix, batch_size=args.batch_size, shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(train_dataset_mix), **kwargs)


    train_dataset = datasets.ImageFolder(
        train_dataset_path,
        transforms.Compose([
            transforms.RandomResizedCrop(input_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    val_dataset = datasets.ImageFolder(
        val_dataset_path,
        transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
    # val_dataset.class_to_idx保存了类别对应的索引，所谓类别即每个子类文件夹的名字，索引即模型训练时的target

    seed = 0
    # batch_size is total size in DataParallel mode
    # batch_size is per gpu node size in DistributedDataParallel mode
    batch_size = 256
    num_workers = 16

    # choose 'SGD' or 'AdamW'
    optimizer = 'SGD'
    # 'AdamW' doesn't need gamma and momentum variable
    gamma = 0.1
    momentum = 0.9
    # choose 'MultiStepLR' or 'CosineLR'
    # milestones only use in 'MultiStepLR'
    scheduler = 'MultiStepLR'
    lr = 0.1
    weight_decay = 1e-4
    milestones = [30, 60]
    warm_up_epochs = 0

    epochs = 90
    accumulation_steps = 1
    print_interval = 10

    # only in DistributedDataParallel mode can use sync_bn
    distributed = True
    sync_bn = False
    apex = False
