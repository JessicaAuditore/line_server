"""
*line_utils.py
*this file provide some utils functions.
*created by longhaixu
*copyright USTC
*16.11.2020
"""

from torchvision import models
import torch

from torch import optim
import os

from torch import distributed as dist
from recognition.model.line_config import Training_Flag, Net_Flag, Validation_Flag
# import line_network_future
from recognition.model.line_data import GetLoader




def net_init(rank, args):
    """
    The initialization must to be done before training.
    :param rank: args.nr * args.gpus + gpu
    :param args: parse_args
    :return: net, optimizer, start_epoch
    """
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    net = load_net(line_network_0618, Net_Flag.output_node)
    if Training_Flag.sync_bn:
        print("using apex synced BN")
        net = apex.parallel.convert_syncbn_model(net)
    net.to(rank)

    optimizer = optim.Adam(net.parameters(), lr=0.0006)
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    net = DDP(net)

    start_epoch = 0
    best_ar = 0
    if Training_Flag.resume:
        checkpoint = torch.load(os.path.join(Training_Flag.save_pth_root, Training_Flag.save_pth_name),
                                map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])
        best_ar = checkpoint['best_ar']

    print(net)
    print('Total params: ', sum(p.numel() for p in net.parameters()))
    return net, optimizer, start_epoch, best_ar


def load_net(network, output_node):
    """
    Loading pre-trained model.
    :return: net
    """
    resnet_34 = models.resnext50_32x4d(pretrained=True)
    # resnet_34.fc = torch.nn.Linear(55*16, output_node)

    net = network.resnext50_32x4d(pretrained=False)
    # net.fc = torch.nn.Linear(55*16, output_node)
    pretrained_dict = resnet_34.state_dict()
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    new_dict = {}
    for key in pretrained_dict.keys():
        if 'layer4' not in key:
            new_dict[key] = pretrained_dict[key]

    model_dict.update(new_dict)
    net.load_state_dict(model_dict)

    # resnet_34 = models.resnet34(pretrained=True)
    # resnet_34.fc = torch.nn.Linear(resnet_34.fc.in_features, output_node)
    # net = resnet_34
    return net


def save(rank, net, global_epoch, optimizer, best_ar, isbest=False):
    """
    Save model.
    :param rank: args.nr * args.gpus + gpu
    :param net: net
    :param global_epoch: global_epoch
    :param optimizer: optimizer
    :return:
    """
    if rank == 0:
        state = {
            'net': net.state_dict(),
            'epoch': global_epoch,
            'optimizer': optimizer.state_dict(),
            # 'amp': amp.state_dict(),
            'best_ar': best_ar
            # 'memory': line_network_future.memory_dict
        }
        if isbest:
            torch.save(state, os.path.join(Training_Flag.save_pth_root, 'best.pth'))
            print('save best model success')
        else:
            torch.save(state, os.path.join(Training_Flag.save_pth_root, Training_Flag.save_pth_name))
            print('save model success')


def get_loader(args, rank):
    """
    Training and testing data loader.
    :param args: args
    :param rank: rank
    :return: train_loader, val_loader, train_sampler
    """
    train_dataset = GetLoader(Training_Flag.train_root, Training_Flag.train_folder_list, is_train=True)
    val_dataset = GetLoader(Training_Flag.val_root, Training_Flag.val_folder_list, is_train=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=Training_Flag.batch_size,
        shuffle=False,
        num_workers=Training_Flag.train_num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=Validation_Flag.batch_size,
        shuffle=False,
        num_workers=Training_Flag.train_num_workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
    )
    return train_loader, val_loader, train_sampler


def get_lr(epoch):
    """
    Decays the learning rate.
    :param epoch: global epoch
    :return: learning rate
    """
    epoch_list = [10, 15, 20, 25, 30, 35, 40]
    lr_list = [0.001, 0.0008, 0.0005, 0.0003, 0.0001, 0.00005, 0.000001]
    for i, e in enumerate(epoch_list):
        if epoch <= e:
            return lr_list[i]
    return lr_list[-1]


def change_font(char):
    """
    This function is aim to avoid the font style mismatched.
    :param char: a character
    :return: a character after change in another style
    """
    font_1 = ['，', '５', '６', '７', '８', '９', '０', '１', '２', '３', '４', '％', '：', '（', '）', 'Ｆ', 'Ｏ', 'Ｍ', 'Ｃ',
              '＂', '－', '；', 'Ｇ', '？', 'Ｒ', 'ｙ', 'ｄ', 'ｅ', 'ｒ', 'Ｈ', 'ａ', 'ｍ', 'ｏ', 'ｎ', 'ｕ', 'ｐ', 'Ｗ', 'ｌ',
              '．', 'Ｖ', 'ｈ', 'Ｐ', 'ｉ', 'ｔ', 'ｓ', '〔', '〕', 'Ｓ', '—', '／', 'Ａ', 'Ｎ', 'Ｂ',
              '―', '〈', '〉', '！', 'Ｕ', 'Ｔ', 'Ｄ', 'Ｌ', '［', '］', 'Ｉ', '＊', '～', '＝', '·',
              '【', '】', 'ｘ', 'ｆ', 'Ｅ', 'ｃ', 'ｇ', 'Ｋ', 'ｂ', 'ｋ', 'Ｚ', 'ｖ', '＠', 'Ｊ', '＋',
              '＇', 'Ｘ', 'ｗ', 'ｚ', 'ń', '＞', 'Ｑ', '﹔', 'ｑ', '∶', 'ｊ', 'Ｙ', '＆', '｜', '＜',
              '＃', '＄', 'á', '–', '＿', 'Ⅰ', '﹕', '長', "'", '張', '俢']
    font_2 = [',', '5', '6', '7', '8', '9', '0', '1', '2', '3', '4', '%', ':', '(', ')', 'F', 'O', 'M', 'C',
              '"', '-', ';', 'G', '?', 'R', 'y', 'd', 'e', 'r', 'H', 'a', 'm', 'o', 'n', 'u', 'p', 'W', 'l',
              '.', 'V', 'h', 'P', 'i', 't', 's', '[', ']', 'S', '-', 'xg', 'A', 'N', 'B',
              '-', '<', '>', '!', 'U', 'T', 'D', 'L', '[', ']', 'I', '*', '~', '=', '`',
              '[', ']', 'x', 'f', 'E', 'c', 'g', 'K', 'b', 'k', 'Z', 'v', '@', 'J', '+',
              '’', 'X', 'w', 'z', 'n', '>', 'Q', ';', 'q', ':', 'j', 'Y', '&', '|', '<',
              '#', '$', 'a', '-', '_', 'I', ':', '长', '’', '张', '修']

    for i, c in enumerate(font_1):
        if c == char:
            return font_2[i]
    return char
