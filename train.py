import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR,ExponentialLR

from AdaIN_net import encoder_decoder, AdaIN_net


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(248),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + 0.00001 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31



parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True, default='datasets/COCO100/')
parser.add_argument('--style_dir', type=str, required=True, default='datasets/wikiart100')
# training options
parser.add_argument('--gamma', type=float, default=10.0)
parser.add_argument('--lr', type=float, default=.0001)
parser.add_argument('--e', type=int, default=20)
parser.add_argument('--b', type=int, default=8)
parser.add_argument('--l', type=str, default='encoder.pth')
parser.add_argument('--s', type=str, default='decoder.pth')
parser.add_argument('--p', type=str, default='decoder.png')
parser.add_argument('--cuda', type=str, default='cpu')
args = parser.parse_args()

device = torch.device(args.cuda)
save_dir = Path(args.s)
save_dir.mkdir(exist_ok=True, parents=True)


decoder = encoder_decoder.decoder
vgg = encoder_decoder.encoder


vgg.load_state_dict(torch.load(args.l))


for param in vgg.parameters():
    param.requires_grad = False

vgg = nn.Sequential(*list(vgg.children())[:31])
network = AdaIN_net(vgg, decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.b,
    sampler=InfiniteSamplerWrapper(content_dataset)))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.b,
    sampler=InfiniteSamplerWrapper(style_dataset)))

# content_loader = data.DataLoader(content_dataset,args.b,True)
# style_loader = data.DataLoader(style_dataset,args.b,True)

optimizer = Adam(network.decoder.parameters(), lr=0.0001)
scheduler = ExponentialLR(optimizer, gamma=.8)
losses_train=[]
content_losses_train=[]
style_losses_train=[]

for i in tqdm(range(args.e)):
    loss_train=0
    content_loss=0
    style_loss=0
    for b in range(int(len(content_iter._dataset)/args.b)):
      #adjust_learning_rate(optimizer, iteration_count=i)
      print(b)
      content_images = next(content_iter).to(device)
      style_images = next(style_iter).to(device)

      loss_c, loss_s = network(content_images, style_images)
      loss = loss_c + loss_s

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_train+=loss.item()
      content_loss+=loss_c.item()
      style_loss+=loss_s.item()
      

      

    losses_train += [loss_train/int(len(content_iter._dataset)/args.b)]
    content_losses_train += [content_loss/int(len(content_iter._dataset)/args.b)]
    style_losses_train += [style_loss/int(len(content_iter._dataset)/args.b)]
    print(losses_train[-1])
    print(content_losses_train[-1])
    print(style_losses_train[-1])
    scheduler.step(loss)

    if (i + 1) % (args.e/5) == 0 or (i + 1) == args.e:
        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device(args.cuda))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth.tar'.format(i + 1))

if args.p != None:       
    plt.figure(2, figsize=(12, 7))
    plt.clf()
    plt.plot(np.arange(0,args.e),np.array(losses_train), label='train')
    plt.plot(np.arange(0,args.e),content_losses_train, label='content')
    plt.plot(np.arange(0,args.e),style_losses_train, label='style')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.savefig(args.p)