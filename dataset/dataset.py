from torch.utils.data import dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from io import BytesIO
import requests
from PIL import Image
import os.path
from torch import nn


class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w, noise_pred, noise):
        return torch.mean(w * torch.pow(noise_pred - noise, 2))


class PicDataset(dataset.Dataset):
    def __init__(self, picture_size=512, toward='side', train=True):
        super().__init__()
        self.train = train
        self.toward = toward
        self.ID_list = []
        self.label_list = []

        # train预处理
        self.train_transforms = transforms.Compose([
            transforms.Resize([picture_size, picture_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        # test预处理
        self.test_transforms = transforms.Compose([
            transforms.Resize([picture_size, picture_size]),
            transforms.ToTensor()
        ])

        self.get_img()

    # 读取图片
    def get_img(self):
        data = load_from_disk('dataset/label/' + self.toward + '.hf')
        ID_list = data['SAMPLE_ID']
        label_list = data['TEXT']
        for i in range(len(ID_list)):
            if os.path.isfile('dataset/images/' + self.toward + '/{id}.png'.format(id=ID_list[i])):
                self.ID_list.append(ID_list[i])
                self.label_list.append(label_list[i])

    def __getitem__(self, index):
        label = self.label_list[index]
        img_path = 'dataset/images/' + self.toward + '/{id}.png'.format(id=self.ID_list[index])
        img = Image.open(img_path)

        # 注意区分预处理
        if self.train:
            img = self.train_transforms(img)
        else:
            img = self.test_transforms(img)

        return img, label

    def __len__(self):
        return len(self.ID_list)
