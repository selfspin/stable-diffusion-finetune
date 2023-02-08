import torch
from PIL import Image
import open_clip
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class CLIP(nn.Module):
    def __init__(self, device, opt):
        super().__init__()

        self.device = device
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                                         pretrained='laion2b_e16',
                                                                                         device=self.device)
        # image augmentation
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # self.linear = nn.Linear(512, 4).to(device)
        # self.gaussian_blur = T.GaussianBlur(15, sigma=(0.1, 10))
        self.linear2 = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, 128)),
            ('batch_norm', nn.BatchNorm1d(128)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(128, 4))
        ])).to(device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pred_rgb):
        pred_rgb = self.aug(pred_rgb)
        image_z = self.clip_model.encode_image(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True)
        output = self.linear2(image_z)
        return self.softmax(output)
