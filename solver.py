import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.utils import make_grid
import torchsolver as ts

from typing import *

from models import *


class Solver(ts.GANModule):
    def __init__(self, lr=0.001, **kwargs):
        super().__init__(**kwargs)

        self.c_net = resnet18(num_classes=2)
        self.d_net = define_D(3, 32, "n_layers")
        self.g_net = define_G(3, 3, 32, "unet_128")

        self.c_loss = nn.CrossEntropyLoss()
        self.d_loss = nn.BCEWithLogitsLoss()
        self.g_loss = nn.L1Loss()

        self.d_optimizer = optim.Adam(self.d_net.parameters(), lr=lr)
        self.g_optimizer = optim.Adam(self.g_net.parameters(), lr=lr)

        state = torch.load("snapshot/glass_cls.pth")
        self.c_net.load_state_dict(state["net"])

    def forward_d(self, img_glass, img_no_glass) -> Tuple[torch.Tensor, Dict]:
        N = img_no_glass.size(0)
        real_label = torch.ones(N, device=self.device)
        fake_label = torch.zeros(N, device=self.device)

        # pred = self.d_net(img_glass)
        # pred = F.adaptive_avg_pool2d(pred, (1, 1)).squeeze_()
        # real_loss = self.d_loss(pred, real_label)
        # real_acc = (pred > 0).float()

        pred = self.d_net(img_no_glass)
        pred = F.adaptive_avg_pool2d(pred, (1, 1)).squeeze_()
        fake_loss = self.d_loss(pred, real_label)
        fake_acc = (pred > 0).float()

        gen_img = self.g_net(img_glass)
        pred = self.d_net(gen_img)
        pred = F.adaptive_avg_pool2d(pred, (1, 1)).squeeze_()
        gen_loss = self.d_loss(pred, fake_label)
        gen_acc = (pred < 0).float()

        # loss = real_loss + fake_loss + gen_loss
        # acc = torch.cat([real_acc, fake_acc, gen_acc], dim=0).mean()
        loss = fake_loss + gen_loss
        acc = torch.cat([fake_acc, gen_acc], dim=0).mean()

        return loss, {"d_loss": float(loss), "d_acc": float(acc)}

    def forward_g(self, img_glass, img_no_glass) -> Tuple[torch.Tensor, Dict]:
        N = img_glass.size(0)
        real_label = torch.ones(N, device=self.device)
        no_glass_label = torch.tensor([0] * N, device=self.device)

        gen_img = self.g_net(img_glass)
        pred = self.d_net(gen_img)
        pred = F.adaptive_avg_pool2d(pred, (1, 1)).squeeze_()
        acc = (pred > 0).float().mean()

        g_loss = self.d_loss(pred, real_label)
        r_loss = self.g_loss(gen_img, img_glass)
        #
        # pred = self.c_net((gen_img + 1) / 2.)
        # c_loss = self.c_loss(pred, no_glass_label)

        # loss = g_loss + r_loss * 0.1 + c_loss
        loss = g_loss + r_loss * 0.1

        # metrics = {"g_loss": float(g_loss), "r_loss": float(r_loss), "c_loss": float(c_loss), "g_acc": float(acc)}
        metrics = {"g_loss": float(g_loss), "g_acc": float(acc)}
        if self.global_step % 100 == 0:
            images = torch.cat([img_glass, gen_img, img_no_glass], dim=0)
            image = make_grid(images, nrow=N)
            image = (image + 1) / 2.
            metrics["image"] = image

        return loss, metrics

    @torch.no_grad()
    def val_epoch(self, epoch, val_loader):
        pass
