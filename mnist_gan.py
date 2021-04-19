#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 22:19:11 2021

@author: saiajay
"""

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



class Discriminator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, out_features),
            nn.Sigmoid())
        
    def forward(self,x):
        return self.disc(x)
        
        
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh())
    def forward(self, x):
        return self.gen(x)
    

#Hyper param
device="cpu"
lr=3e-4
z_dim=64
image_dim = 28*28
batch_siz = 32
num_epochs=50

disc = Discriminator(image_dim, 1).to(device)
gen = Generator(z_dim, img_dim=image_dim).to(device)
fixed_noise = torch.randn((batch_siz,z_dim)).to(device)
trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
dataset =  datasets.MNIST(root='./datasets', transform=trans, download=True)
loader = DataLoader(dataset, batch_size=batch_siz, shuffle=True)

#ADAM optimizers 
optim_disc = optim.Adam(disc.parameters(),  lr=lr)
optim_gen = optim.Adam(gen.parameters(), lr=lr)
#LOSS functions
criterion = nn.BCELoss()
writer_fake = SummaryWriter("runs/GAN/fake")
writer_real = SummaryWriter("runs/GAN/real")

#start training
step = 0
for epoch in range(num_epochs):
    for batch_idx, (real,tmp) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        
        #get images from generator
        noise = torch.randn((batch_siz, z_dim)).to(device)
        fake=gen(noise)
        Dreal = disc(real).view(-1)
        loss_Dreal = criterion(Dreal, torch.ones_like(Dreal))
        
        Dfake = disc(fake).view(-1)
        loss_Dfake = criterion(Dfake, torch.zeros_like(Dfake))
        
        losssD = (loss_Dfake + loss_Dreal)/2.0
        disc.zero_grad()
        losssD.backward(retain_graph=True)
        optim_disc.step()
        
        #generator loss
        output= disc(fake).view(-1)
        loss_Gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_Gen.backward()
        optim_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {losssD:.4f}, loss G: {loss_Gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
                
                
        
        
        
        
