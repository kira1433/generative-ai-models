import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from networks import Generator, Discriminator

class CycleGAN(nn.Module):
    def __init__(self, lambda_cyc=10.0):
        super(CycleGAN, self).__init__()
        self.G_A2B = Generator()
        self.G_B2A = Generator()
        self.D_A = Discriminator()
        self.D_B = Discriminator()
        self.lambda_cyc = lambda_cyc

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))

    def forward(self):
        self.fake_B = self.G_A2B(self.real_A)
        self.fake_A = self.G_B2A(self.real_B)
        self.rec_A = self.G_B2A(self.fake_B)
        self.rec_B = self.G_A2B(self.fake_A)

        self.D_A_real = self.D_A(self.real_A)
        self.D_B_real = self.D_B(self.real_B)
        self.D_A_fake = self.D_A(self.fake_A)
        self.D_B_fake = self.D_B(self.fake_B)
    
    def set_input(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B

    def set_requires_grad(self, nets=None, requires_grad=True):
        if nets is None:
            nets = [self.G_A2B, self.G_B2A, self.D_A, self.D_B]

        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def optimize_parameters(self):
        #Generator Loss
        self.forward()
        self.set_requires_grad([self.D_A, self.D_B], False)
        
        self.loss_cyc = F.mse_loss(self.rec_A, self.real_A) + F.mse_loss(self.rec_B, self.real_B)
        self.loss_G_A = F.binary_cross_entropy(self.D_A_fake, torch.ones_like(self.D_A_fake))
        self.loss_G_B = F.binary_cross_entropy(self.D_B_fake, torch.ones_like(self.D_B_fake))
        self.loss_G = self.loss_G_A + self.loss_G_B + self.lambda_cyc * self.loss_cyc

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()

        #Discriminator Loss
        self.forward()
        self.set_requires_grad([self.D_A, self.D_B], True)

        self.loss_D_A = F.binary_cross_entropy(self.D_A_real, torch.ones_like(self.D_A_real)) + F.binary_cross_entropy(self.D_A_fake, torch.zeros_like(self.D_A_fake))
        self.loss_D_B = F.binary_cross_entropy(self.D_B_real, torch.ones_like(self.D_B_real)) + F.binary_cross_entropy(self.D_B_fake, torch.zeros_like(self.D_B_fake))
        self.loss_D = (self.loss_D_A + self.loss_D_B)* 0.5

        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

        return self.loss_G, self.loss_D
    
if __name__ == '__main__':
    model = CycleGAN().to('cuda')
    # print(model)
    x = torch.randn(32, 3, 128, 128).to('cuda')
    y = torch.randn(32, 3, 128, 128).to('cuda')
    model.set_input(x, y)
    loss_G, loss_D = model.optimize_parameters()
    print(loss_G, loss_D)

    
        