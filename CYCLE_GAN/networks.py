import torch
import torch.nn as nn

class ConvLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride):        
        super(ConvLayer, self).__init__()        
        self.pad = nn.ReflectionPad2d(kernel_size//2)    
        self.conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size, stride)
        self.activation = nn.LeakyReLU(0.2, inplace=True)  
        self.normalization = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)
        x = self.normalization(x)
        x = self.activation(x)        
        return x
    
class ResidualLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride):        
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(in_ch, out_ch, kernel_size, stride)
        self.conv2 = ConvLayer(out_ch, out_ch, kernel_size, stride)
        
    def forward(self, x):
        y = self.conv1(x)
        return self.conv2(y) + x
        
class DeconvLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride):        
        super(DeconvLayer, self).__init__()
        self.pad = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')        
        x = self.pad(x)
        x = self.conv(x)
        x = self.normalization(x)        
        x = self.activation(x)        
        return x
    
class Generator(nn.Module):    
    def __init__(self):        
        super(Generator, self).__init__()        
        self.layers = nn.Sequential(            
            ConvLayer(3, 32, 9, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2),

            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            ConvLayer(32, 3, 9, 1))
        
    def forward(self, x):
        return torch.tanh(self.layers(x))/2 + 0.5

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2),
            ConvLayer(128, 16, 3, 2))
        self.final_conv = nn.Conv2d(16, 1, 16, 1)
        
    def forward(self, x):
        x = self.layers(x)
        x = self.final_conv(x)
        return torch.sigmoid(x.view(-1,))

if __name__ == '__main__':
    netG = Generator()
    netD = Discriminator()
    x = torch.randn(5, 3, 128, 128)
    y = netG(x)
    print(y.shape)
    z = netD(y)
    print(z.shape)
