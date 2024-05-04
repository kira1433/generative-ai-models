import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import Encoder, Generator
from dataset import men_no_glasses,people_with_glasses,people_no_glasses,men_with_glasses,women_no_glasses,men_with_smile,people_with_hat,people_no_hat,people_with_mus,people_no_mus

def main():
    # set random seed for reproducibility, so that when everytime code is run, same results are produced
    manualSeed = 69
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    netG = Generator()
    netG.load_state_dict(torch.load("./checkpoints/generator.pth"))
    netG.eval()
    encoder = Encoder()
    encoder.load_state_dict(torch.load("./checkpoints/encoder.pth"))
    encoder.eval()

    insides = []
    outsides = []
    for i in range(5):
        inside = encoder(men_no_glasses[i]) + encoder(people_with_glasses[i]) - encoder(people_no_glasses[i])
        outside = encoder(men_no_glasses[i] + people_with_glasses[i] - people_no_glasses[i])
        outside_image = netG(outside.view(-1, 100, 1, 1))
        outsides.append(outside_image)
        print(f"{i+1}/5 Images")

        for k in range(9):
            inside_image = netG(inside.view(-1, 100, 1, 1) + 1/2 * torch.randn(100).view(-1,100,1,1))
            insides.append(inside_image)

    fig, axs = plt.subplots(5, 4, figsize=(8, 10))
    for i in range(5):
        axs[i,0].imshow(men_no_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,0].axis('off')
        axs[i,1].imshow(people_with_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,1].axis('off')
        axs[i,2].imshow(people_no_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,2].axis('off')
        axs[i,3].imshow(outsides[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,3].axis('off')
    plt.tight_layout()
    plt.savefig(f"results_fin/outside_men_with_glass.png")
    plt.close()
    print(f"Image saved")
 
    fff, axx = plt.subplots(15,6, figsize=(12, 30))
    for i in range(5):
        axx[3*i + 1,0].imshow(men_no_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,0].axis('off')
        axx[3*i + 1,0].axis('off')
        axx[3*i + 2,0].axis('off')
        axx[3*i + 1,1].imshow(people_with_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,1].axis('off')
        axx[3*i + 1,1].axis('off')
        axx[3*i + 2,1].axis('off')
        axx[3*i + 1,2].imshow(people_no_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,2].axis('off')
        axx[3*i + 1,2].axis('off')
        axx[3*i + 2,2].axis('off')
        for j in range(9):
            axx[3*i + j//3,3+j%3].imshow(insides[i*9+j].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
            axx[3*i + j//3,3+j%3].axis('off')
    plt.tight_layout()
    plt.savefig(f"results_fin/inside_men_with_glass.png")
        

    insides = []
    outsides = []
    for i in range(5):
        inside = encoder(men_with_glasses[i]) - encoder(men_no_glasses[i]) + encoder(women_no_glasses[i])
        outside = encoder(men_with_glasses[i] - men_no_glasses[i] + women_no_glasses[i])
        outside_image = netG(outside.view(-1, 100, 1, 1))
        outsides.append(outside_image)
        for k in range(9):
            inside_image = netG(inside.view(-1, 100, 1, 1) + 1/2 * torch.randn(100).view(-1,100,1,1))
            insides.append(inside_image)
        print(f"{i+1}/5 Images")

    fig, axs = plt.subplots(5, 4, figsize=(8, 10))
    for i in range(5):
        axs[i,0].imshow(men_with_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,0].axis('off')
        axs[i,1].imshow(men_no_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,1].axis('off')
        axs[i,2].imshow(women_no_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,2].axis('off')
        axs[i,3].imshow(outsides[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,3].axis('off')
    plt.tight_layout()
    plt.savefig(f"results_fin/outside_women_with_glass.png")
    plt.close()
    print(f"Image saved")

    fff, axx = plt.subplots(15,6, figsize=(12, 30))
    for i in range(5):
        axx[3*i + 1,0].imshow(men_with_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,0].axis('off')
        axx[3*i + 1,0].axis('off')
        axx[3*i + 2,0].axis('off')
        axx[3*i + 1,1].imshow(men_no_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,1].axis('off')
        axx[3*i + 1,1].axis('off')
        axx[3*i + 2,1].axis('off')
        axx[3*i + 1,2].imshow(women_no_glasses[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,2].axis('off')
        axx[3*i + 1,2].axis('off')
        axx[3*i + 2,2].axis('off')
        for j in range(9):
            axx[3*i + j//3,3+j%3].imshow(insides[i*9+j].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
            axx[3*i + j//3,3+j%3].axis('off')
        plt.tight_layout()
        plt.savefig(f"results_fin/inside_women_with_glass.png")
    
    insides = []
    outsides = []
    for i in range(5):
        inside = encoder(men_with_smile[i]) + encoder(people_with_hat[i]) - encoder(people_no_hat[i]) + encoder(people_with_mus[i]) - encoder(people_no_mus[i])
        outside = encoder(men_with_smile[i] + people_with_hat[i] - people_no_hat[i] + people_with_mus[i] - people_no_mus[i])
        outside_image = netG(outside.view(-1, 100, 1, 1))
        outsides.append(outside_image)
        for k in range(9):
            inside_image = netG(inside.view(-1, 100, 1, 1) + 1/2 * torch.randn(100).view(-1,100,1,1))
            insides.append(inside_image)
        print(f"{i+1}/5 Images")


    fig, axs = plt.subplots(5, 6, figsize=(12, 10))
    for i in range(5):
        axs[i,0].imshow(men_with_smile[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,0].axis('off')
        axs[i,1].imshow(people_with_hat[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,1].axis('off')
        axs[i,2].imshow(people_no_hat[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,2].axis('off')
        axs[i,3].imshow(people_with_mus[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,3].axis('off')
        axs[i,4].imshow(people_no_mus[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,4].axis('off')
        axs[i,5].imshow(outsides[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axs[i,5].axis('off')
    plt.tight_layout()
    plt.savefig(f"results_ fin/outside_men_with_hat_smile_mustache.png")
    plt.close()
    print(f"Image saved")

    fff, axx = plt.subplots(15,8, figsize=(16, 30))
    for i in range(5):
        axx[3*i + 1,0].imshow(men_with_smile[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,0].axis('off')
        axx[3*i + 1,0].axis('off')
        axx[3*i + 2,0].axis('off')
        axx[3*i + 1,1].imshow(people_with_hat[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,1].axis('off')
        axx[3*i + 1,1].axis('off')
        axx[3*i + 2,1].axis('off')
        axx[3*i + 1,2].imshow(people_no_hat[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,2].axis('off')
        axx[3*i + 1,2].axis('off')
        axx[3*i + 2,2].axis('off')
        axx[3*i + 1,3].imshow(people_with_mus[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,3].axis('off')
        axx[3*i + 1,3].axis('off')
        axx[3*i + 2,3].axis('off')
        axx[3*i + 1,4].imshow(people_no_mus[i].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
        axx[3*i + 0,4].axis('off')
        axx[3*i + 1,4].axis('off')
        axx[3*i + 2,4].axis('off')
        for j in range(9):
            axx[3*i + j//3,5+j%3].imshow(insides[i*9+j].view(3,64,64).detach().cpu().numpy().transpose(1, 2, 0))
            axx[3*i + j//3,5+j%3].axis('off')
        plt.tight_layout()
        plt.savefig(f"results_fin/inside_men_with_hat_smile_mustache.png")


if __name__ == "__main__":
    main()
    