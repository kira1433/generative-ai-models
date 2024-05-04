import sys
import torch
import matplotlib.pyplot as plt
from cycle_gan_model import CycleGAN
from dataset import men_with_glasses_loader, men_no_glasses_loader, women_with_glasses_loader


def train(mode):
    model = CycleGAN().to('cuda:' + mode)
    # print(model)

    if mode == "1":
        dataloader_A = men_with_glasses_loader
        dataloader_B = men_no_glasses_loader
    else:
        dataloader_A = men_with_glasses_loader
        dataloader_B = women_with_glasses_loader

    for epoch in range(101):
        for iter,(real_A,real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            if(real_A.shape[0] != 32 or real_B.shape[0] != 32):
                    break
            model.set_requires_grad()
            x = real_A.to('cuda:' + mode)
            y = real_B.to('cuda:' + mode)
            model.set_input(x, y)
            loss_G, loss_D = model.optimize_parameters()
            print(f"Epoch: {epoch}, Iteration: {iter}, Loss_G: {loss_G}, Loss_D: {loss_D} \n")

        if epoch % 10 == 0:
            model.set_requires_grad(requires_grad=False)
            images = []
            for iter,(real_A,real_B) in enumerate(zip(dataloader_A, dataloader_B)):
                if(real_A.shape[0] != 32 or real_B.shape[0] != 32):
                    break
                x = real_A.to('cuda:' + mode)
                y = real_B.to('cuda:' + mode)
                model.set_input(x, y)
                model.forward()
                #save images
                for i in range(32):
                    image = [real_A[i], real_B[i], model.fake_A[i], model.fake_B[i], model.rec_A[i], model.rec_B[i]]
                    fig, axs = plt.subplots(1, 6, figsize=(18, 3))
                    for j in range(6):
                        axs[j].imshow(image[j].detach().cpu().numpy().transpose(1, 2, 0))
                        axs[j].axis('off')
                    plt.tight_layout()
                    plt.savefig(f"results_{mode}/images_{epoch}_{i}.png")
                    plt.close()
                print(f"Images saved for epoch {epoch}")
                torch.save(model.state_dict(), f"checkpoints/model_{mode}.pth")
                print(f"Model saved for epoch {epoch}")
                break


if __name__ == '__main__':
    mode = sys.argv[1]
    train(mode)