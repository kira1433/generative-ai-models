import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  Dataset
import os
import pandas as pd
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, mode, data_dir = '/mnt/MIG_store/Datasets/celeba'):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        )
        
        if mode == "men_with_glasses":
            self.originals = ["182647.jpg", "182648.jpg", "182662.jpg", "182673.jpg", "182689.jpg"]
        elif mode == "men_no_glasses":
            self.originals = ["182643.jpg", "182644.jpg", "182649.jpg", "182652.jpg", "182653.jpg"]
        elif mode == "women_no_glasses":
            self.originals = ["182638.jpg", "182639.jpg", "182640.jpg", "182641.jpg", "182642.jpg"]
        elif mode == "people_with_glasses":
            self.originals = ["182647.jpg", "182648.jpg", "182662.jpg", "182671.jpg", "182673.jpg"]
        elif mode == "people_no_glasses":
            self.originals = ["182638.jpg", "182639.jpg", "182640.jpg", "182641.jpg", "182642.jpg"]
        elif mode == "men_with_smile":
            self.originals = ["182649.jpg", "182675.jpg", "182681.jpg", "182682.jpg", "182686.jpg"]
        elif mode == "people_with_hat":
            self.originals = ["182638.jpg", "182680.jpg", "182687.jpg", "182689.jpg", "182725.jpg"]
        elif mode == "people_no_hat":
            self.originals = ["182638.jpg", "182680.jpg", "182687.jpg", "182689.jpg", "182725.jpg"]
        elif mode == "people_with_mus":
            self.originals = ["182647.jpg", "182649.jpg", "182742.jpg", "182797.jpg", "182806.jpg"]
        elif mode == "people_no_mus":
            self.originals = ["182638.jpg", "182639.jpg", "182640.jpg", "182641.jpg", "182642.jpg"]

        self.image_paths = [
            os.path.join(data_dir, "img_align_celeba", "img_align_celeba", f"{img_id}")
            for img_id in self.originals
        ]
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

# Datasets
men_no_glasses = [CelebADataset("men_no_glasses").__getitem__(i).view(1,3,64,64) for i in range(5)]
people_with_glasses = [CelebADataset("people_with_glasses").__getitem__(i).view(1,3,64,64) for i in range(5)]
people_no_glasses = [CelebADataset("people_no_glasses").__getitem__(i).view(1,3,64,64) for i in range(5)]
men_with_glasses = [CelebADataset("men_with_glasses").__getitem__(i).view(1,3,64,64) for i in range(5)]
women_no_glasses = [CelebADataset("women_no_glasses").__getitem__(i).view(1,3,64,64) for i in range(5)]
men_with_smile = [CelebADataset('men_with_smile').__getitem__(i).view(1,3,64,64) for i in range(5)]
people_with_hat = [CelebADataset('people_with_hat').__getitem__(i).view(1,3,64,64) for i in range(5)]
people_no_hat = [CelebADataset('people_no_hat').__getitem__(i).view(1,3,64,64) for i in range(5)]
people_with_mus = [CelebADataset('people_with_mus').__getitem__(i).view(1,3,64,64) for i in range(5)]
people_no_mus = [CelebADataset('people_no_mus').__getitem__(i).view(1,3,64,64) for i in range(5)]

print("Dataset loaded successfully")