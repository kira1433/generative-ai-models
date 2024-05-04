import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from PIL import Image


batch_size=32

class CelebADataset(Dataset):
    def __init__(self, mode, data_dir = '/mnt/MIG_store/Datasets/celeba'):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ]
        )

        self.df = pd.read_csv(os.path.join(data_dir, "list_attr_celeba.csv"))
        if mode == "men_no_glasses":
            self.df = self.df[(self.df["Male"] == 1) & (self.df["Eyeglasses"] == -1)]
        elif mode == "men_with_glasses":
            self.df = self.df[(self.df["Male"] == 1) & (self.df["Eyeglasses"] == 1)]
        elif mode == "women_with_glasses":
            self.df = self.df[(self.df["Male"] == -1) & (self.df["Eyeglasses"] == 1)]
        else:
            raise ValueError("Invalid mode")

        self.image_paths = [
            os.path.join(data_dir, "img_align_celeba", "img_align_celeba", f"{img_id}")
            for img_id in self.df["image_id"]
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
men_no_glasses_dataset = CelebADataset("men_no_glasses")
men_with_glasses_dataset = CelebADataset("men_with_glasses")
women_with_glasses_dataset = CelebADataset("women_with_glasses")

# Dataloaders
men_no_glasses_loader = DataLoader(men_no_glasses_dataset, batch_size=batch_size, shuffle=True)
men_with_glasses_loader = DataLoader(men_with_glasses_dataset, batch_size=batch_size, shuffle=True)
women_with_glasses_loader = DataLoader(women_with_glasses_dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    print(men_no_glasses_dataset.__len__())
    print(men_with_glasses_dataset.__len__())
    print(women_with_glasses_dataset.__len__())