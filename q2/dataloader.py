from glob import glob

import numpy as np
import pandas as pd
import torch
from skimage import io, transform, color
from torch.utils.data import Dataset
from torchvision import models


def load_data(path="data/tiny-imagenet-200/train/*"):
    np.random.seed(42)
    all_dirs = glob(path)
    dirs = np.random.choice(all_dirs, 22)
    data = []
    for img_dir in dirs:
        imgs_in_dir = glob(img_dir + "/images/*.JPEG")
        name = img_dir.split("/")[-1]
        for img in imgs_in_dir:
            data.append([name, img])

    data_df = pd.DataFrame(data, columns=["name", "image"])

    test_df = data_df.groupby("name").sample(n=20)
    train_df = data_df[~data_df.index.isin(test_df.index)]
    return test_df, train_df


torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


class ImageDataset(Dataset):
    def __init__(self, df, transform=None, alexnet_feature=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        # self.root_dir = root_dir
        self.transform = transform
        self.alexnet_feature = alexnet_feature

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.df.iloc[idx].image)
        # image = read_image(self.df.iloc[idx].image).float()
        # image = Image.open(self.df.iloc[idx].image)
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        # adds another dimension to the image channel
        name = f"{self.df.iloc[idx].name}"
        sample = {"name": name, "image": image}
        # print("name", sample['name'], sample['image'].shape)
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        if self.alexnet_feature:
            sample["alex"] = self.alexnet_feature(sample["image"])

        return sample


class ToTensor(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size  # tuple

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img


class GetAlexConv(object):
    """
    Returns the feature map layer of the AlexNet model
    """

    def __init__(self, layer_number, layer_name):
        self.layer_number = layer_number
        self.layer_name = layer_name

        self.alex = models.alexnet(pretrained=True).to(device)
        for param in self.alex.parameters():
            param.requires_grad = False
        child = list(self.alex.children())
        conv_layer = child[self.layer_number[0]][self.layer_number[1]]
        conv_layer.register_forward_hook(get_activation(self.layer_name))

    def __call__(self, image):
        batch = image.to(device).unsqueeze(0)
        self.alex(batch)
        feature = activation[self.layer_name].squeeze(0)
        return feature
