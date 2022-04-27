import torch
from glob import glob

import numpy as np
import pandas as pd
import torch
from skimage import io, transform, color
from torch.utils.data import Dataset

np.random.seed(42)


def load_data(path="data/Stimuli/*/*.jpg"):
    all_images = glob(path)
    # print(all_images)
    data = []
    for img in all_images:
        stimuli = img.split("/")[2]
        data.append([stimuli, img, img.replace("Stimuli", "FIXATIONMAPS")])

    data_df = pd.DataFrame(data, columns=["Stimuli", "Image", "FixationMap"])
    train_df = (
        data_df.groupby("Stimuli")
        .apply(lambda x: x.sample(int((x.count() * 85 / 100).Image)))
        .reset_index(drop=True)
    )
    test_df = data_df[~data_df.Image.isin(train_df.Image)]

    return train_df, test_df


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.df.iloc[idx].Image)
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        # adds another dimension to the image channel
        fix_map = io.imread(self.df.iloc[idx].FixationMap)
        fix_map = np.expand_dims(fix_map, axis=2)

        name = f"{self.df.iloc[idx].Image}"
        sample = {"name": name, "image": image, "fix_map": fix_map}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size  # tuple

    def __call__(self, sample):
        image, fix_map = sample["image"], sample["fix_map"]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for fix_map because for images,
        # x and y axes are axis 1 and 0 respectively
        fix = transform.resize(fix_map, (new_h, new_w))
        # fix_map = fix_map * [new_w / w, new_h / h]
        return {"name": sample["name"], "image": img, "fix_map": fix}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, fix_map = sample["image"], sample["fix_map"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(sample['name'], sample['image'].shape, sample['fix_map'].shape)
        image = image.transpose((2, 0, 1))
        fix_map = fix_map.transpose((2, 0, 1))
        return {
            "name": sample["name"],
            "image": torch.from_numpy(image).float(),
            "fix_map": torch.from_numpy(fix_map).float(),
        }
