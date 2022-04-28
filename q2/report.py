import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader import load_data, ImageDataset, Rescale, ToTensor, GetAlexConv, get_datasets, get_data_loader
from torchvision import transforms

from model import AlexConv5, AlexFc6, AlexConv2

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# state = torch.load('data/state/state-AlexConv2.pth', map_location='cpu')
state = torch.load('data/state/state-AlexConv2.pth', map_location='cpu')

res = state['res']


data_path = None
train_df, test_df = load_data()
train_dataset, test_dataset = get_datasets(train_df, test_df)

print("train_dataset:", len(train_dataset))
print("test_dataset", len(test_dataset))

alxeconv5 = AlexConv2().to(device)
print(alxeconv5)
with torch.no_grad():
    alxeconv5.load_state_dict(state['state_dict'])
    alxeconv5.eval()
    for i in range(5):
        r = np.random.randint(0, len(train_dataset))
        img = train_dataset[r]['image']
        print(img.shape)
        ans = alxeconv5(img.unsqueeze(0).to(device))
        ans = ans.squeeze(0).cpu()
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img.permute(1, 2, 0))
        ax[1].imshow(ans.permute(1, 2, 0))
        plt.show()
        # plt.imshow(img.transpose(1, 2, 0))
        # print(ans.shape)
print(res)