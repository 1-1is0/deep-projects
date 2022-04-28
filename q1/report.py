import torch
import os
import numpy as np
from dataloader import load_data, ImageDataset, Rescale, ToTensor
import matplotlib.pyplot as plt
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load('state/state-Net.pth', map_location='cpu')
res = state['res']
# fig = plt.figure()

def plot_fig():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Loss')
    ax[0].plot(np.arange(len(res["epoch_loss_train"])), res["epoch_loss_train"], label="train")
    ax[0].legend()
    ax[0].set_title("Train Loss")

    ax[1].plot(np.arange(len(res["epoch_loss_val"])), res["epoch_loss_val"], label="val")
    ax[1].legend()
    ax[1].set_title("Validation Loss")

    plt.savefig("report/loss.png")


def show_im():

    train_df, test_df = load_data()
    print("Train:", train_df.shape)
    print("Test:", test_df.shape)
    train_dataset = ImageDataset(train_df,
                                 transform=transforms.Compose([
                                     Rescale((320, 240)),
                                     # RandomCrop(224),
                                     ToTensor()
                                 ]))


    test_dataset = ImageDataset(test_df,
                                transform=transforms.Compose([
                                    Rescale((320, 240)),
                                    # RandomCrop(224),
                                    ToTensor()
                                ]))



    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                              shuffle=True, num_workers=1)

    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
    #                                          shuffle=True, num_workers=1)
    from model import Net
    net = Net().to(device)
    print(net)
    with torch.no_grad():
        net.load_state_dict(state['state_dict'])
        net.eval()

        fig, ax = plt.subplots(6, 3, figsize=(10, 15))
        fig.suptitle('Prediction')
        for i in range(6):
            rand_num = np.random.randint(0, len(train_dataset))
            sample = train_dataset[rand_num]
            # i = i.unsqueeze(0)
            predict = net(sample['image'].float())

            ax[i, 0].imshow(sample['image'].permute(1, 2, 0).detach().numpy())
            ax[i, 0].set_title('Original Image')

            ax[i, 1].imshow(sample['fix_map'].permute(1, 2, 0).detach().numpy())
            ax[i, 1].set_title('Fixation Map')

            ax[i, 2].imshow(predict.permute(1, 2, 0).detach().numpy())
            ax[i, 2].set_title('Prediction')

        plt.savefig("report/prediction.png")


def gen_report():
    os.makedirs('report', exist_ok=True)
    plot_fig()
    show_im()


gen_report()