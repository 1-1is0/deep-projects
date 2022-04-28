import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from torchvision import models


class AlexConv2(nn.Module):
    def __init__(self):
        super(AlexConv2, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        for params in self.alexnet.parameters():
            params.requires_grad = False
        self.alexnet.features = self.alexnet.features[:6]

        self.conv1 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.upconv1 = nn.ConvTranspose2d(
            256, 256, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv5 = nn.ConvTranspose2d(
            32, 3, kernel_size=5, stride=2, padding=2, output_padding=1
        )

    def forward(self, x):
        x = self.alexnet.features(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.batchnorm1(x))
        x = F.leaky_relu(self.upconv1(x))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv5(x))
        x = F.interpolate(x, 227)
        return x

    def initialize_weights(self):
        # https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


class AlexConv5(nn.Module):
    def __init__(self):
        super(AlexConv5, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        for params in self.alexnet.parameters():
            params.requires_grad = False
        # self.alexnet.features = self.alexnet.features[:6]

        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(
            256, 256, kernel_size=5, stride=2, padding=1, output_padding=1
        )
        self.upconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=5, stride=2, padding=1, output_padding=1
        )
        self.upconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, stride=2, padding=1, output_padding=1
        )
        self.upconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=5, stride=2, padding=1, output_padding=1
        )
        self.upconv5 = nn.ConvTranspose2d(
            32, 3, kernel_size=5, stride=2, padding=1, output_padding=1
        )

    def forward(self, x):
        x = self.alexnet.features(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.upconv1(x))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv5(x))
        x = F.interpolate(x, 227)
        return x

    def initialize_weights(self):
        # https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class AlexFc6(nn.Module):
    def __init__(self):
        super(AlexFc6, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        for params in self.alexnet.parameters():
            params.requires_grad = False
        self.alexnet.classifier = self.alexnet.classifier[:4]

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.upconv1 = nn.ConvTranspose2d(
            256, 256, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv5 = nn.ConvTranspose2d(
            32, 3, kernel_size=5, stride=2, padding=2, output_padding=1
        )

    def forward(self, x):
        x = self.alexnet(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = x.view(x.size(0), 256, 4, 4)
        x = F.leaky_relu(self.upconv1(x))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv5(x))
        x = F.interpolate(x, 227)
        return x


    def initialize_weights(self):
        # https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class AlexFc8(nn.Module):
    def __init__(self):
        super(AlexFc8, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        for params in self.alexnet.parameters():
            params.requires_grad = False

        self.fc1 = nn.Linear(1000, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.upconv1 = nn.ConvTranspose2d(
            256, 256, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv4 = nn.ConvTranspose2d(
            64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.upconv5 = nn.ConvTranspose2d(
            32, 3, kernel_size=5, stride=2, padding=2, output_padding=1
        )

    def forward(self, x):
        x = self.alexnet(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = x.view(x.size(0), 256, 4, 4)
        x = F.leaky_relu(self.upconv1(x))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv5(x))
        x = F.interpolate(x, 227)
        return x


    def initialize_weights(self):
        # https://www.askpython.com/python-modules/initialize-model-weights-pytorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def train_model(
    data_loader,
    dataset_size,
    model,
    criterion,
    optimizer,
    scheduler,
    epochs,
    device,
    path="/content/drive/MyDrive/data/state",
):
    state_file_name = f"{path}/state-{model._get_name()}.pth"
    print("state_file_name", state_file_name)
    state_res = {}
    state_epoch = 0
    if os.path.exists(state_file_name):
        state = torch.load(state_file_name)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        state_res = state["res"]
        state_epoch = state["epoch"]

    res = {
        "epoch_loss_train": state_res.get("epoch_loss_train", []),
        "loss_history_train": state_res.get("loss_history_train", []),
        "lr": state_res.get("lr", []),
        "epoch_loss_val": state_res.get("epoch_loss_val", []),
        "loss_history_val": state_res.get("loss_history_val", []),
    }
    # remaining epochs
    epochs = epochs - state_epoch
    # loss_history = []
    # epochs_loss = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            log = 0

            for i, data in enumerate(data_loader[phase], 0):
                # get the inputs; data is a list of [inputs, labels]
                # name, image, fix_map = data
                image = data["image"].to(device)
                # input_image = data["alex"].to(device)
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(image)
                # resize the image
                # target_size = outputs.size()[-1]
                # image_resize = fn.resize(image, size=target_size)
                loss = criterion(outputs, image)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                # print statistics
                # print("data shape", data["image"].shape)
                now_batch_size = data["image"].size()[0]
                # print("len dataloader", dataset_size[phase], "now batch size", now_batch_size)

                running_loss += loss.item() * now_batch_size
                log += 1
                if len(data_loader) / 5 == log:
                    print(". ", end="")
                    log = 0
            # add loss at the end of each iteration
            # loss_history.append(running_loss)
            if phase == "train":
                res["lr"].append(optimizer.param_groups[0]["lr"])
                scheduler.step()


            epoch_loss = running_loss / dataset_size[phase]

            print("epoch", epoch, epoch_loss)
            res[f"epoch_loss_{phase}"].append(epoch_loss)
            res[f"loss_history_{phase}"].append(running_loss)

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "res": res,
        }
        torch.save(state, state_file_name)
    print("Finished Training")
    return res
