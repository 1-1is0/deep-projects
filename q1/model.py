import torch
import os
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 96, 7, stride=1, padding=3)
        self.local_response_norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        self.max_pool1 = nn.MaxPool2d((3,3), stride=2)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.max_pool2 = nn.MaxPool2d((3,3), stride=2, padding=0)
        self.conv3 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(512, 512, 5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(512, 256, 7, stride=1, padding=3)
        self.conv7 = nn.Conv2d(256, 128, 11, stride=1, padding=5)
        self.conv8 = nn.Conv2d(128, 32, 11, stride=1, padding=5)
        # self.conv9 = nn.Conv2d(32, 1, 13, stride=1, padding=6)
        # I remove padding 2 from Conv2d Transpose because the output dimension was not right
        self.deconv = nn.ConvTranspose2d(32, 1, 8, stride=4, bias=False)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.local_response_norm(x)
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.deconv(x)
        return x

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
                image = data['image'].to(device)
                fix_map = data['fix_map'].to(device)
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(image)
                loss = criterion(outputs, fix_map)
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

            print("epoch", epoch, running_loss / len(data_loader[phase]))

            epoch_loss = running_loss / dataset_size[phase]
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
