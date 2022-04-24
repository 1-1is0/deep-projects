import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn


class AlexConv2(nn.Module):
    def __init__(self):
        super(AlexConv2, self).__init__()
        self.conv2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(192, 192, kernel_size=5, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(192, 96, kernel_size=5, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(96, 48, kernel_size=5, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(48, 24, kernel_size=5, stride=2, padding=1)
        self.upconv5 = nn.ConvTranspose2d(24, 3, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.upconv1(x))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv5(x))
        return x


class AlexConv5(nn.Module):
    def __init__(self):
        super(AlexConv5, self).__init__()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1)
        self.upconv5 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.upconv1(x))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv5(x))
        return x

class AlexFc6(nn.Module):
    def __init__(self):
        super(AlexFc6, self).__init__()
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
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = x.view(x.size(0), 256, 4, 4)
        x = F.leaky_relu(self.upconv1(x))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv5(x))
        return x


class AlexFc8(nn.Module):
    def __init__(self):
        super(AlexFc8, self).__init__()
        self.fc1 = nn.Linear(1000, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.upconv5 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = x.view(x.size(0), 256, 4, 4)
        x = F.leaky_relu(self.upconv1(x))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv5(x))
        return x


def train_model(data_loader, model, criterion, optimizer, scheduler, epochs, device):
    loss_history = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        log = 0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # name, image, fix_map = data
            image = data["image"].to(device)
            input_image = data["alex"].to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_image)
            # resize the image
            target_size = outputs.size()[-1]
            image_resize = fn.resize(image, size=target_size)

            loss = criterion(outputs, image_resize)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            log += 1
            if len(data_loader) / 5 == log:
                print(". ", end="")
                log = 0
        # add loss at the end of each iteration
        loss_history.append(running_loss)
        print("lr before", optimizer.param_groups[0]['lr'], end="")
        scheduler.step()
        print("lr after", optimizer.param_groups[0]['lr'], end="### \n")
        print("epoch", epoch, running_loss / len(data_loader))

    print("Finished Training")
    return loss_history
