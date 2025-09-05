import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import pickle
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import torch.nn as nn
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import random
from absl import app
from absl import flags
import numpy as np
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=9):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=3)

        # Find and replace the first convolutional layer
        first_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                first_conv = module
                break

        if first_conv is not None:
            # Creating a new Conv2d to replace the first convolutional layer
            new_conv = nn.Conv2d(1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                                 stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias)
            if pretrained:
                # Initialize new_conv weights from original weights (average RGB weights)
                new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data

            # Replace the first convolutional layer
            if name == 'conv1':
                self.model.conv1 = new_conv
            else:
                # Replace the module within nested structures
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = dict(self.model.named_modules())[parent_name]
                setattr(parent_module, child_name, new_conv)

    def forward(self, x):
        return self.model(x)


class CustomDataset(Dataset):
    def __init__(self, pickle_dir):
        self.pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pickle')]
        self.data = []
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95,1.05)),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ])
        for pickle_file in self.pickle_files:
            with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
                self.data.extend(pickle.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        image = data_point['image']
        labels = torch.tensor(data_point['info']['labels'], dtype=torch.float32)
        additional_info = torch.tensor(data_point['info']['additional_info'], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels, additional_info


class CustomDatasetTest(Dataset):
    def __init__(self, pickle_dir):
        self.pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pickle')]
        self.data = []
        self.transform = transforms.Compose([
              transforms.Normalize(mean=[0.449], std=[0.226]),
            ])
        for pickle_file in self.pickle_files:
            with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
                self.data.extend(pickle.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        image = data_point['image']
        if self.transform:
            image = self.transform(image)
        return image


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    running_loss = 0.0
    for batch, (images, labels, additional_info) in enumerate(dataloader):
        optimizer.zero_grad()

        images, labels, additional_info = images.to(device), labels.to(device), additional_info.to(device)

        #  For CrossEntropyLoss
        labels = F.softmax(labels, dim=-1)

        prediction = model(images)
        loss = loss_fn(prediction, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    loss_of_epoch = running_loss/len(dataloader)
    logging.info(f'Loss: {loss_of_epoch:.4f}')

    return loss_of_epoch


def validation(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, labels, additional_info in dataloader:
            images, labels, additional_info = images.to(device), labels.to(device), additional_info.to(device)
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
    test_loss /= num_batches
    logging.info(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")

    return test_loss


def set_seed(RANDOM_SEED):
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():

    num_classes = 9
    lr = 0.01
    n_epochs = 30
    batch_size = 64
    batch_size_test = 64
    RANDOM_SEED = 1839
    pretrained = True

    dirname = 'celoss_resnetv2_50x1_bit.goog_in21k_less_rotate_translate_scale_normalize_resnet50_bs-64_pretrained_256'
    timestamp = int(datetime.timestamp(datetime.now()))
    output_dir = os.path.join("outputs", f"{dirname}_{timestamp}")
    model_dir = os.path.join(output_dir, "model")

    # For all output
    os.makedirs(output_dir, exist_ok=True)
    # For output model (checkpoints)
    os.makedirs(model_dir, exist_ok=True)

    # Record current config
    with open(os.path.join(output_dir, "classification.py"), 'w') as fout:
        with open('./classification.py', 'r') as fin:
            fout.write(fin.read())

    log_file = open(os.path.join(output_dir, 'output.txt'), 'w')
    handler = logging.StreamHandler(log_file)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    plot_dir = os.path.join(output_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using {device}.')

    # For reproducibility
    set_seed(RANDOM_SEED)

    # Training set
    pickle_dir = './data/chexperttrain2023_size_256'
    custom_dataset = CustomDataset(pickle_dir)

    # For train and validation split
    train_size = int(0.95 * len(custom_dataset))
    validation_size = len(custom_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(custom_dataset, [train_size, validation_size])

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_validation = DataLoader(validation_dataset, batch_size=batch_size_test, shuffle=False)

    # Test set
    # pickle_dir_test = ''
    # custom_dataset_test = CustomDatasetTest(pickle_dir_test)
    # dataloader_test = DataLoader(custom_dataset_test, batch_size=batch_size_test, shuffle=True)

    # Model Creation
    # model = (timm.create_model('resnetv2_50x1_bit.goog_in21k', pretrained=True, num_classes=9, in_chans=3)).to(device)
    model = CustomModel('resnetv2_50x1_bit.goog_in21k', pretrained=pretrained, num_classes=9).to(device)

    # First layer accepts grayscale images
    # model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)
    #  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #  optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epoch, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)


    logging.info(f'Start training ...')
    losses = []
    losses_validation = []
    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch+1}\n-------------------------------")

        loss = train(dataloader_train, model, loss_fn, optimizer, device)
        losses.append(loss)

        loss_validation = validation(dataloader_validation, model, loss_fn, device)
        losses_validation.append(loss_validation)

        scheduler.step()

        plt.figure(figsize=(10, 5))
        plt.title("Training losses")
        plt.plot(losses, label="Training loss")
        plt.ylim(ymin=0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'Training_losses.png'))

        plt.figure(figsize=(10, 5))
        plt.title("Validation losses")
        plt.plot(losses_validation, label="Validation loss")
        plt.ylim(ymin=0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'Validation_losses.png'))

        plt.figure(figsize=(10, 5))
        plt.title("Training and validation losses")
        plt.plot(losses, label="Training loss")
        plt.plot(losses_validation, label="Validation loss")
        plt.ylim(ymin=0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'Losses.png'))

        if (epoch % 10) == 0:
            # Save the trained model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(model_dir, f'latest.ckpt'))

    # Save the trained model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_dir, f'latest.ckpt'))





if __name__ == "__main__":
    main()
