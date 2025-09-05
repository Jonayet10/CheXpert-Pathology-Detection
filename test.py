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
import csv


class CustomModel(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=9):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=1)
        # Modify the first layer to accept single-channel input
        if hasattr(self.model, 'conv1'):
            conv1 = self.model.conv1
            self.model.conv1 = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size,
                                         stride=conv1.stride, padding=conv1.padding, bias=conv1.bias)
        elif hasattr(self.model, 'stem'):
            stem = self.model.stem
            for name, module in stem.named_children():
                if name == 'conv1':
                    self.model.stem[0] = nn.Conv2d(1, module.out_channels, kernel_size=module.kernel_size,
                                                   stride=module.stride, padding=module.padding, bias=module.bias)

    def forward(self, x):
        return self.model(x)


class CustomDatasetTest(Dataset):
    def __init__(self, pickle_dir):
        self.pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pickle')]
        self.data = []
        for pickle_file in self.pickle_files:
            with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
                self.data.extend(pickle.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        img_id = data_point['info']['id']
        image = data_point['image']
        return image, img_id


def test_and_write(dataloader, model, device, csv_file):
    model.eval()
    with torch.no_grad():
        for images, img_ids in dataloader:
            images = images.to(device)
            predictions = model(images)

            for i in range(len(predictions)):
                prediction = predictions[i]
                img_id = img_ids[i]

                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([img_id] + prediction.tolist())
    logging.info(f"Testing completed.")


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
    lr = 0.001
    n_epochs = 50
    batch_size = 64
    batch_size_test = 64
    RANDOM_SEED = 1839
    pretrained = True

    # We can adjust to get this as an argument later
    dirname = 'test1'
    timestamp = int(datetime.timestamp(datetime.now()))
    output_dir = os.path.join("outputs", f"{dirname}_{timestamp}")
    model_dir = os.path.join(output_dir, "model")

    # For all output
    os.makedirs(output_dir, exist_ok=True)

    log_file = open(os.path.join(output_dir, 'output.txt'), 'w')
    handler = logging.StreamHandler(log_file)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using {device}.')

    # For reproducibility
    set_seed(RANDOM_SEED)

    # Test set
    pickle_dir_test = './data/chexpert_test_ids_size_128'
    custom_dataset_test = CustomDatasetTest(pickle_dir_test)
    dataloader_test = DataLoader(custom_dataset_test, batch_size=batch_size_test, shuffle=False)

    # Model Creation
    model = CustomModel('resnetv2_50x1_bit', pretrained=pretrained, num_classes=9).to(device)

    saved_model_dir = os.path.join('outputs', 'pretrained_sgd_128_1713500471', 'model', 'latest.ckpt')
    loaded_model = torch.load(saved_model_dir, map_location=torch.device(device))
    model.load_state_dict(loaded_model['model_state_dict'])


    logging.info(f'Start training ...')

    csv_dir = os.path.join(output_dir, 'predictions.csv')
    with open(csv_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Pneumonia",
                         "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"])

    test_and_write(dataloader_test, model, device, csv_dir)


if __name__ == "__main__":
    main()
