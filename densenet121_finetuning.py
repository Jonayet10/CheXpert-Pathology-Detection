import os
import torch
import torchxrayvision as xrv
import skimage.io
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class XRayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels_frame.iloc[idx, 2])
        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255)  # Adjust the pixel values

        # Ensure image is single-channel
        if len(img.shape) > 2:  # If this, image is not grayscale
            img = img.mean(2)
        img = img[None, :, :]  # Allows to be batch processed by model (1, height, width)

        if self.transform:
            img = self.transform(img)

        labels = self.labels_frame.iloc[idx, 3:].values
        labels = torch.from_numpy(labels.astype('float32'))
        
        return img, labels

transform = transforms.Compose([
    xrv.datasets.XRayCenterCrop(), 
    xrv.datasets.XRayResizer(224)
])

dataset = XRayDataset(csv_file='train2023.csv', img_dir='/central/groups/CS156b/data', transform=transform) # change img_dir to where the images are stored, same with csv
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xrv.models.DenseNet(weights="densenet121-res224-nih").to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.float().to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch {epoch+1} Loss: {epoch_loss:.4f}')

print("Training complete")
