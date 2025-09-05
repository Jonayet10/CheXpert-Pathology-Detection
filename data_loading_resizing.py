import csv
from pathlib import Path
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import pickle


img_size = 256
output_dir = f'./cs156_dataset_size_{img_size}_singleprocessing'
Path(output_dir).mkdir(parents=True, exist_ok=True)

def load_image(file_path, img_size):
    try:
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        shorter_edge = min(image.size)

        preprocess = transforms.Compose([
            transforms.CenterCrop(shorter_edge),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # Add any additional transformations here
        ])
        image = preprocess(image)
        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None
    

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


labels = list()
with open('train2023.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    next(reader)
    for row in reader:
        data_point = dict()
        data_point['path'] = row[2]

        is_male = int(row[3] == 'Male')
        age = int(row[4])
        is_frontal = int(row[5] == 'Frontal')
        is_ap = int(row[6] == 'AP')
        data_point['additional_info'] = [is_male, age, is_frontal, is_ap]

        no_finding = int(float(row[7] or 0))
        enlarged_cardiomediastinum = int(float(row[8] or 0))
        cardiomegaly = int(float(row[9] or 0))
        lung_opacity = int(float(row[10] or 0))
        pnuemonia = int(float(row[11] or 0))
        pleural_effusion = int(float(row[12] or 0))
        pleuroal_other = int(float(row[13] or 0))
        fracture = int(float(row[14] or 0))
        support_devices = int(float(row[15] or 0))

        data_point['labels'] = [no_finding, enlarged_cardiomediastinum, cardiomegaly, lung_opacity, pnuemonia,
                               pleural_effusion, pleuroal_other, fracture, support_devices]

        labels.append(data_point)

labels = [label_row for label_row in labels if
          os.path.exists(os.path.join('/central/groups/CS156b/data', label_row['path']))]

# Load the images as black and white
labels_chunks = list(chunks(labels, 5000))
for chunk_no, labels in enumerate(labels_chunks):
    dataset = []
    for label in labels:
        # image_path = os.path.join('/central/groups/CS156b/data', label['path'])
        image_path = os.path.join('/central/groups/CS156b/data', label['path'])
        if os.path.exists(image_path):
            image = load_image(image_path, img_size)

            data_point = dict()
            data_point = {'info': label, 'image': image}
            dataset.append(data_point)

    with open(os.path.join(output_dir, f'part_{chunk_no + 1}.pickle'), 'wb') as file:
        pickle.dump(dataset, file)

