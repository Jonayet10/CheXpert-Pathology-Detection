import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
from pathlib import Path

img_size = 256
output_dir = './cs156_dataset_size_{}_singleprocessing'.format(img_size)
Path(output_dir).mkdir(parents=True, exist_ok=True)

def rotate_image(img, angle):
    return TF.rotate(img, angle)

def translate_image(img, tx, ty):
    return TF.affine(img, angle=0, translate=(tx, ty), scale=1, shear=0)


def dilate_image(img, size):
    kernel = torch.ones((size, size), dtype=torch.uint8)
    img = TF.to_pil_image(img)
    img = TF.to_tensor(TF.dilate(img, kernel))
    return img

def load_image(file_path, img_size):
    try:
        image = Image.open(file_path).convert('L')  

        preprocess = transforms.Compose([
            transforms.ElasticTransform(alpha=100.0, p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomAffine(degree=(-10, 10), translate=(-0.05, 0.05)),
            transforms.Lambda(lambda img: dilate_image(img, size=3))
        ])
        image = preprocess(image)

        if random.random() < 0.3:
            image = dilate_image(image, size=3)

        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

if __name__ == "__main__":

    file_path = 'view1_frontal.jpg'
    img_tensor = load_image(file_path, img_size)
    
    if img_tensor is not None:
        file_name = Path(output_dir) / 'augmented_image.jpg'
        TF.to_pil_image(img_tensor).save(file_name)
        print(f"Image saved at {file_name}")
