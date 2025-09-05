import torch
from PIL import Image
import torchvision.transforms as transforms 
from torchvision.transforms import v2
from torchvision.utils import save_image

def flip_image(file_path, sharpness, output_path):
    try:
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        shorter_edge = min(image.size)

        random_rotation = transforms.Compose([
            transforms.RandomApply(transforms=[transforms.RandomHorizontalFlip(p = 0.5), transforms.RandomVerticalFlip(p = 0.5)], p = 0.5),
            transforms.ToTensor()
            # Add any additional transformations here
        ])
        image = random_rotation(image)
        save_image(image, output_path)
        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# flip_image("view1_frontal.jpg", 0, "img1.png")