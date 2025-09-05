import torch
from PIL import Image
import torchvision.transforms as transforms 
from torchvision.transforms import v2
from torchvision.utils import save_image

def elastic_image(file_path, alpha, output_path):
    try:
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        shorter_edge = min(image.size)

        random_rotation = transforms.Compose([
            transforms.ElasticTransform(alpha=alpha),
            transforms.ToTensor()
            # Add any additional transformations here
        ])
        image = random_rotation(image)
        save_image(image, output_path)
        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

# elastic_image("view1_frontal.jpg", 250.0, "img1.png")