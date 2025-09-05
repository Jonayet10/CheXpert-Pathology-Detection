import torch
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms 
from torchvision.transforms import v2
from torchvision.utils import save_image

class EdgeDetection(object):
    def __init__(self, p = 0.5):
        self.p = p 

    def __call__(self, img):
        if torch.rand(1) < self.p:
            img_2 = cv2.GaussianBlur(np.transpose(img.numpy(), (1, 2, 0)), (3, 3), 0)
            edges = cv2.Canny((img_2*255).astype(np.uint8), threshold1 = 100, threshold2 = 150)
            img = torch.from_numpy(edges.astype(np.float32))
            save_image(torch.from_numpy(edges.astype(np.float32)), "img1.png")
            return img 
        return img

    def __repr_(self) -> str:
        return f"{self.__class__.__name__}(p = {self.p})"


def affine_image(file_path, degree_amount, translate_amount, output_path):
    try:
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        shorter_edge = min(image.size)

        random_rotation = transforms.Compose([
            transforms.RandomAffine(degrees=(-degree_amount, degree_amount),
                                    translate=(0, translate_amount)),
            transforms.ToTensor(),
            EdgeDetection()
            # Add any additional transformations here
        ])
        image = random_rotation(image)
        # image_2 = cv2.GaussianBlur(np.transpose(image.numpy(), (1, 2, 0)), (3, 3), 0)
        # edges = cv2.Canny((image_2*255).astype(np.uint8), threshold1 = 100, threshold2 = 150)
        # save_image(torch.from_numpy(edges.astype(np.float32)), output_path)
        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

affine_image("view1_frontal.jpg", 20, 0.1, "img1.png")