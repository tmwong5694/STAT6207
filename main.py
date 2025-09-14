import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
import torchvision.transforms.v2 as v2
import numpy as np
import os
from PIL import Image
import kagglehub


def download_dataset() -> None:
    """
    Functions to download cat and dog dataset

    The dataset should be downloaded to a path:
    ~/.cache/kagglehub/datasets/bhavikjikadara/dog-and-cat-classification-dataset/versions/<N>
    """
    path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")

trans = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    # Use the mean and std vectors of ResNet to normalize
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def image_to_tensor(pil_img: Image.Image) -> Tensor:
    """Convert an image into tensor and add dimension if it does not reach dimmension 3."""

    # Transform the image into tensor
    input_tensor = trans(pil_img)
    # Add a dimension if the ndim is 2
    if input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(0)
    return input_tensor



weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)



if __name__ == '__main__':
    ## The dataset is already download to this project folder at "./data"
    # download_dataset()
    cat_folder = "./data/dog-and-cat-classification-dataset/versions/1/PetImages/Cat"
    dog_folder = "./data/dog-and-cat-classification-dataset/versions/1/PetImages/Dog"
    cat = os.path.join(cat_folder, "0.jpg")
    img = Image.open(cat)
    
    #img.show()
    tensor = image_to_tensor(pil_img=img)

    pass