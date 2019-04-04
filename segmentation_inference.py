import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from auggi_dataset_david import AuggiDetectionDataset
from models.segnet import SegNet
from helpers.iou import calculateIOU
import datetime
import torchvision.transforms.functional as F
from bbox_from_mask import BoundingBox
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


# IMAGE DIMENSIONS
IMG_SHAPE = (224, 224)
IMAGE_SIZE = IMG_SHAPE[0]*IMG_SHAPE[1]
IMAGE_WIDTH = IMG_SHAPE[0]
IMAGE_HEIGHT = IMG_SHAPE[1]
N_LABELS = 1


# global vars
BATCH_SIZE = 1
IMAGE_PRINT_PERIOD = 1
USE_DATA_AUGMENTATION = False
DATA_AUGMENTATION_UNIFORM_RANDOM_THRESHOLD = 0.5
USE_COCO_DATASET = False
TEST_MODE = True
DISPLAY = True


# MODEL PATH
MODEL_PATH = 'AUGGI_SEGNET_STATE_2018_11_28_5_7_NEPOCHS_100_TRAINAVGLOSS_13_8_TESTAVGLOSS_13_7.pth'

###
## LOAD DATASET
##
train_dataset = AuggiDetectionDataset(
    mode="TRAIN",
    size=IMG_SHAPE,
    use_coco_dataset=USE_COCO_DATASET,
    use_augmentation=USE_DATA_AUGMENTATION,
    augmentation_uniform_threshold=DATA_AUGMENTATION_UNIFORM_RANDOM_THRESHOLD
)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = AuggiDetectionDataset(
    mode="TEST",
    size=IMG_SHAPE,
    use_coco_dataset=USE_COCO_DATASET,
    use_augmentation=False,
    augmentation_uniform_threshold=DATA_AUGMENTATION_UNIFORM_RANDOM_THRESHOLD
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# CHECK CUDA AVAILABILITY
CUDA_AVAILABLE = torch.cuda.is_available()

##
## INSTANTIATE MODEL
##
model = SegNet(input_nbr=3, label_nbr=N_LABELS)
model.load_from_filename(MODEL_PATH) # load segnet weights
model.eval() # set to eval mode

if CUDA_AVAILABLE: # convert to cuda if needed
    model.cuda()
else:
    model.float()


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def denormalize(img):
    # takes 3 dim tensor and denormalizes

    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406

    return img

def draw_rect(drawcontext, coords, outline=None, width=0):
    x1, y1, x2, y2 = coords
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

boxer = BoundingBox()

THRESHOLD = 210

# start process
for data in train_dataloader:

    img, mask, masked, bristol = data  # retrieve data

    img = img.to(device)  # put data onto available device
    predicted_mask_tensor = model(img)  # forward prop

    # convert mask tensor to np
    mask = predicted_mask_tensor[0]
    mask_as_img = F.to_pil_image(mask)
    mask_np = np.array(mask_as_img)
    mask_np = np.expand_dims(mask_np, axis=2)

    # threshold the binary image
    ret, thresh = cv2.threshold(mask_np, THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = np.expand_dims(thresh, axis=2)
    x_min, y_min, x_max, y_max = boxer.get_box(img, thresh, False)

    # if received bounding box
    if x_min:

        if DISPLAY:

            print(x_min, y_min, x_max, y_max)
            img = img[0]  # grab first tensor
            img = denormalize(img) # denormalize the img

            # convert to pil image
            img = transforms.ToPILImage()(img)

            # draw the bounding box
            draw = ImageDraw.Draw(img)


            draw_rect(draw, [x_min, y_min, x_max, y_max], outline="green", width=3)


            # draw.rectangle([x_min, y_min, x_max, y_max], outline='green')
            img.show()


    # # resize the image to display smaller but undistorted
    # NEW_HEIGHT = 600
    # ratio = NEW_HEIGHT / img_height
    # dim = (int(img_width * ratio) , NEW_HEIGHT)
    # display_img = cv2.resize(display_img, dim)   



















