import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from UNET import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_images
)

learning_rate = 1e-04
batch_size = 16
num_epochs = 3
num_workers = 2
image_height = 160
image_width = 240
pin_memory = True
load_model = False
train_img_dir = 'dataset/image/'
train_mask_dir = 'dataset/mask/'
test_img_dir = 'dataset/test_image/'
test_mask_dir = 'dataset/test_mask/'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        targets = targets.float().unsqueeze(1)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm_loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    test_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                        mean=[0.0,0.0,0.0],
                        std=[1.0,1.0,1.0],
                        max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=1)
    loss_fn = nn.BCEWithLogitsLoss() #Since we are not doing Sigmoid on the output of the model.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = get_loaders(
        train_img_dir,
        train_mask_dir,
        test_img_dir,
        test_mask_dir,
        batch_size,
        train_transform,
        test_transforms,
        num_workers,
        pin_memory
    )

    if load_model:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)

    check_accuracy(test_loader, model)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
        check_accuracy(test_loader, model)
        save_predictions_as_images(test_loader, model, folder='saved_images/')


if __name__ == "__main__":
    main()
