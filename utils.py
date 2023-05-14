import torch
import torchvision
from dataset import ChestDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print('=> Saving Checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(train_dir, train_maskdir, test_dir, test_maskdir, batch_size, train_transform, test_transform, num_workers=4, pin_memory=True):
    train_ds = ChestDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    test_ds = ChestDataset(image_dir=test_dir, mask_dir=test_maskdir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, test_loader

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            y = y.unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds> 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds+y).sum() + 1e-8)
    print(
        f"Got {num_correct}/ {num_pixels} with accuracy {num_correct/num_pixels*100:.2f}"
    )
    print(f'Dice Score: {dice_score/len(loader)}')
    model.train()

def save_predictions_as_images(loader, model, folder='saved_images/'):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds>0.5).float()
        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.png'
        )
        torchvision.utils.save_image(y.unsqueeze(1), f'{folder}{idx}.png')

    model.train()