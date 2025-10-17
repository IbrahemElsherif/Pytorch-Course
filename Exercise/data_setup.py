# YOUR CODE HERE
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = 0
def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size:int,
    num_workers: int=NUM_WORKERS
):
    """Creates training and testing dataloaders
    
    Takes in a training directory and test directory 
    and turns them into Pytorch datasets then into Pytorch dataloaders
    
    train_dir: path to training directory.
    test_dir: path to testing directory.
    transform: torchvison transforms to perform on training data and testing data.
    batch_size: Number of sampels per batch in each of the dataloaders.
    num_workers: int number for number of workers per dataloader.
    
    Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    where class_names is a list of the target classes.
    Ecample usage:
    train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                            test_dir=path/to/test_dir,
                            transform= some_transform,
                            batch_size=32,
                            num_workers=4)
    
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    # Get class names
    class_names = train_data.classes
    
    # Turn Image into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader, class_names
