from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from prep_data import MyDataset


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    transform_target: transforms.Compose,
    batch_size: int,
):
    train_data = MyDataset(
        train_dir, transform=transform, transform_target=transform_target
    )
    test_data = MyDataset(
        test_dir, transform=transform, transform_target=transform_target
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_dataloader, test_dataloader
