import data_setup
from models import FSRCNN
import engine
import os
import torch
from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 4
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
SCALE = 4
SEED = 42

# Setup directories
train_dir = "dataset/train/class_0"
test_dir = "dataset/test/class_0"

# Setup target device
device = "cpu" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(SEED)


allowed_extensions = (".png", ".jpeg", ".jpg")

train_dir = "dataset/train/class_0"
test_dir = "dataset/test/class_0"
train_paths = [
    os.path.join(train_dir, i)
    for i in os.listdir(train_dir)
    if i.endswith(allowed_extensions)
]
test_paths = [
    os.path.join(test_dir, i)
    for i in os.listdir(test_dir)
    if i.endswith(allowed_extensions)
]

transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((162, 162)),
        transforms.ToTensor(),
    ]
)

transform_target = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((648, 648)),
        transforms.ToTensor(),
    ]
)


# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir=train_paths,
    test_dir=test_paths,
    transform=transform,
    transform_target=transform_target,
    batch_size=BATCH_SIZE,
)

# Create Model
model = FSRCNN(scale_factor=SCALE)

# Set loss and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

# Start training
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,
)
# # Save the model with help from utils.py
# utils.save_model(
#     model=model_builder_builder,
#     target_dir="models",
#     model_name="05_going_modular_script_mode_tinyvgg_model.pth",
# )
