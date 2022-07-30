import pathlib
import torch
import albumentations
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from skimage.transform import resize
from inference import predict
from customdatasets import SegmentationDataSet1
from transformations import re_normalize
from skimage.io import imread
from skimage.transform import resize
import os
from customdatasets import SegmentationDataSet3
from transformations import (
    ComposeDouble,
    normalize_01,
    FunctionWrapperDouble,
    create_dense_target,
    AlbuSeg3d,
)
from unet import UNet
from trainer import Trainer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
PART 1: Data
"""

# root directory
root = pathlib.Path.cwd() / "Microtubules3D"


def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# input and target files
inputs = get_filenames_of_path(root / "Input")
targets = get_filenames_of_path(root / "Target")

# training transformations and augmentations
# example how to properly resize and use AlbuSeg3d
# please note that the input is grayscale and the channel dimension of size 1 is added
# also note that the AlbuSeg3d currently only works with input that does not have a C dim!
transforms_training = ComposeDouble(
    [
        # FunctionWrapperDouble(resize, input=True, target=False, output_shape=(16, 100, 100)),
        # FunctionWrapperDouble(resize, input=False, target=True, output_shape=(16, 100, 100), order=0, anti_aliasing=False, preserve_range=True),
        # AlbuSeg3d(albumentations.HorizontalFlip(p=0.5)),
        # AlbuSeg3d(albumentations.VerticalFlip(p=0.5)),
        # AlbuSeg3d(albumentations.Rotate(p=0.5)),
        AlbuSeg3d(albumentations.RandomRotate90(p=0.5)),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(np.expand_dims, axis=0),
        # RandomFlip(ndim_spatial=3),
        FunctionWrapperDouble(normalize_01),
    ]
)

# random seed
random_seed = 42

# dataset training
dataset_train = SegmentationDataSet3(
    inputs=inputs,
    targets=targets,
    transform=transforms_training,
    use_cache=False,
    pre_transform=None,
)

x, y = dataset_train[1]
print(x.shape)
print(x.min(), x.max())
print(y.shape)
print(torch.unique(y))

# dataloader training
dataloader_training = DataLoader(
    dataset=dataset_train,
    batch_size=1,
    # batch_size of 2 won't work because the depth dimension is different between the 2 samples
    shuffle=True,
)

dataloader_validation = DataLoader(
    dataset=dataset_train,
    batch_size=1,
    # batch_size of 2 won't work because the depth dimension is different between the 2 samples
    shuffle=True,
)

batch = next(iter(dataloader_training))
x, y = batch
print("x.shape:", x.shape)
print(x.min(), x.max())
print("y.shape:", y.shape)
print(torch.unique(y))

"""
PART 2: Model
"""

num_classes = 3
model = UNet(
    in_channels=1,
    attention=True,
    out_channels=num_classes,
    n_blocks=3,
    start_filts=32,
    activation="relu",
    normalization="batch",
    conv_mode="same",
    dim=3,
)

x = torch.randn(size=(1, 1, 8, 200, 200), dtype=torch.float32)
with torch.no_grad():
    out = model(x)

print(f"Out: {out.shape}")

from torchsummary import summary
summary = summary(model, (1, 8, 200, 200))

"""
PART 3: TRAINING
"""

# criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# trainer
trainer = Trainer(
    model=model,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    training_dataloader=dataloader_training,
    validation_dataloader=dataloader_validation,
    lr_scheduler=None,
    epochs=2,
    epoch=0,
    notebook=True,
)

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()

from visual import plot_training

fig = plot_training(
    training_losses,
    validation_losses,
    lr_rates,
    gaussian=True,
    sigma=1,
    figsize=(10, 4),
)

sys.exit()
"""
PART 4: EVALUATION
"""
# root directory
root = pathlib.Path.cwd() / "Carvana" / "Test"


def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# input and target files
images_names = get_filenames_of_path(root / "Input")
targets_names = get_filenames_of_path(root / "Target")

# read images and store them in memory
images = [imread(img_name) for img_name in images_names]
targets = [imread(tar_name) for tar_name in targets_names]

# Resize images and targets
images_res = [resize(img, (128, 128, 3)) for img in images]
resize_kwargs = {"order": 0, "anti_aliasing": False, "preserve_range": True}
targets_res = [resize(tar, (128, 128), **resize_kwargs) for tar in targets]

epoch_num = 2
model_name = "carvana_model_epoch_" + str(epoch_num) + ".pt"
exp_dir = pathlib.Path(os.path.expanduser("~") + '/pytorch_exp')
model_weights = torch.load(exp_dir / model_name, map_location=device)
print(f"Loading from {exp_dir / model_name}")
model.load_state_dict(model_weights)

# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img


# postprocess function
def postprocess(img: torch.tensor):
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    img = re_normalize(img)  # scale it to the range [0-255]
    return img


# predict the segmentation maps
output = [predict(img, model, preprocess, postprocess, device) for img in images_res]



