import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

'''The CustomDataset class is responsible for loading and preprocessing the dataset. 
The dataset is dynamically balanced between images with and without ships, ensuring a specified ship ratio. '''

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=(768, 768), ship_ratio=0.5):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_size = target_size

        # Filter images without ships
        images_with_ship = self.data[self.data['EncodedPixels'].notnull()]
        num_images_with_ship = len(images_with_ship)

        # Calculate the number of images without ships dynamically
        num_images_without_ship = int(num_images_with_ship / ship_ratio) - num_images_with_ship

        # Ensure at least one image without ships is included
        num_images_without_ship = max(num_images_without_ship, 1)

        # Sample images without ships without replacement
        images_without_ship = self.data[self.data['EncodedPixels'].isnull()].sample(
            n=num_images_without_ship, random_state=42, replace=False)

        # Concatenate the two subsets
        self.data = pd.concat([images_with_ship, images_without_ship], ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, self.data.columns.get_loc('ImageId')])
        # Load the image using PIL
        image = Image.open(img_name).convert('RGB')

        # Resize the image to the target size
        image = image.resize(self.target_size)

        mask_str = self.data.iloc[idx, self.data.columns.get_loc('EncodedPixels')]

        # Convert the mask string to a NumPy array
        mask = rle_decode(mask_str, target_size=self.target_size)

        # Convert the mask to a PIL image
        mask = Image.fromarray(mask)

        # Apply transformations to convert to tensors
        transform = transforms.Compose([
            transforms.ToTensor(), ])

        image = transform(image)
        mask = transform(mask)

        sample = {'image': image, 'mask': mask}

        return sample


'''The rle_decode function decodes Run-Length Encoding (RLE)-encoded masks into NumPy arrays. 
It handles missing masks and generates zero-filled arrays for those instances.'''

def rle_decode(mask_rle, target_size=(768, 768)):
    if pd.isna(mask_rle):
        return np.zeros(target_size, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(np.prod(target_size), dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(target_size, order='F')

'''The UNet class defines the architecture of the U-Net model. It consists of a downsampling path, an upsampling path with skip connections, 
and a final convolutional layer. The downsampling path includes four convolutional blocks (conv1 to conv4), each followed by a ReLU activation function and max-pooling. 
The upsampling path (upconv4 to upconv1) uses transpose convolutions and concatenates skip connections from the corresponding downsampling layers. 
The final layer (final_conv) produces the segmentation output with a single channel.'''

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Downsampling path
        self.conv1 = self.conv_block(3, 8)
        self.conv2 = self.conv_block(8, 16)
        self.conv3 = self.conv_block(16, 32)
        self.conv4 = self.conv_block(32, 64)

        # Upsampling path
        self.upconv4 = self.upconv_block(64, 32)
        self.upconv3 = self.upconv_block(32, 16, skip_channels=32)
        self.upconv2 = self.upconv_block(16, 8, skip_channels=16)
        self.upconv1 = self.upconv_block(8, 8, skip_channels=8)

        # Final convolutional layer
        self.final_conv = nn.Conv2d(8, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def upconv_block(self, in_channels, out_channels, skip_channels=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels + skip_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        # Downsampling
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Upsampling with skip connections
        x = self.upconv4(x4)
        x = self.upconv3(torch.cat([x, x3], dim=1))
        x = self.upconv2(torch.cat([x, x2], dim=1))
        x = self.upconv1(torch.cat([x, x1], dim=1))

        # Final convolutional layer
        x = self.final_conv(x)

        return x

model = UNet()
#print(model)

