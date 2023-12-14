import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from torchvision import transforms
import matplotlib.pyplot as plt
from model import UNet

class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, target_size=(768, 768)):
        self.data = pd.read_csv(path_csv_file)
        self.path_testdataset = path_testdataset
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path_testdataset, self.data.iloc[idx, self.data.columns.get_loc('ImageId')])
        # Load the image using PIL
        image = Image.open(img_name).convert('RGB')
        image = image.resize(self.target_size)

        mask_str = self.data.iloc[idx, self.data.columns.get_loc('EncodedPixels')]

        # Apply transformations to convert to tensors
        transform = transforms.Compose([
            transforms.ToTensor(), ])

        image = transform(image)

        sample = {'image': image}

        return sample

path_csv_file = 'C:\\Users\\Микола\\Downloads\\airbus-ship-detection\\sample_submission_v2.csv'
path_testdataset = 'C:\\Users\\Микола\\Downloads\\airbus-ship-detection\\test_v2'

test_dataset = TestDataset(csv_file=path_csv_file, root_dir=path_testdataset)
test_loader = DataLoader(dataset=test_dataset, shuffle=False)

# Move the model to GPU if available;
model = UNet()
device = torch.device("cuda:0")
assert torch.cuda.is_available(), "CUDA is not available on this machine."
model = model.to(device)
model.eval()

with torch.no_grad():
    example_counter = 0

    for batch in test_loader:
        images = batch['image'].to(device)

        # Forward pass
        outputs = model(images)

        # Calculate Dice Score (assuming binary segmentation)
        predictions = torch.sigmoid(outputs)

        # Select a random sample from the batch for visualization
        idx = random.randint(0, images.size(0) - 1)
        input_image = images[idx].cpu().squeeze().numpy()
        predicted_mask = predictions[idx].cpu().squeeze().numpy()

        # Binarize the predicted mask
        binary_predicted_mask = (predicted_mask > 0).astype(float)

        # Plotting
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow(input_image[i], cmap='gray')
            plt.title(f"Input Image {i + 1}")

        plt.subplot(1, 2, 2)
        plt.imshow(binary_predicted_mask, cmap='gray')
        plt.title("Predicted Mask")

        plt.show()

        example_counter += 1
        if example_counter == 10:
            break












