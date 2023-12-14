import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
from model import UNet, CustomDataset
from tqdm import tqdm

'''
Parametrs of model
ship_ratio = 0.5 (50% images with ships)
batch_size = 4
lr = 0.001
num_epochs = 10
'''
# Define the Dice Score function
def dice_score(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


# Define the Dice Loss function
def dice_loss(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

if __name__ == "__main__":
    csv_file = "C:\\Users\\Микола\\Downloads\\airbus-ship-detection\\train_ship_segmentations_v2.csv"
    root_dir = "C:\\Users\\Микола\\Downloads\\airbus-ship-detection\\train_v2"

    custom_dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir)

    # Calculate the index for splitting into train and validation sets
    split_index = int(0.8 * len(custom_dataset))

    # Create training and validation datasets
    train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [split_index, len(custom_dataset) - split_index])

    # Create DataLoaders for training and validation datasets
    batch_size = 4
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# Move the model to GPU if available;
model = UNet()
device = torch.device("cuda:0")
assert torch.cuda.is_available(), "CUDA is not available on this machine."
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # BCE loss for pixel-wise accuracy
dice_criterion = dice_loss  # Dice loss for object localization
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Forward pass
        outputs = model(images)

        # Calculate BCE loss
        bce_loss = criterion(outputs, masks)

        # Calculate Dice Loss
        dice_loss = dice_criterion(torch.sigmoid(outputs), masks)

        loss = bce_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate Dice Score (assuming binary segmentation)
        predictions = torch.sigmoid(outputs)
        dice = dice_score(predictions > 0.5, masks)

        total_loss += loss.item()
        total_dice += dice

    average_loss = total_loss / len(train_loader)
    average_dice = total_dice / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}, Average Dice Score: {average_dice:.4f}")

torch.save(model.state_dict(), 'trained_model.pth')

model.eval()
with torch.no_grad():
    example_counter = 0

    for batch in val_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Forward pass
        outputs = model(images)

        # Calculate Dice Score (assuming binary segmentation)
        predictions = torch.sigmoid(outputs)

        # Select a random sample from the batch for visualization
        idx = random.randint(0, images.size(0) - 1)
        input_image = images[idx].cpu().squeeze().numpy()
        true_mask = masks[idx].cpu().squeeze().numpy()
        predicted_mask = predictions[idx].cpu().squeeze().numpy()

        # Binarize the predicted mask
        binary_predicted_mask = (predicted_mask > 0).astype(float)

        # Plotting
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(input_image[i], cmap='gray')
            plt.title(f"Input Image")

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title("True Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(binary_predicted_mask, cmap='gray')
        plt.title("Predicted Mask")

        plt.show()

        example_counter += 1
        if example_counter == 3:
            break

        # Validation loop
model.eval()
total_dice_val = 0.0

with torch.no_grad():
    for batch_val in tqdm(val_loader, desc="Validation"):
        images_val = batch_val['image'].to(device)
        masks_val = batch_val['mask'].to(device)

        # Forward pass
        outputs_val = model(images_val)

        # Calculate Dice Score (assuming binary segmentation)
        predictions_val = torch.sigmoid(outputs_val)
        dice_val = dice_score(predictions_val > 0.5, masks_val)

        total_dice_val += dice_val

average_dice_val = total_dice_val / len(val_loader)
print(f"Average Dice Score on Validation Set: {average_dice_val:.4f}")