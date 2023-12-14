

# Airbus Ship Detection Challenge

**Ship Detection and Segmentation with U-Net in PyTorch**

### Table of Contents
- [Task Overview](#task-overview)
- [Project Structure](#project-structure)
- [Preparation](#preparation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Visualization](#visualization)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Lessons Learned](#lessons-learned)

## Task overview
This project focuses on ship detection and segmentation, aiming to accurately identify and delineate ship boundaries in images.

## Project structure
- `model.py`: Contains the U-Net model implementation.
- `model_training.py`: Script for training the model.
- `model_inference.py`: Script for making predictions using the trained model.
- `data_analysis.ipynb`: Jupyter notebook with data preprocessing.
- `trained_model.pth`: Saved parameters and weights of your trained U-Net model. 
- `rough_draft.ipynb`: A draft, with different stages and mistakes.
- `readme.md`: The current file providing an overview of the project.
- `requirements.txt`: List of packages required for running the project.

## Preparation
1. **To train locally, on your own host machine, it's required to download kaggle competition [testset](https://www.kaggle.com/c/airbus-ship-detection/data)  "sample_submission_v2.csv" and "test_v2".**
2. **Initialize your own `.env` file, create a Python virtual environment, and install necessary packages:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e . --no-deps
pip install -r requirements.txt
```
## Dataset
Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

## Model architecture
**Downsampling Path:**
Four convolutional blocks, each consisting of two convolutional layers with ReLU activation functions and max-pooling. The purpose of these blocks is to progressively reduce the spatial resolution and extract hierarchical features from the input image.

**Upsampling Path:**
Four upsampling blocks that mirror the downsampling path. These blocks use transpose convolutions (also known as deconvolutions) to upsample the feature maps. Skip connections are incorporated by concatenating feature maps from the corresponding downsampling layers to preserve detailed information during the upsampling process.

**Convolutional Block Function:**
Defines a standard convolutional block used in both the downsampling and upsampling paths. It consists of two convolutional layers with ReLU activation functions and a max-pooling operation in the downsampling path.

**Upconvolutional Block Function:**
Defines an upsampling block used in the upsampling path. It includes a transpose convolution for upsampling, ReLU activation, and an additional convolutional layer.

**Final Convolutional Layer:**
This layer produces the segmentation output with a single channel. It uses a 1x1 convolution to map the learned features to the desired output channel, which is typically used for binary segmentation tasks.

**Forward Method:**
In essence, the forward method serves as the orchestrator of a meticulously choreographed dance, guiding the input data through a symphony of downsampling, upsampling, and convolutional operations, ultimately culminating in the generation of a meaningful and nuanced segmentation output.

## Training
1. Preprocess the dataset.
2. Define and instantiate the U-Net model.
3. Specify loss function and optimizer.
4. Train the model using the training dataset.
5. Save the trained model weights for future use.

## Visualization
Visual representations of model predictions on sample images to demonstrate its efficacy.

## Results
1. Data processing was carried out.
2. The model structure was created.
3. Data visualization is performed.
4. Train the model using the training dataset.
5. Predicted mask shows poor results, but [future improvements](#future-improvements) indicate further steps for improvement.
6. Saved the weights of the trained model for further use.

## Future improvements
1. Use Albumentations transforms.
2. Implement the "early stopping" method during training.
3. Train contains photos of the same area/ships made with 256px shifts. In order to improve the model and avoid leakage, these photos should be put into the same fold.
4. Fine-tuning the model.
5. Explore other architectures.
6. Incorporate additional data for improved generalization.

## Lessons learned
1. Visualize.
2. Save network's weights.
3. Use P100 Tesla free on Kaggle, not your capacity, if it's better.
4. Don't be afraid to make mistakes and take some advice.
5. Keep going no matter how terrible your results; believe in yourself.
6. Look at the big picture, but don't forget the little things.

