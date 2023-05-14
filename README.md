
# Project Title

Semantic segmentation of CXR images from pneumonia dataset using U-NET architecture.


## Introduction

Developed by Olaf Ronneberger et al. for Biomedical Image Segmentation in 2015 at the University of Freiburg, Germany, U-Net is an architecture for semantic segmentation. Semantic segmentation is a computer vision task that involves assigning a label to each pixel in an image, enabling pixel-level understanding. This project aims to implement semantic segmentation on CXR images using the U-Net architecture, enabling accurate identification and segmentation of pneumonia regions.
## Dataset
The dataset for this project consist of Chest X-ray images and the masked version for training. Along with it we have the test images (or cross validation images) to further test our algorithm. This dataset is taken from an online repository of the dataset on kaggle. You can download this dataset using this [url]('https://www.kaggle.com/datasets/newra008/chest-segmentation-image?resource=download'). This dataset consist of images of size 512 X 512.
## Installation
To run this implementation, follow these steps:
1. Clone this repository: 'git clone https://github.com/Narayan-21/Semantic-segmentation-of-CXR-images-from-pneumonia-dataset-using-U-NET-architecture.git'
2. Install the required dependencies: 'pip install -r requirements.txt'
## Usage
1. Navigate to the project directory: 'cd your-repo'
2. You can change the hyperparameters that are otherwise predecided in the 'train.py' file.
3. Run the UNET model using following command on your CLI: 
```python
python train.py
```
## Model Architecture

UNET is a U-shaped encoder-decoder network architecture, which consists of four encoder blocks and four decoder blocks that are connected via a bridge. The architecture of UNET consists of contracting path to capture the context and a symmetric expanding path that enables precise localization. The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels. Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.
The below figure visualizes the UNET architecture:

![u-net-architecture](https://github.com/Narayan-21/Semantic-segmentation-of-CXR-images-from-pneumonia-dataset-using-U-NET-architecture/assets/64371700/079bb7fe-5a65-43c0-95c8-a6e6aeab6ac0)
Figure-1: The UNET architecture.

## Training Process
The UNET was trained for 3 epochs with a batch size of 16 due to computation constraints. The learning rate was set to 0.0001, and Adam optimizer and Binary cross-entropy with logits loss which combines a Sigmoid layer and the BCELoss in one single class was used.
## Results
After running the 'train.py' file, the results automatically will be saved in the 'saved_images' folder. The accuracy for each epoch can be seen on the CLI while running 'train.py' file. We used the dice score to check the accuracy of our predictions which is a spatial overlap index and a reproducibility validation metric. 
Below are the two images comparing the original masked version of the CXR image and the generated masked version using UNET architecture.

![0](https://github.com/Narayan-21/Semantic-segmentation-of-CXR-images-from-pneumonia-dataset-using-U-NET-architecture/assets/64371700/4df619dc-455f-4828-b734-d24e4e871d1a)
Figure-2: The original masked version of the CXR images.


![pred_0](https://github.com/Narayan-21/Semantic-segmentation-of-CXR-images-from-pneumonia-dataset-using-U-NET-architecture/assets/64371700/4e809189-f211-451f-89b1-a71db6b69151)
Figure-3: The generated masked images.

## Usage
1. Navigate to the project directory: 'cd your-repo'
2. You can change the hyperparameters that are otherwise predecided in the 'train.py' file.
3. Run the UNET model using following command on your CLI: 
```python
python train.py
```
## License
This Project is licensed under [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
## Acknowledgements

 - [The PyTorch team for their excellent deep learning library.](https://pytorch.org/)
 - [The authors of the original UNET paper: Olaf Ronneberger, Philipp Fischer, Thomas Brox.](https://arxiv.org/pdf/1505.04597.pdf)
 - [The author of the dataset: Neelkant Newra](https://www.kaggle.com/datasets/newra008/chest-segmentation-image?resource=download)
## Contact Information
For any questions or feedback, please contact me at naaidjan.19@gmail.com.
