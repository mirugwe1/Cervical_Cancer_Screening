# Deep Learning CNN Cervical Cancer Screening Algorithm

## Introduction

This repository contains the implementation of deep learning Convolutional Neural Network (CNN) algorithms for cervical cancer screening. The algorithm aims to assist in the early detection and classification of cervical cancer from digital cervical images. We trained one from scratch, and then we also tried transfer learning by training on top of VGG16 and ResNet50, but this never improved the performance. 

Cervical cancer is a significant global health issue, particularly in low-resource settings. Early detection and accurate classification of cervical abnormalities are crucial for effective treatment and improved patient outcomes. This repository presents a deep learning algorithm that utilizes a CNN to analyze digital cervical images and identify potential cancerous or pre-cancerous regions.


## Table of Contents

[Introduction](https://github.com/mirugwe1/Cervical_Cancer_Screening#introduction)\
[System Requirements](https://github.com/mirugwe1/Cervical_Cancer_Screening#system-requirements)\
[Required Dependencies](https://github.com/mirugwe1/Cervical_Cancer_Screening/tree/master#required-dependencies)\
[Dataset](https://github.com/mirugwe1/Cervical_Cancer_Screening#dataset)
* [Sample Images](https://github.com/mirugwe1/Cervical_Cancer_Screening#sample-images)\
[Model Architecture](https://github.com/mirugwe1/Cervical_Cancer_Screening#model-architecture)\
[Training](https://github.com/mirugwe1/Cervical_Cancer_Screening#training)\
[Evaluation](https://github.com/mirugwe1/Cervical_Cancer_Screening#evaluation)



## System Requirements
The algorithm was developed in Python, utilizing;
 - TensorFlow 2.11.0 
 - Keras 2.12.0  
 - MSI GL75 Leopard 10SFR laptop 
 -  8GB NVIDIA RTX 2070 GDDR6 Graphical Processing Unit (GPU) 
 -  CUDA 12.1 
 - cuDNN SDK 8.7.0 
 
 The [CUDA 12.1 ](https://developer.download.nvidia.com/compute/cuda/12.1.1/network_installers/cuda_12.1.1_windows_network.exe) and [cuDNN 8.7.0](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse870-118) can be downloaded  from the official [NVIDIA Website](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local).


## Required Dependencies

```
NumPy
Pandas
Tensorflow
OpenCV
```


## Dataset
The dataset used for training and evaluation can be obtained from [Kaggle](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data). It consists of a collection of digital cervical images labelled with corresponding class annotations.

### Sample Images
![](https://github.com/mirugwe1/Cervical_Cancer_Screening/blob/master/visuals/sample.jpg)


## Model Architecture

The detailed architecture of the CNN is as follows:
![](https://github.com/mirugwe1/Cervical_Cancer_Screening/blob/master/visuals/Architecture.png)



## Training
To train the deep learning model, follow these steps:

Preprocess the dataset to prepare the images and labels for training.
Configure the training parameters such as batch size, learning rate, and number of epochs in the training script.
Run the training script:
Copy code
python train.py
Monitor the training progress and wait until the training completes.


## Evaluation
Once the model is trained, you can evaluate its performance on a separate test set. To evaluate the model, follow these steps:

Prepare the test set with appropriate annotations.
Run the evaluation script:
Copy code
python evaluate.py
The evaluation script will load the trained model and compute various evaluation metrics such as accuracy, precision, recall, and F1 score.
Usage
After training and evaluating the model, you can utilize it for cervical cancer screening on new, unseen images. Here's how to use the model for inference:

Load the trained model weights:

python
Copy code
from model import CervicalCancerModel

model = CervicalCancerModel()
model.load_weights('path/to/weights.h5')
Preprocess the input image(s) to match the model's input requirements.

Feed the preprocessed image(s) to the model for prediction:

python
Copy code
predictions = model.predict(image)
The model will output predictions, indicating the likelihood of cervical abnormalities.

Contributing
Contributions to this project are welcome. To contribute, follow these steps:

Fork the repository.
Create a new branch.
Make your changes and commit them.
Push your changes to your forked repository.
Submit a pull request, describing your changes and contributions.