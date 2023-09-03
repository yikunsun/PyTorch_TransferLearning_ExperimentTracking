# **Transfer Learning**
While looking at different models that can be used for image classification tasks, I found that **PyTorch's TorchVision** library offers **pre-trained weights** for every provided model architecture. This led me to wonder if a pre-trained model can achieve better accuracy than [the TinyVGG model](https://github.com/yikunsun/CNN_FashionMNIST) I built, so I looked into the process of using TorchVision's pre-trained model and encountered a concept called **transfer learning**. 

## **What is transfer learning?**
Transfer learning is the process of taking a pre-trained model that has been proven to work well on a similar task and customizing it to our use case. In my case, I want to use a pre-trained model to classify images in the CIFAR10 dataset.

## **Benefits of transfer learning?**
- Transfer learning saves time and effort in developing a model from scratch.
- It also helps achieve better model performance especially when we don't have large amounts of training data.


---
# **Experiment Tracking**
Since using PyTorch, I have been using **Python dictionaries to track model results**. The results can be easily compared using dictionaries when I have only 1 or 2 models. However, for this project, I wanted to run several models and compare their results. Using dictionaries is not well suited for this task.

Therefore, I looked into tools for tracking results that can help me easily compare results and identify the best-performing model. The official name for tracking model performance is **experiment tracking**, *the process of organizing, recording, and analyzing the results of machine learning experiments.* Experiment tracking helps us **identify the best-performing model out of tens or even hundreds of models** we may try.

## **Tools for tracking experiments**
There are many tools for tracking experiments, such as MLFlow, TensorBoard, Weights & Biases. Due to TensorBoard's tight integration with PyTorch, I will use TensorBoard for experiment tracking. With TensorBoard, I can visualize the results of machine learning experiments.



---

# **Table of Contents**
## **Section 1: Get CIFAR10 dataset**
## **Section 2: TinyVGG for image classification**
- 2.1 Prepare DataLoaders
- 2.2 Define functions for training and testing loops
- 2.3 Build the TinyVGG model
- 2.4 Train and test the TinyVGG model

## **Section 3: Transfer learning**
- 3.1 Prepare DataLoaders
- 3.2 Set up a pre-trained EfficientNet B0 model 
- 3.3 Get a summary of the model with `torchinfo.summary()`
- 3.4 Freeze base layers of the pre-trained model and customize the output layer
- 3.5 Train and test the EfficientNet B0 model

## **Section 4: Experiment tracking**
- 4.1 Track only 1 experiment
    - 4.1.1 Set up instance of `SummaryWriter()` and track results of EfficientNet B0 model
    - 4.1.2 View model's results in TensorBoard
- 4.2 Create a function to build `SummaryWriter()` instances
- 4.3 Track multiple experiments
    - 4.3.1 Decide experiments to run and track
    - 4.3.2 Create Datasets and Prepare DataLoaders
    - 4.3.3 Define pre-trained models
    - 4.3.4 Create experiments and train models
    - 4.3.5 View results of 8 experiments in TensorBoard
    - 4.3.6 Make predictions with the best-performing model



---
# **Summary**
**Transfer learning** is powerful especially when we don't have a large amount of training data. The **EfficientNet B0 model** with pre-trained weights offered by TorchVision performs much better than the [**TinyVGG model**](https://github.com/yikunsun/CNN_FashionMNIST) even though the **EfficientNet B0 model** was trained with only 500 images while the **TinyVGG model** was trained with 10000 images. Next time when I take on an image classification task, besides building a model myself, I will use a few pre-trained models, compare their performances, and select a best-performing model.

**TensorBoard** is a great tool for **experiment tracking** since I can visualize the performance of different models. It also enables tracking multiple metrics at the same time, such as loss and accuracy. 


---

# **Reference**
- Daniel Bourke's Learn PyTorch for Deep Learning: Zero to Mastery: https://www.learnpytorch.io/
