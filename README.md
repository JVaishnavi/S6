# Assignment 6 ERA course

## MNIST dataset

The MNIST database (Modified National Institute of Standards and Technology database[1]) is a large database of handwritten digits that is commonly used for training various image processing systems.

This repo contains code to identify handwritten numbers.

Input: Image of size 28x28 Output: Class value between 0-9 each value signifying which digit the image shows

## Folder structure

S6.ipynb: notebook to run the model on the dataset
README.md: Details regarding the model

## Model

The objective of this model is to get an accuracy on 99.4%+ using the given conditions 

* 99.4% validation accuracy
* Less than 20k Parameters
* You can use anything from above you want. 
* Less than 20 Epochs
* Have used BN, Dropout

## Model Architecture

### Layer 1:

- Network: CNN
- Kernel: 3x3
- Padding: 0
- Input channel size: 1
- Output channel size: 32
- Input dimension: 28x28x1
- Output dimension: 26x26x32
- Activation function: Relu
- Batch normalisation is performed

### Layer 2:

- Network: CNN
- Kernel: 3x3
- Padding: 0
- Input channel size: 32
- Output channel size: 32
- Input dimension: 26x26x32
- Output dimension: 24x24x32
- Batch normalisation is performed

### Layer 3:

- Network: Max Pooling
- Input channel size: 32
- Output channel size: 32
- Input dimension: 24x24x32
- Output dimension: 12x12x32
- Activation function: Relu
- Dropout: 10% (0.1)

### Layer 4:

- Network: CNN
- Kernel: 1x1
- Padding: 1
- Input channel size: 32
- Output channel size: 8
- Input dimension: 12x12x32
- Output dimension: 12x12x8
- Activation function: Relu
- Batch normalisation is performed

### Layer 5:

- Network: CNN
- Kernel: 5x5 (5x5 is used to reduce the size of the image faster with lesser parameters)
- Padding: 0
- Input channel size: 8
- Output channel size: 32
- Input dimension: 12x12x8
- Output dimension: 8x8x32

### Layer 6:

- Network: Max Pooling
- Input channel size: 32
- Output channel size: 32
- Input dimension: 8x8x32
- Output dimension: 4x4x32
- Activation function: Relu
- Dropout: 10% (0.1)

### Layer 7:

- Network: CNN
- Kernel: 1x1
- Padding: 0
- Input channel size: 32
- Output channel size: 16
- Input dimension: 4x4x32
- Output dimension: 4x4x16

### Layer 8:

- Network: FC
- Input dimension: 4x4x16 (i.e.256)
- Output dimension: 10
- Activation function: Log softmax


### Summary of the network


```python
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 26, 26]             320
           BatchNorm2d-2           [-1, 32, 26, 26]              64
                Conv2d-3           [-1, 32, 24, 24]           9,248
           BatchNorm2d-4           [-1, 32, 24, 24]              64
             MaxPool2d-5           [-1, 32, 12, 12]               0
               Dropout-6           [-1, 32, 12, 12]               0
                Conv2d-7            [-1, 8, 12, 12]             264
           BatchNorm2d-8            [-1, 8, 12, 12]              16
                Conv2d-9             [-1, 32, 8, 8]           6,432
          BatchNorm2d-10             [-1, 32, 8, 8]              64
            MaxPool2d-11             [-1, 32, 4, 4]               0
              Dropout-12             [-1, 32, 4, 4]               0
               Conv2d-13             [-1, 16, 4, 4]             528
          BatchNorm2d-14             [-1, 16, 4, 4]              32
               Linear-15                   [-1, 10]           2,570
    ================================================================
    Total params: 19,602
    Trainable params: 19,602
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.74
    Params size (MB): 0.07
    Estimated Total Size (MB): 0.82
    ----------------------------------------------------------------
    

