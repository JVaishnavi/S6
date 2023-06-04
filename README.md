# S6

```python
#!pip install torchsummary
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
