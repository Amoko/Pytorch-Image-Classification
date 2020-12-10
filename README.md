# Pytorch-Image-Classification
This a pytorch demo for image classification with some useful features:

- A method to save every checkpoint by far with the highest accuracy 
- A loss trick  that  allow you to do some rule embedding
- Multi-gpu training with DistributedDataParallel
- Automatic mixed precision training
- Some scripts to do image clean and convert



And the packages version that I use:

```shell
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchsummary
pip install efficientnet_pytorch
```
