import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Introduction to tensors
# Tensors are the fundamental building blocks of PyTorch
# Tensors are similar to NumPy arrays but can be used on GPUs


# creating  tensors
scalar = torch.tensor(7)


# print(scalar.ndim)  # number of dimensions 0
# print(scalar.item())  # get the value of the tensor

# vector
vector = torch.tensor([7, 7])
# print(vector.ndim)  # number of dimensions 1
# print(vector.shape)  # shape of the tensor

# matrix
matrix = torch.tensor([
    [7, 8],
    [9, 10]
])
# print(matrix.ndim)  # number of dimensions 2
# print(matrix.shape)  # shape of the tensor

# tensor
tensor = torch.tensor([
    [[7, 8, 9],
     [9, 10, 0],
     [11, 12, 3]]
])

# print(tensor.ndim)  # number of dimensions 3
# print(tensor.shape)  # shape of the tensor

# creating tensors with random numbers
# Random tensors are important because neural networks start with tensors full of random numbers
# and then gradually adjust them to better represent the data.

random_tensor = torch.rand(3, 4)
# print(random_tensor.ndim)
# print(random_tensor)

#create random tensor with similar shape to an image
random_image_tensor = torch.rand(3, 224, 224)
# print(random_image_tensor.ndim)
# print(random_image_tensor.shape)  # shape of the tensor


# creating a tensor with zeros and ones
zeros_tensor = torch.zeros(3, 4)
# print(zeros_tensor.ndim)
# print(zeros_tensor)  # shape of the tensor

# product =  zeros_tensor * random_tensor
# print(product)

ones_tensor = torch.ones(3, 4)
# print(ones_tensor.ndim)
# print(ones_tensor)  # shape of the tensor


### Creating a range of tensors and tensors like
# range of tensors
tensor_1_10 = torch.arange(1, 10)
one_to_ten = torch.arange(start=1, end=11, step=1)
# print(one_to_ten)  # range of tensors
# print(tensor_1_10)  # range of tensors

# tensor like
tensor_1_10_zeros = torch.zeros_like(input=one_to_ten)
# print(tensor_1_10_zeros)  # tensor like

# dtype of the tensor
# print(ones_tensor.dtype)  # data type of the tensor

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                                dtype=None,
                                device=None, # device is the device on which the tensor is stored cpu or cuda
                                requires_grad=False) #you want to track the gradient of the tensor as it goess through the model
# print(float_32_tensor.dtype)  # data type of the tensor


# Top errors in pytorch
# 1. Tensor shape mismatch
# 2. Tensor data type mismatch
# 3. Tensor device mismatch

float_16_tensor = float_32_tensor.type(torch.float16)
# print(float_16_tensor)  # data type of the tensor

int_32_tensor = float_32_tensor.type(torch.int32)

# print(int_32_tensor * float_32_tensor)

# # get information about the tensor
# print(float_32_tensor.device)  # device of the tensor
# # print(float_32_tensor.shape)  # shape of the tensor
# # print(float_32_tensor.dtype)  # data type of the tensor
# print(float_32_tensor.requires_grad)  # requires grad of the tensor


##Manipulating tensors (tensor operations)
# 1. Addition
# 2. Subtraction
# 3. Multiplication (element-wise)
# 4. Division
# 5. Matrix multiplication
# 6. Reshaping


tensor = torch.tensor([1, 2, 3])
# print(tensor + 10)  # addition [11, 12, 13]
# print(tensor - 10)  # subtraction [-9, -8, -7]
# print(tensor * 10)  # multiplication [10, 20, 30]
# print(tensor / 10)  # division [0.1, 0.2, 0.3]

# # in-built functions
# print(tensor.add(10))  # addition [11, 12, 13]
# print(tensor.sub(10))  # subtraction [-9, -8, -7]
# print(tensor.mul(10))  # multiplication [10, 20, 30]
# print(tensor.div(10))  # division [0.1, 0.2, 0.3]

#matrix multiplication
# 1. Element wise multiplication
# 2. Matrix multiplication (dot product)

## There are two main rules for matrix multiplication:
# 1. the inner dimensions must match:
# (3, 2) @ (3, 2) => wont work
# (2, 3) @ (3, 2) => works
# (2, 3) @ (2, 2) => works

# 2. the resulting matrix will have the shape of the outer dimensions:

tensor_1 = torch.tensor([
    [1, 2],
    [3, 4]
])

tensor_2 = torch.tensor([
    [5, 6],
    [7, 8]
])

tensor =  torch.tensor([1,2,3])
# print(tensor, "x" ,tensor)  # matrix multiplication [14]
# print(f"Equals: {tensor * tensor}")

# matrix multiplication
start = time.time()
value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
# print(value)  # matrix multiplication [14]
end = time.time()
# time taken to multiply the tensors
# print(f"Time taken: {(end - start) * 1000 } milliseconds")

start = time.time()

product = torch.matmul(tensor, tensor)
# print(product)  # matrix multiplication [14]
end = time.time()
# time taken to multiply the tensors
# print(f"Time taken: {(end - start) * 1000 } milliseconds")


#to fix tensor shape errors we can use transpose to manipulate the shape of the tensor
# transpose switches the axes/dimensions of a given tensor
tensor_A = torch.tensor([
    [1, 2],
    [3, 4],
    [5, 6]
])

tensor_B = torch.tensor([
    [7, 8],
    [9, 10],
    [11, 12]
])

# torch.matmul(tensor_A, tensor_B)  # this will give an error because the shapes of the tensors are not compatible
# we need to transpose one of the tensors to make it fit the rules of matrix multiplication
tensor_A_T = tensor_A.T
# print(tensor_A_T)  # shape of the tensor
tensor_A_B = torch.matmul(tensor_A_T, tensor_B)
# print(tensor_A_B)

## Tensor Aggregation
x = torch.arange(1, 100, 10)

# print(x)  # tensor with values from 0 to 100 with step of 10
# #minimum value of the tensor
# print(torch.min(x), x.min())  # minimum value of the tensor

# #maximum value of the tensor
# print(torch.max(x), x.max())  # maximum value of the tensor

# #mean value of the tensor
# print(torch.mean(x.type(torch.float32)), x.type(torch.float32).mean())  # mean value of the tensor

# #sum of the tensor
# print(torch.sum(x), x.sum())  # sum of the tensor

#ArgMin and ArgMax
# ArgMin and ArgMax return the index of the minimum and maximum values of the tensor
min_idx = torch.argmin(x)
# print(min_idx)  # index of the minimum value of the tensor
# print(x[min_idx])

max_idx = torch.argmax(x)
# print(max_idx)  # index of the maximum value of the tensor
# print(x[max_idx])  # maximum value of the tensor

# Reshaping - changing the shape of the tensor
# Viewing - return a view of an input tensor of certain shape but keeps the same memory of the input tensor
# Stacking - combine multiple tensors on top of each other (vertically) or side by side (horizontally)
# Squeezing - remove all single `1` dimensions from a tensor
# Unsqueezing - add a single `1` dimension to a tensor
# Permute - return a view of the input tensor with dimensions permuted in a certain way

x = torch.arange(1., 10.)
# print(x, x.shape)

#Reshaping - Reshape has to be compatible with the number of elements in the tensor
x_reshaped = x.reshape(1, 9)
# print(x_reshaped, x_reshaped.shape)  # reshaped tensor

# View
z = x.view(1, 9)
# print(z, z.shape)  # view of the tensor
z[:, 0] = 5
# print(z, x)  # view of the tensor

# Stacking
x_stack = torch.stack([x, x, x, x], dim=0)
# print(x_stack, x_stack.shape)  # stacked tensor

z = torch.tensor([0, 1, 2, 3, 4, 5])
print(z[1:5:2])

# Squeezing - removes all single `1` dimensions from a tensor
print(x_reshaped)
x_squeezed = x_reshaped.squeeze()
print(x_squeezed, x_squeezed.shape)  # squeezed tensor

# Unsqueezing - adds a single `1` dimension to a tensor
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(x_unsqueezed, x_unsqueezed.shape)  # unsqueezed tensor

x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print(x_unsqueezed, x_unsqueezed.shape)  # unsqueezed tensor

# Permute - return a view of the input tensor with dimensions permuted in a certain way
x_original = torch.rand(224, 224, 3) # [height, width, color channels]
print(x_original.shape)  # original tensor

x_permuted = x_original.permute(2, 0, 1) # [color channels, height, width]
print(x_permuted.shape)  # permuted tensor

## Indexing tensors
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)  # original tensor

print(x[0][0][0]) 
print(x[0, 1, 1])  # indexing tensor

# you can use : to select all elements in a dimension
print(x[:, 0])  # select all elements in the first dimension and first row

# get all values in the 0th dimension and 1st dimension but only index 1 of 2nd dimensions
print(x[:, :, 1])

# get all values of the 0th dimension but only the 1st index of 1st and 2nd dimension
print(x[:, 1, 1])
print(x[:, :, 2])


##Runing tensors and PyTorch on the GPU
# PyTorch can run on the GPU for faster computation
# to check if GPU is available
def check_gpu():
    if torch.backends.mps.is_available():
        print("GPU is available")
        device = "mps" # Metal Performance Shaders for macOS
    else:
        print("GPU is not available, using CPU")
        device = "cpu"
    return device
device = check_gpu()

# 1. Easiest  way is to use Google colab
# 2. If you have a GPU, you can install PyTorch with CUDA support
# 3. Use cloud computing services like AWS, GCP, Azure, etc.

## For 2 and 3, you can follow the instructions on the PyTorch website to install the correct version of PyTorch with CUDA support.

#setup device agnostic code

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)  # device of the tensor

#count number of devices
# print(torch.mps.device_count())  # number of devices available
#check CUDA semantics in docs

tensor = torch.tensor([1, 2, 3], device=device)
print(tensor)  # tensor on the device

tensor_back_on_cpu = tensor.to("cpu").numpy()  # move tensor back to CPU
print(tensor_back_on_cpu)  # tensor on the CPU
print(torch.__version__)