# Numpy is  a popular scientifc python numerical computing library
# and because of this, pytorch has functionality to interact with it
# eg: data is in numpy and want it in pytorch tensor -> torch.from_numpy(data)
#eg: data is in pytorch tensor and want it in numpy -> torch.Tensor.numpy()

import torch
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) #when you convert it uses numpy default dtype which is float64
#you can convert it to float32
tensor = tensor.type(torch.float32) #convert to float32
print(array, tensor)

#tensor to numpy array
tensor =  torch.ones(7)
numpy_tensor = tensor.numpy() #convert to numpy array
print(tensor, numpy_tensor.dtype)

#change the tensor
tensor = tensor + 1
print(tensor, numpy_tensor) # numpy array does not change because they do not share memory