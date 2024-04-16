Hi guys!

Please go through the requirements.txt file and pip install all the modules to the specific versions mentioned..
Tensorflow is running on 2.12.0 so Cuda version must be 11.8 and cuDNN must be 8.9
Please also ensure cuda is on your path otherwise tensorflow will forcefully use your CPU which is inefficient given the size of the dataset. 
