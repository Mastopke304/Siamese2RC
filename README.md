# README

This is a deep learning model that to calculate the similarity between a pair of images from MNIST dataset via a Siamese network based on CNN and reservoir computing. 
The code is based on Python 3.8.8 and Pytorch 1.9.1

## Train

To run the training process, using following code:

```python
python main.py
```

Parameters:

--**device**: str; The device you want to run the program. default=cpu

--**seed**: int; Random seed, default=304, set the seed of training the same as testing to obtain the same performance.

--**noise**: int; If 1, gaussian noise will be add to both train data and test data, default=False

--**sigma**: int; The $\sigma$ of gaussian noise, $\sigma^2_T=\frac{\sigma^2}{255^2}$, cause the image is normalized into [0,1], default=30

--**filters**: list of length 4; The number of filters of CNN. default=[32,64,96,96]

--**num_vec**: int; How many vectors does the feature maps be divided to, default=16

--**outSize**: Output size of reservoir for single input, default=1 to obtain the similarity.

--**resSize**: int; The size of reservoir, default=2000

--**alpha**: float; Leaky rate. default=0.3

--**trainLen**: int; The number of data for training

--**batch_size_train**: int; batch size of train dataset.

--**batch_size_test**: int; batch size of test dataset.

--**save_dir**: str; The file to save the weights. default='./result/train/weights/'

--**img_save_dir**: str; The file to save the images. default='./result/train/'

--**load_dir**: str; The file to load the weights. default='./result/train/'

## Test

```python
python test.py
```

Parameters:

--**device**: str; The device you want to run the program. default=cpu

--**seed**: int; Random seed, default=304, set the seed of training the same as testing to obtain the same performance.

--**noise**: int; If 1, gaussian noise will be add to both train data and test data, default=False

--**sigma**: int; The $\sigma$ of gaussian noise, $\sigma^2_T=\frac{\sigma^2}{255^2}$, cause the image is normalized into [0,1], default=30

--**weight_type**: str; Choose the weights that trained on clean dataset or noise dataset for testing

--**filters**: list of length 4; The number of filters of CNN. default=[32,64,96,96]

--**num_vec**: int; How many vectors does the feature maps be divided to, default=16

--**outSize**: Output size of reservoir for single input, default=1 to obtain the similarity.

--**resSize**: int; The size of reservoir, default=2000

--**alpha**: float; Leaky rate. default=0.3

--**trainLen**: int; The number of data for training

--**batch_size_test**: int; batch size of test dataset.

--**save_dir**: str; he file to save the test result. default='./result/test/'

--**load_dir**: str; The file to load the weights. default='./result/train/weights/'

## Note

I'm a beginner of deep learning and python, this project is for practice. I spent a little time before I wrote this README to make the program run on the GPU, but the CPU and GPU had different results. I didn't fix it due to time, so I didn't add it to the main program. You can try it in Jupyter Notebook.

## Experiment Result

The experiment result is in the folder called "result".
