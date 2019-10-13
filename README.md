# MNIST-with-ANN-and-SVM
MNIST classification with ANN and SVM

## Dataset:
The dataset was downloaded from :
MNIST handwritten digit database, Yann LeCun, Corinna Cortes and Chris Burges
Link: http://yann.lecun.com/exdb/mnist/

The dataset consists of 4 files:
train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes) 

These files are organized and renamed as images and labels under Training and Testing directories:
- ├── Testing
- │   ├── images
- │   └── labels
- ├── Training
- │   ├── images
- │   └── labels

### IDX file format is a binary file format.
The basic format is

magic number
 size in dimension 0
 size in dimension 1
 size in dimension 2
 …..
 size in dimension N
 data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
 0x08: unsigned byte
 0x09: signed byte
 0x0B: short (2 bytes)
 0x0C: int (4 bytes)
 0x0D: float (4 bytes)
 0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices….
The sizes in each dimension are 4-byte integers.
The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.

On the execution side, the IDX files are converted into numpy array for training/testing purpose.

# Command line arguments:
-t : To provide the training data directory. Also works as a flag to say whether to train or not.

-p : To provide the testing data directory. Also works as a flag to say whether to test or not.

-m : To decide which algorithm to use ["SVM" or "ANN"]. Default is "ANN"

# SVM: 
Link: https://scikit-learn.org/stable/modules/svm.html

Results: gamma = 'scale' , random_state = 1 [To get constant results for a fixed configuration]

Different values of max_iterations [10,20,30,50,100] can be used to train more and more. The more you train, better results are obtained.

# ANN:
Link: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

Results: solver = 'adam', alpha = 1e-5, random_state = 1 [To get constant results for a fixed configuration]

hidden_layer_sizes = The ith element represents the number of neurons in the ith hidden layer.

Different shapes of hideed layers can be used [(3,2), (5,4), (10,7), (20,15,10), (200,100,50)]

The larger the size, the model learns more parameters, but has a chance of overfitting the training data. 

The attribute max_iterations can also be used in limiting the number of epochs. Default value is 200. It is recommended that, the larger the size of dataset, training for more epochs is helpful.


# To run:
1. Train and test ANN
python3 mnist_project.py -t Training/ -p Testing/ -m ANN
2. Train and test SVM
python3 mnist_project.py -t Training/ -p Testing/ -m SVM
3. Only train and save model
python3 mnist_project.py -t Training/ -m "ANN or SVM"
4. Only test and display accuracy
python3 mnist_project.py -p Testing/ -m "ANN or SVM"
5. When -m argument is not used, ANN is used by default for both training and testing purpose.
6. To try with custom data with dashboard support
python3 mnist_project.py -c Custom/

# Results
![Screenshot from 2019-05-18 12-24-52](https://user-images.githubusercontent.com/33830482/57965900-57510d00-7968-11e9-81ab-c5b106ca192d.png)

![Screenshot from 2019-05-18 12-25-24](https://user-images.githubusercontent.com/33830482/57965908-7b145300-7968-11e9-84e7-984472a15136.png)

![Screenshot from 2019-05-18 12-26-01](https://user-images.githubusercontent.com/33830482/57965909-7b145300-7968-11e9-99cd-b6b9f94a9f4b.png)
