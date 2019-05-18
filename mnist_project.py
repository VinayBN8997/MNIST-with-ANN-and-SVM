import argparse
import struct as st
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import datetime
import os
import cv2
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def open_folder(path):
	print("Loading data")
	images_file = open(path+'/images','rb')
	labels_file = open(path+'/labels','rb')

	#Getting labels data
	magic = st.unpack('>4B',labels_file.read(4))
	nImg = st.unpack('>I',labels_file.read(4))[0] #Number of images
	nBytesTotal = nImg #Since each pixel data is 1 byte
	labels_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,labels_file.read(nBytesTotal))).reshape((nImg)).reshape(nImg,-1)

	#Getting images data
	magic = st.unpack('>4B',images_file.read(4))
	nImg = st.unpack('>I',images_file.read(4))[0] #num of images
	nR = st.unpack('>I',images_file.read(4))[0] #num of rows
	nC = st.unpack('>I',images_file.read(4))[0] #num of column
	nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
	images_array = (255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,images_file.read(nBytesTotal))).reshape((nImg,nR,nC))).reshape(nImg,-1)

	images_array = images_array.astype('float32') / 255.

	return [images_array,labels_array]

def open_custom(custom_dir):
	#To open the custom images folder and return the list of numpy arrays of the images
	images_list = []
	names = os.listdir(custom_dir)
	for name in names:
		if '.png' in name or '.jpg' in name or 'jpeg' in name:
			image = cv2.imread(custom_dir+"/"+name,0) #Open gray scale image 
			image = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
			images_list.append(image)
	return images_list


def train_model(training_images,training_labels,model_name = 'ANN'):
	#To tran the model
	print("Training model")
	if model_name == 'SVM':
		model = svm.SVC(gamma='scale',max_iter = 100, random_state =1, probability=True)
	else:
		model = MLPClassifier(solver='adam',hidden_layer_sizes=(50,20), random_state=2019, verbose=True, max_iter = 30)
	time_1 = datetime.datetime.now()
	model.fit(training_images, training_labels)
	time_2 = datetime.datetime.now()
	duration = (time_2 - time_1).seconds
	print("Time taken for training:" , duration)
	pickle.dump(model, open(model_name+".sav", 'wb'))
	return model

def test_model(testing_images,model_name = 'ANN'):
	#To test the model
	print("Testing data")
	model = pickle.load(open(model_name+".sav", 'rb'))
	predict_labels = model.predict(testing_images)
	return predict_labels


def get_results(y_true, y_predict):
	#To get the accuracy of the predictions
	return round(accuracy_score(y_true, y_predict)*100,2)

def get_confusionmatrix(y_true,y_predict):
	#To get the confusion matrix of the predctions
	res = confusion_matrix(y_true, y_predict)
	for row in res:
		print(row)

def training_stage(values):
	print("Training stage")
	training_data = open_folder(values.t)
	training_images = training_data[0]
	training_labels = training_data[1]
	model = train_model(training_images,training_labels,values.m)

def testing_stage(values):
	print("Testing stage")
	testing_data = open_folder(values.p)
	testing_images = testing_data[0]
	testing_labels = testing_data[1]
	'''
	#To save test data images of MNIST
	for i in range(len(testing_images)):
		print(testing_images[i].shape)
		print(np.mean(testing_images[i]))
		image = testing_images[i].reshape(28,28) * 255;
		cv2.imwrite("custom_3/"+str(i)+".jpg",image)
	'''
	
	predict_labels = test_model(testing_images,values.m)
	accuracy = get_results(testing_labels,predict_labels)
	print("===========================")
	print("Accuracy: ",accuracy)
	print("===========================")
	print("===========================")
	print("Confusion Matrix")
	get_confusionmatrix(testing_labels,predict_labels)
	print("===========================")
	

parser = argparse.ArgumentParser()
parser.add_argument('-t', help="Training files directory")
parser.add_argument('-p', help="Testing files directory") 
parser.add_argument('-m', help="To select model, either SVM or ANN", default = 'ANN') #For model selection
parser.add_argument('-c', help="Custom images directory") #For giving a custom image
args = parser.parse_args()

if args.c:
	print(args.m + " model")
	images_list = open_custom(args.c)
	model = pickle.load(open(args.m+".sav", 'rb'))
	count = 1
	for image in images_list:
		count += 1
		image = image.astype('float32') / 255.
		predict_label = model.predict_proba(image.reshape(1,-1))[0]
		label = 0
		proba = 0.0
		for i in range(10):
			if predict_label[i] > proba:
				proba = predict_label[i]
				label = i
		print("Predicted label: ",label, " with probability: ",proba)
		fig = plt.figure(figsize=(4, 2))
		ax = fig.add_subplot(1, 2, 1)
		plt.imshow(image, cmap='gray')
		if proba >= 0.9:
			ax.text(30,7,"Predicted label: \n"+str(label),  fontsize=15)
		else:
			ax.text(30,7,"Not a number",  fontsize=15)
		plt.show()





elif args.t and not args.p: #Only training and no testing - Train the model and save it for traing data
	print(args.m + " model")
	training_stage(args)

elif args.p and not args.t: #Only tesing and no training - Just test the saved model for testing data
	print(args.m + " model")
	testing_stage(args)

elif args.t and args.p: #First traing the model, save it and then use it for testing
	print(args.m + " model")
	training_stage(args)
	testing_stage(args)

else:
	print("===========================")
	print("No option was selected")
	print("===========================")


