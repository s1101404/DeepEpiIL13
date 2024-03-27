import h5py
import os
import pickle

from tqdm import tqdm
from time import gmtime, strftime

import numpy as np
import math

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_curve

import tensorflow as tf
from tensorflow.keras import layers,Model

##

from sklearn.model_selection import KFold

import gc

import time
from sklearn.model_selection import KFold

import import_IL13Pred as load_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-D_tr","--Data_train_path", type=str, help="input_file")
parser.add_argument("-L_tr","--Label_train_path", type=str, help="test_file")
parser.add_argument("-D_ts","--Data_test_path", type=str, help="input_file")
parser.add_argument("-L_ts","--Label_test_path", type=str, help="test_file")

parser.add_argument("-n_fil","--num_filter", type=int, default=256, help="the number of filters in the convolutional layer")
parser.add_argument("-n_hid","--num_hidden", type=int, default=1000, help="the number of hidden units in the dense layer")
parser.add_argument("-bs","--batch_size", type=int, default=1000, help="the batch size")
parser.add_argument("-ws","--window_sizes", nargs="+", type=int, default=[2,4,8], help="the window sizes for convolutional filters")
parser.add_argument("-n_cls","--num_classes", type=int, default=2, help="the number of classes")
parser.add_argument("-n_feat","--num_feature", type=int, default=1024, help="the number of features")
parser.add_argument("-e","--epochs", type=int, default=20, help="the number of epochs for training")
parser.add_argument("-val","--validation_mod", type=str, default="independent", help="the mod for validation 'cross' or 'independent'")
parser.add_argument("-k_fold","--num_k_fold", type=int, default=5, help="the number of k for k_fold cross validation")


# # PARAM

DATA_LABEL=load_data.data_label()

args=parser.parse_args()
MAXSEQ = 35

DATA_TYPE = "IL-13"
NUM_FEATURE = args.num_feature
NUM_FILTER = args.num_filter
NUM_HIDDEN = args.num_hidden
BATCH_SIZE  = args.batch_size
WINDOW_SIZES = args.window_sizes
NUM_CLASSES = 2
CLASS_NAMES = ['Negative','Positive']
EPOCHS      = args.epochs
K_Fold = args.num_k_fold
VALIDATION_MODE=args.validation_mod

#Data Path
x_train_pt = args.Data_train_path
y_train_pt = args.Label_train_path

x_test_pt = args.Data_test_path
y_test_pt = args.Label_test_path

#Data Load
x_train,y_train=data_load(x_train_pt,y_train_pt,35)
x_test,y_test=data_load(x_test_pt,y_test_pt,35)

print(x_train.shape)
print(y_train.shape)

import datetime

write_data=[]
a=datetime.datetime.now()
write_data.append(time.ctime())
write_data.append(DATA_LABEL)
write_data.append(DATA_TYPE)
write_data.append(WINDOW_SIZES)
write_data.append(NUM_FILTER)
write_data.append(NUM_DEPENDENT)
write_data.append(VALIDATION_MODE)

def time_log(message):
    print(message," : ",strftime("%Y-%m-%d %H:%M:%S", gmtime()))

#Model
class DeepScan(Model):

	def __init__(self,
	             input_shape=(1, MAXSEQ, NUM_FEATURE),
	             window_sizes=[1024],
	             num_filters=256,
	             num_hidden=1000):
		super(DeepScan, self).__init__()
		# Add input layer
		self.input_layer = tf.keras.Input(input_shape)
		self.window_sizes = window_sizes
		self.conv2d = []
		self.maxpool = []
		self.flatten = []
		for window_size in self.window_sizes:
			self.conv2d.append(
			 layers.Conv2D(filters=num_filters,
			               kernel_size=(1, window_size),
			               activation=tf.nn.relu,
			               padding='valid',
			               bias_initializer=tf.constant_initializer(0.1),
			               kernel_initializer=tf.keras.initializers.GlorotUniform()))
			self.maxpool.append(
			 layers.MaxPooling2D(pool_size=(1, MAXSEQ - window_size + 1),
			                     strides=(1, MAXSEQ),
			                     padding='valid'))
			self.flatten.append(layers.Flatten())
		self.dropout = layers.Dropout(rate=0.7)
		self.fc1 = layers.Dense(
		 num_hidden,
		 activation=tf.nn.relu,
		 bias_initializer=tf.constant_initializer(0.1),
		 kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.fc2 = layers.Dense(NUM_CLASSES,
		                        activation='softmax',
		                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))

		# Get output layer with `call` method
		self.out = self.call(self.input_layer)

	def call(self, x, training=False):
		_x = []
		for i in range(len(self.window_sizes)):
			x_conv = self.conv2d[i](x)
			x_maxp = self.maxpool[i](x_conv)
			x_flat = self.flatten[i](x_maxp)
			_x.append(x_flat)

		x = tf.concat(_x, 1)
		x = self.dropout(x, training=training)
		x = self.fc1(x)
		x = self.fc2(x)  #Best Threshold
		return x

def data_load(x_folder, y_folder,NUM_CLASSES,):
    x_test = []
    y_test = []

    x_files = [file for file in os.listdir(x_folder) if file.endswith('.npy')]
    
    # Iterate through x_folder with tqdm
    for file in tqdm(x_files, desc="Loading data", unit="file"):
        x_path = os.path.join(x_folder, file)
        x_data = np.load(x_path)
        
        x_test.append(x_data.astype('float16'))

        # Get the corresponding y file
        y_file = file[:-4] + '.label'
        y_path = os.path.join(y_folder, y_file)

        with open(y_path, 'r') as y_f:
            lines = y_f.readlines()
            y_data = np.array([int(x) for x in lines[1].strip()])
            y_test.append(y_data.astype('float16'))
            
            del y_data
            gc.collect()
            
        del x_data
        del y_file
        gc.collect()
        
    # Concatenate all the data
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Add new dimensions to x_test and y_test
    x_test = np.expand_dims(x_test, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    y_test = tf.keras.utils.to_categorical(y_test,NUM_CLASSES)
    
    del x_files
    gc.collect()
    
    return x_test, y_test

# np.savez(f"train.npz", feature=x_train, label=y_train)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def model_test(model, x_test, y_test):

    print(x_test.shape)
    pred_test = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:,1], pred_test[:, 1])
    AUC = metrics.auc(fpr, tpr)
    #tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='mCNN')
    display.plot()
    

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]

    y_pred = (pred_test[:, 1] >= threshold).astype(int)

    TN, FP, FN, TP =  metrics.confusion_matrix(y_test[0:][:,1], y_pred).ravel()
    # tn=TN/100
    # fp=FP/100
    # fn=FN/100
    # tp=TP/100

    # a = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
    Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
    Acc = (TP+TN)/(TP+FP+TN+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
    # MCC = (tp*tn-fp*fn)/math.sqrt(a) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
    F1 = 2*TP/(2*TP+FP+FN)
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}\n')
    #SAVEROC(fpr,tpr,AUC)
    return TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC


if(VALIDATION_MODE=="cross"):
	#Training model with cross validation
    time_log("Start cross")
    config = tf.compat.v1.ConfigProto()

    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as session:
    	kfold = KFold(n_splits = K_Fold, shuffle = True, random_state = 2)
    	results=[]
    	i=1
    	for train_index, test_index in kfold.split(x_train):
    		print(i,"/",K_Fold,'\n')
    		X_train, X_test = x_train[train_index], x_train[test_index]
    		Y_train, Y_test = y_train[train_index], y_train[test_index]
    		
    		model = DeepScan(
    		num_filters=NUM_FILTER,
    			num_hidden=NUM_HIDDEN,
    			window_sizes=WINDOW_SIZES)
    		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    		model.build(input_shape=X_train.shape)
    		history=model.fit(
    			X_train,
    			Y_train,
    			batch_size=BATCH_SIZE,
    			epochs=EPOCHS,
    			callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
    			verbose=1,
    			shuffle=True
    		)

			#testing model with independent validation 
    		TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(model, X_test, Y_test)
    		results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
    		i+=1
    		
    		del X_train
    		del X_test
    		del Y_train
    		del Y_test
    		gc.collect()
    		
    	mean_results = np.mean(results, axis=0)
    	print(f'TP={mean_results[0]:.4}, FP={mean_results[1]:.4}, TN={mean_results[2]:.4}, FN={mean_results[3]:.4}, Sens={mean_results[4]:.4}, Spec={mean_results[5]:.4}, Acc={mean_results[6]:.4}, MCC={mean_results[7]:.4}, AUC={mean_results[8]:.4}\n')
    	write_data.append(mean_results[0])
    	write_data.append(mean_results[1])
    	write_data.append(mean_results[2])
    	write_data.append(mean_results[3])
    	write_data.append(mean_results[4])
    	write_data.append(mean_results[5])
    	write_data.append(mean_results[6])
    	write_data.append(mean_results[7])
    	write_data.append(mean_results[8])
        
elif (VALIDATION_MODE=="independent"):
	#Training model with independent validation
    time_log("Start Model Train")
    results=[]
    model = DeepScan(
        num_filters=NUM_FILTER,
        num_hidden=NUM_HIDDEN,
        window_sizes=WINDOW_SIZES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=x_train.shape)
    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
    )
    time_log("End Model Train")
    time_log("Start Model Test")
    
    del x_train
    del y_train
    gc.collect()

    #testing model with independent validation 
    TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(model, x_test, y_test)
    results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
    print(f'TP={results[0][0]:.4}, FP={results[0][1]:.4}, TN={results[0][2]:.4}, FN={results[0][3]:.4}, Sens={results[0][4]:.4}, Spec={results[0][5]:.4}, Acc={results[0][6]:.4}, MCC={results[0][7]:.4}, AUC={results[0][8]:.4}\n')
    print(results)
    write_data.append(results[0])
    write_data.append(results[1])
    write_data.append(results[2])
    write_data.append(results[3])
    write_data.append(results[4])
    write_data.append(results[5])
    write_data.append(results[6])
    write_data.append(results[7])
    write_data.append(results[8])

def save_csv(write_data,a):
    import csv
    b=datetime.datetime.now()
    write_data.append(b-a)
    open_csv=open("relue.csv","a")
    write_csv=csv.writer(open_csv)
    write_csv.writerow(write_data)
    
save_csv(write_data,a)