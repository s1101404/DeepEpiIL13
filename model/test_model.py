from sklearn import metrics
from sklearn.metrics import roc_curve
from tensorflow.keras.models import load_model
import numpy as np
import math
import argparse
import os
from tqdm import tqdm
import tensorflow as tf
import gc

parser = argparse.ArgumentParser()
parser.add_argument("-m_path","--model_path", type=str, help="the model path")
parser.add_argument("-x_ts_data","--x_test_path", type=str, help="x_test data path")
parser.add_argument("-y_ts_data","--y_test_path", type=str, help="y_test data path")
args=parser.parse_args()

model_path = args.model_path
x_test_path = args.x_test_path
y_test_path = args.y_test_path
print("m_p",model_path)
print("x_t",x_test_path)
print("y_t",y_test_path)

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
    print(f'Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]

    
    print(f'Best Threshold={threshold}')

    y_pred = (pred_test[:, 1] >= threshold).astype(int)

    TN, FP, FN, TP =  metrics.confusion_matrix(y_test[0:][:,1], y_pred).ravel()
    Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
    Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
    Acc = (TP+TN)/(TP+FP+TN+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
    F1 = 2*TP/(2*TP+FP+FN)
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}\n')
    #SAVEROC(fpr,tpr,AUC)
    return TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC

model = load_model(model_path)

x_test,y_test=data_load(x_test_path,y_test_path,35)
# x_test = np.load(x_test_path)
# y_test = np.load(y_test_path)

TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(model, x_test, y_test)