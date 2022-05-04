# este scriot obtiene las métricas de las bases de datos entrenadas
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import keras

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import sys
from argparse import ArgumentParser
import cv2
import os
import glob

##################################################################
# arguments
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="path_model", help="model drowsy path", metavar="DIR")
parser.add_argument("-d", "--dataset", dest="path_dataset", help="dataset TEST path", metavar="DIR")  
args = parser.parse_args()
##################################################################

print(args.path_model)
print(args.path_dataset)

current_dir = os.path.dirname(os.path.abspath(__file__))

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return 1-preds[0][0]


model = keras.models.load_model(args.path_model)

files_dormido = glob.glob(args.path_dataset + "/dormido/*") 
files_despierto = glob.glob(args.path_dataset + "/despierto/*") 

y_true = np.hstack(( np.ones(len(files_dormido)), np.zeros(len(files_despierto)) )) 

y_pred = []
for file in files_dormido:
    img = cv2.imread(file)
    img = cv2.resize(img, (224,224))    
    y_pred.append(predict(model, img))

for file in files_despierto:
    img = cv2.imread(file)
    img = cv2.resize(img, (224,224))    
    y_pred.append(predict(model, img))

y_pred = np.array(y_pred).round()

#print(y_true)
#print(y_pred)

acc = accuracy_score(y_true, y_pred)
matrix = confusion_matrix(y_true, y_pred, labels=[1,0])
f1Score= f1_score(y_true, y_pred , average="macro")

print(acc)
print(matrix)
print()
print("f1 score:")
print(f1Score)
# python3 test.py -m ../models/modelSiso_y_NTHU_Inception3.h5 -d /home/vicente/datasets/SISO/TEST/
# python3 test.py -m /mnt/disk3/ENTRENAMIENTOS/siso_nthu/100_epochs_SISO_NTHU.h5 -d /mnt/disk3/ENTRENAMIENTOS/siso_nthu/TEST/
