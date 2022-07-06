

#import all libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import itertools

#make a dataset
df=pd.read_pickle("final_audio_csv_datset/prepare_data.csv")

def convert_bytes(size,unit=None):
    if unit=="KB":
        return print('file size: '+str(round(size/1024,3))+'Kilobytes')
    elif unit=="MB":
        return print('file size: '+str(round(size/(1024*1024),3))+'Megabytes')
    else:
        return print('file size: '+str(size)+'bytes')
    


X=df["feature"].values
X=np.concatenate(X,axis=0).reshape(len(X),40)

y=np.array(df["class_label"].tolist())
y=to_categorical(y)

#train test splite dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

def get_file_size(file_path):
    size=os.path.getsize(file_path)
    return size

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

model=Sequential([
    Dense(256,input_shape=X_train[0].shape),
    Activation('relu'),
    Dropout(0.5),
    Dense(256),
    Activation('relu'),
    Dropout(0.5),
    Dense(2,activation='softmax')
    ])

print(model.summary())
model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
    )

print("Model Score: \n")
history=model.fit(X_train,y_train,epochs=50)
wake_word_train_model="saved_model/WWD.h5"
model.save(wake_word_train_model)
convert_bytes(get_file_size(wake_word_train_model),"MB")
score=model.evaluate(X_test,y_test)
print(score)

print("Model classification report: \n")
y_pred=np.argmax(model.predict(X_test),axis=1)
cm=confusion_matrix(np.argmax(y_test,axis=1), y_pred)
print(classification_report(np.argmax(y_test,axis=1), y_pred))

plot_confusion_matrix(cm,classes=["Does not contain wake word","has wake word"])
#plot_confusion_matrix(X= np.argmax(y_test,axis=1), y_true= y_pred,labels= ["Does not contain wake word","has wake word"], normalize=False)


#tenosr flow lite model

TF_LITE_MODEL_FILE_NAME="tf_lite_model.tflite"

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sys import getsizeof

tf_lite_converter=tf.lite.TFLiteConverter.from_keras_model(model)
#tf_lite_converter.optimizations=[tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tf_lite_converter.optimizations=[tf.lite.Optimize.DEFAULT]
tf_lite_converter.target_spec.supported_types=[tf.float16]

tflite_model=tf_lite_converter.convert()

tflite_model_name=TF_LITE_MODEL_FILE_NAME
open(tflite_model_name,"wb").write(tflite_model)

convert_bytes(get_file_size(TF_LITE_MODEL_FILE_NAME),"KB")




