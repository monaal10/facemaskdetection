from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import glob
import os
import cv2
import random as rand
import xml.etree.ElementTree as et
for dirname, _, filenames in os.walk(r'C:\Users\Monaal\Desktop\monaal\data science\Face mask detection'):
    for filename in filenames:
       print(os.path.join(dirname, filename))     
        
dic = {"image": [],"Dimensions": []}
for i in range(1,116):
	dic[f'Object {i}']=[]

for file in os.listdir(r"C:\Users\Monaal\Desktop\monaal\data science\Face mask detection\annotations") :
    row = []
    parser = et.XMLParser(encoding="utf-8")
    xml= et.parse(r"C:/Users/Monaal/Desktop/monaal/data science/Face mask detection/annotations/" +file, parser=parser)
    root = xml.getroot()
    img = root[1].text
    row.append(img)
    h,w = root[2][0].text,root[2][1].text
    row.append([h,w])
    for i in range(4,len(root)):
        temp = []
        temp.append(root[i][0].text)
        for point in root[i][5]:
            temp.append(point.text)
        row.append(temp)
    for i in range(len(row),119):
        row.append(0)
    for i,each in enumerate(dic):
        dic[each].append(row[i])
df = pd.DataFrame(dic)

image_directories=sorted(glob.glob(os.path.join( r"C:\Users\Monaal\Desktop\monaal\data science\Face mask detection\images\*.png*")))

classes = ["without_mask","mask_weared_incorrect","with_mask"]
labels = []
data = []
for idx,image in enumerate(image_directories):
    img  = cv2.imread(image)
    #scale to dimension
    X,Y = df["Dimensions"][idx]
    cv2.resize(img,(int(X),int(Y)))
    for obj in df.columns[3:]:
        info = df[obj][idx]
        if info!=0:
            label = info[0]
            info[0] = info[0].replace(str(label), str(classes.index(label)))
            info=[int(each) for each in info]
            face = img[info[2]:info[4],info[1]:info[3]]
            if((info[3]-info[1])>40 and (info[4]-info[2])>40):
                try:
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    data.append(face)
                    labels.append(label)
                    if(label=="mask_weared_incorrect"):
                        data.append(face)
                        labels.append(label)
                except:
                 pass
data = np.array(data, dtype="float32")
labels = np.array(labels)
            
 #cv2.imshow('grey image resized', c)
 #k = cv2.waitKey(1000)
 #cv2.destroyAllWindows()
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#train-test split
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, stratify=labels, random_state=42)

#hyperparameters
ilr = 1e-4
ep = 40
bs = 32

#training image gen for augmentation
aug = ImageDataGenerator(
    zoom_range=0.2,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
    )
 
#loading MobileNetV2 network for fine-tuning
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

#contruct headModel that will be on top of the baseModel
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

#consruct the actual trainable model
model = Model(inputs=baseModel.input, outputs=headModel)
#loop base layers to freeze them (they won't be updated during the initial training process)
for layer in baseModel.layers:
    layer.trainable = False
# compile model
opt = Adam(lr = ilr, decay = ilr / ep)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the head of the network
H = model.fit(
    aug.flow(trainX, trainY, batch_size=bs),
    steps_per_epoch=len(trainX) // bs,
    validation_data=(testX, testY),
    validation_steps=len(testX) // bs,
    epochs=ep)
 
predIdxs = model.predict(testX, batch_size = bs)
predIdxs = np.argmax(predIdxs, axis = 1)

#classification report
print(classification_report(testY.argmax(axis = 1), predIdxs,
    target_names = lb.classes_))

#plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, ep), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, ep), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, ep), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, ep), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss & Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig(["plot"])