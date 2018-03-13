from sklearn.linear_model import SGDClassifier
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import cv2

classes = ['cats','dogs']

image_size = 128

train_path = "//content//drive4//train"


def read_train_sets(train_path):
    images = []
    labels = []
    img_names = []
    cls = []
    for fields in classes:
        tempImages = []
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)

            image = cv2.resize(image, (image_size, image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            [width1, height1] = [image.shape[0], image.shape[1]]

            f2 = image.reshape(width1 * height1);

            images.append(f2)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)

    print(len(images))
    return images, cls


# We shall load all the training and validation images and labels into memory using openCV and use that during training

data, labels = read_train_sets(train_path)
data=np.array(data)
print(data.shape)
labels=np.array(labels)
print(labels.shape)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                                                                    test_size=0.2, random_state=42)
size=len(train_labels)
train_data=np.array(train_data)
print(train_data.shape)
train_labels=np.array(train_labels)
print(train_labels.shape)

est = SGDClassifier(penalty="l2", alpha=0.001, shuffle=True)
progressive_validation_score = []
train_score = []
for datapoint in range(1, size, 200):

    if datapoint <= size:
        print(datapoint)
        X_batch = train_data[datapoint:datapoint + 200]
        print(X_batch.shape)
        y_batch = train_labels[datapoint:datapoint + 200]
        est.partial_fit(X_batch, y_batch, classes=np.unique(y_batch))

pred = est.predict(test_data)

print(metrics.accuracy_score(test_labels, pred))
