import argparse
import os
import sys
import csv
import base64
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

cwd = os.getcwd()
dataPath = cwd + "/data/"
dbPath = dataPath + "driving_log.csv"
IMGPath = dataPath + "IMG/"

### Import data
data = []
with open(dbPath) as F:
    reader = csv.reader(F)
    for i in reader:
        data.append(i) 

print("data imported")

### Emtpy generators for feature and labels
features = ()
labels = ()

### size of the data
dataSize = len(data)
print("Number of Records:", dataSize)

### This function will resize the images from front, left and
### right camera to 18 x 80 and turn them into lists.
def load_image(data_line, j):
    print(data_line)
    img = plt.imread(data_line[j].strip())[65:135:4,0:-1:4,0]
    lis = img.flatten().tolist()
    return lis

data = data[:100]

# For each item in data, convert camera images to single list
# and save them into features list.
for i in tqdm(range(1, int(len(data))), unit='items'):
    for j in range(3):
        features += (load_image(data[i],j),)
       
item_num = len(features)
print("features size", item_num)

# A single list will be convert back to the original image shapes.
# Each list contains 3 images so the shape of the result will be
# 54 x 80 where 3 images aligned vertically.
features = np.array(features).reshape(item_num, 18, 80, 1)
print("features shape", features.shape)

### Save labels    
for i in tqdm(range(1, int(len(data))), unit='items'):
    for j in range(3):
        labels += (float(data[i][3]),)

labels = np.array(labels)

print("features:", features.shape)
print("labels:", labels.shape)

from sklearn.cross_validation import train_test_split

# Get randomized datasets for training and test
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.20,
    random_state=1221003)

# Get randomized datasets for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=1210093)

# Print out shapes of new arrays
train_size = X_train.shape[0]
test_size = X_test.shape[0]
valid_size = X_valid.shape[0]
input_shape = X_train.shape[1:]
features_count = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]

print("Training Size:", train_size)
print("Validation Size:", valid_size)
print("Test Size:", test_size)
print("Input Shape:", input_shape)
print("Features Count:", features_count)

import pickle

# Save the data for easy access
pickle_file = 'camera.pickle'
stop = False

while not stop:
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': X_train,
                        'train_labels': y_train,
                        'valid_dataset': X_valid,
                        'valid_labels': y_valid,
                        'test_dataset': X_test,
                        'test_labels': y_test,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

        print('Data cached in pickle file.')
        stop = True
    else:
        print("Please use a different file name other than camera.pickle")
        pickle_file = input("Enter: ")
