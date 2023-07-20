import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.io import read_as_csv
from utils.preprocessing import read_image
from viz.visualization import label_to_index
import csv
import os


data_root = "data\middle-ear-dataset"

#transforms
def image_transforms(file_name , label) -> np.ndarray :
    file_path = os.path.join(data_root , label, file_name)
    array = read_image(file_path)
    flatten_image = array.flatten()
    print(flatten_image)
    return flatten_image

def label_transforms( label) -> int :
    #label to index 
    return label_to_index(label)

# load csv
train_files , train_labels = read_as_csv("data/train.csv")
test_files , test_labels = read_as_csv("data/test.csv")

#apply the image tranform function to train and test file
X_train = np.array([image_transforms(file_name , label) for file_name , label in zip(train_files, train_labels)])
Y_train = np.array([label_transforms(label) for label in train_labels])

X_test = np.array([image_transforms(file_name , label) for file_name , label in zip(test_files, test_labels)])
Y_test = np.array([label_transforms(label) for label in test_labels])








# X_train = [
#     [-1, 3], 
#     [2, 1],
#     [-2, 2],
#     [-1, 2],
#     [-1, 0],
#     [1, 1],
# ]

# # print(X_train)


# X_predict = [
# [1,2]
# ]
# # print(X_predict)

# label_to_index = {
#     "red" : 0,
#     "blue" : 1
# }

# index_to_label = {
#     0: "red" ,
#     1 : "blue"
# }

# Y_train_raw = [
#     "red",
#     "blue",
#     "red",
#      "blue",
#      "blue", 
#      "red", 
     
# ]
# Y_train = [label_to_index[l] for l in Y_train_raw]
# # print(Y_train)

clf = KNeighborsClassifier(n_neighbors= 3)
clf.fit(X_train , Y_train)
clf.fit(X_test, Y_test)

# clf.predict(X_predict)

# pred = clf.predict(X_predict)


# print("pred:" , [index_to_label[p] for p in pred])

# dict_proba = []

# for proba in clf.predict_proba(X_predict):
#     dict_proba.append({index_to_label[i]: p for i, p in enumerate(proba)})

# print("pred prob :" , dict_proba)
print("Train score", clf.score(X_train, Y_train))
print("Test score", clf.score(X_test, Y_test))
