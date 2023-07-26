import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.io import read_as_csv
from utils.pre_processing import read_image, label_to_index
import os.path
import joblib
from config  import  MODEL_CHECKPOINT_PATH   

data_root = "data\middle-ear-dataset"


# Transforms
def image_transforms(file_name, label) -> np.ndarray:
    file_path = os.path.join(data_root, label, file_name)
    array = read_image(file_path, "zoom", grayscale=True)
    flatten_image = array.flatten()
    return flatten_image


def label_transforms(label) -> int:
    # label_to_index
    return label_to_index(label)


# load csv
train_files, train_labels = read_as_csv("data/train.csv")
test_files, test_labels = read_as_csv("data/test.csv")

# Apply the image_transforms function to train_files and test_files
X_train = np.array(
    [image_transforms(file, label) for file, label in zip(train_files, train_labels)]
)
# print(X_train)
Y_train = np.array([label_transforms(lab) for lab in train_labels])

X_test = np.array(
    [image_transforms(file, label) for file, label in zip(test_files, test_labels)]
)
Y_test = np.array([label_transforms(lab) for lab in test_labels])


# X_train=[
#     [-1,3],
#     [2,1],
#     [-2,2],
#     [-1,2],
#     [-1,0],
#     [1,1],

#     ]
# x_predict=[[1,2]]
# label_to_index={
#     "Red":0,
#     "Blue":1
# }

# Y_train_raw=[
#     "Red",
#     "Blue",
#     "Red",
#     "Blue",
#     "Blue",
#     "Red"
# ]
# Y_train=[label_to_index[l] for l in Y_train_raw]
# print(Y_train)
# index_to_label={0:"Red",1:"Blue"}

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, Y_train)

# pred=clf.predict(x_predict)
# print("Pred:",[index_to_label[p]for p in pred])
# dict_proba=[]
# for proba in clf.predict_proba(x_predict):
#     dict_proba.append({index_to_label[i]:p for i,p in enumerate(proba)})
# print("pred prob:",dict_proba)

# print("Train score:", clf.score(X_train, Y_train))
# print("Test score:", clf.score(X_test, Y_test))

# Prediction
#print("Before save:",clf.predict(X_test))

# Save the model
joblib.dump(clf, MODEL_CHECKPOINT_PATH)

# Load the k-NN model from the file
loaded_knn_model = joblib.load(MODEL_CHECKPOINT_PATH)


# print(loaded_knn_model.predict(X_test))