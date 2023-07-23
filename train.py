import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.io import read_as_csv
from utils.preprocessing import image_transforms, label_transforms
import os.path
import joblib
from config import MODEL_CHECKPOINT_PATH


# load csv
train_files, train_labels = read_as_csv("data/train.csv")
# test_files, test_labels = read_as_csv("data/test.csv")

# Apply the image_transforms function to train_files and test_files
X_train = np.array(
    [image_transforms(file, label) for file, label in zip(train_files, train_labels)]
)
# print(X_train)
Y_train = np.array([label_transforms(lab) for lab in train_labels])


clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, Y_train)

# Save the model
joblib.dump(clf, MODEL_CHECKPOINT_PATH)

# Save the model
joblib.dump(clf, MODEL_CHECKPOINT_PATH)
