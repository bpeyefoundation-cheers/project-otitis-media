import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.io import read_as_csv
from config import MODEL_CHECKPOINT_PATH
from utils.preprocessing import image_transforms,  label_transforms
import joblib



# load csv
train_files , train_labels = read_as_csv("data/train.csv")
test_files , test_labels = read_as_csv("data/tests.csv")

#apply the image tranform function to train and test file
X_train = np.array([image_transforms(file_name , label) for file_name , label in zip(train_files, train_labels)])
Y_train = np.array([label_transforms(label) for label in train_labels])




clf = KNeighborsClassifier(n_neighbors= 4)
clf.fit(X_train , Y_train)

# clf.predict(X_predict)

# pred = clf.predict(X_predict)


# print("pred:" , [index_to_label[p] for p in pred])

# dict_proba = []

# for proba in clf.predict_proba(X_predict):
#     dict_proba.append({index_to_label[i]: p for i, p in enumerate(proba)})

# print("pred prob :" , dict_proba)
print("Train score", clf.score(X_train, Y_train))

# Save the model
joblib.dump(clf, MODEL_CHECKPOINT_PATH)