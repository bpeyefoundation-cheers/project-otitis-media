import os
from os.path import join
from utils.io import get_image_label_pairs
from sklearn.model_selection import train_test_split
from utils.io import save_as_csv

DATA_DIR = "data\middle-ear-dataset"

data_folders = os.listdir(DATA_DIR)

print(data_folders)

X = []
y = []
for folders in data_folders:
    data_path = join(DATA_DIR, folders)
    label = folders
    print(data_path, label)
    file_names, labels = get_image_label_pairs(data_path, label)
    X.extend(file_names)
    y.extend(labels)

print(len(X), len(y))

# splitting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('X train size:',len(X_test))
print("Y test size:",len(y_test))

#save tthe train and test sets as csv
save_as_csv(X_train,y_train,'data/train.csv')
save_as_csv(X_test,y_test,'data/test.csv')



