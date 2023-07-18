import os
from os.path import join
from utils.io import get_image_label_pairs
from sklearn.model_selection import train_test_split
from utils.io import save_as_csv
import argparse
from argparse import ArgumentParser,BooleanOptionalAction

parser = argparse.ArgumentParser(description='test_train_scripting')
parser.add_argument('-d','--data-dir',type=str, help='data directory',default="data\middle-ear-dataset")
parser.add_argument('-t','--test-size',type=float,help='test size',default=0.2)
parser.add_argument('-r','--random-state',type=int,default=42)
parser.add_argument('-s','--shuffle',type=str,default=True)

parser.add_argument('--do-shuffle',action=BooleanOptionalAction)


args=parser.parse_args()
print(args.data_dir)
print(args.test_size)
print(args.random_state)
print(args.shuffle)


DATA_DIR = args.data_dir

data_folders = os.listdir(DATA_DIR)

#print(data_folders)

X = []
y = []
for folders in data_folders:
    data_path = join(DATA_DIR, folders)
    label = folders
    print(data_path, label)
    file_names, labels = get_image_label_pairs(data_path, label)
    X.extend(file_names)
    y.extend(labels)

# print(len(X), len(y))

# splitting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state,shuffle=args.shuffle,stratify=None)
print('X test size:',len(X_test))
print("Y test size:",len(y_test))

#save tthe train and test sets as csv
save_as_csv(X_train,y_train,'data/train.csv')
save_as_csv(X_test,y_test,'data/test.csv')

