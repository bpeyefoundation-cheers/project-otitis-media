import os
import argparse
from os.path import  join
from utils.io import get_image_label_pairs , save_as_csv
from sklearn.model_selection import train_test_split


#train and test split using argparse

parser = argparse.ArgumentParser(description='input the train test size')
parser.add_argument('--data-dir' ,type =str , default='data\middle-ear-dataset' , help = "enter the data path")
parser.add_argument('--test-size' , type = float , default=0.2 , help = "enter the test size")
parser.add_argument('--random-state' ,type = int, default = 42, help="enter the random state" )
args = parser.parse_args()
print(args.data_dir)
print(args.test_size)
print(args.random_state)

DATA_DIR = args.data_dir

data_folders = os.listdir(DATA_DIR)
print(data_folders)

X =[]
Y= []

for folder in data_folders:
    data_path = join(DATA_DIR, folder)
    label = folder

    file_names , labels = get_image_label_pairs(data_path , label)

    X.extend(file_names)
    Y.extend(labels)

print(len(X) , len(Y))

#split the data into train and test
# x = train_test_split(X ,y, test_size= 0.2 , random_state = 42)
# print(x.__len__())

X_train , X_test , y_train , y_test = train_test_split(X ,Y, test_size= args.test_size, random_state = args.random_state)
print('X test size :' , len(X_test))
print('Y test size :' , len(y_test))








#save the train and test sets as csv

save_as_csv(X_train, y_train, 'data/train.csv')
save_as_csv(X_test, y_test, 'data/tests.csv')



