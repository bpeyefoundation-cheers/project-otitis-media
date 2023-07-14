import os
from os.path import  join
from utils.io import get_image_label_pairs , save_as_csv
from sklearn.model_selection import train_test_split


DATA_DIR = "data\middle-ear-dataset"

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

X_train , X_test , y_train , y_test = train_test_split(X ,Y, test_size= 0.2 , random_state = 42)
print('X test size :' , len(X_test))
print('Y test size :' , len(y_test))


#save the train and test sets as csv

save_as_csv(X_train, y_train, 'data/train.csv')
save_as_csv(X_test, y_test, 'data/tests.csv')



