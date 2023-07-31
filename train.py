import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.io import read_as_csv
from config import MODEL_CHECKPOINT_PATH
from utils.preprocessing import image_transforms,  label_transforms
import joblib
import os
from utils.load_config import config_loader
from models.models import ModelZoo

def train(data_root, train_csv, test_csv, model , checkpoint_path):


    # load csv
    train_path = os.path.join(data_root , train_csv)
    tests_path = os.path.join(data_root, test_csv)
    train_files , train_labels = read_as_csv(train_path)
    test_files , test_labels = read_as_csv(tests_path)

    #apply the image tranform function to train and test file
    X_train = np.array([image_transforms(file_name , label) for file_name , label in zip(train_files, train_labels)])
    Y_train = np.array([label_transforms(label) for label in train_labels])




    clf = model
    clf.fit(X_train , Y_train)

    # clf.predict(X_predict)

    # pred = clf.predict(X_predict)


    # print("pred:" , [index_to_label[p] for p in pred])

    # dict_proba = []

    # for proba in clf.predict_proba(X_predict):
    #     dict_proba.append({index_to_label[i]: p for i, p in enumerate(proba)})

    # print("pred prob :" , dict_proba)
    # print("Train score", clf.score(X_train, Y_train))

    # Save the model
    joblib.dump(clf, checkpoint_path)

if __name__ == "__main__":
    configs = config_loader("configs\config.yaml")
    # print(configs)
    model = ModelZoo(**configs["model"]).get_model()  
    train(data_root=configs["data_root"], train_csv=configs["train_csv"], test_csv=configs["test_csv"] ,model= model, checkpoint_path= configs["checkpoint_path"])