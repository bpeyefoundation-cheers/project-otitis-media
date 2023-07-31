import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.io import read_as_csv
from utils.preprocessing import image_transforms, label_transforms
import os.path
import joblib
from config import MODEL_CHECKPOINT_PATH
from utils.load_config import config_load
from models.models import ModelZoo

def train(data_root,train_csv,test_csv,model,checkpoint_path):
    train_path=os.path.join(data_root,train_csv)
    test_path=os.path.join(data_root,test_csv)
    

    # load csv
    train_files, train_labels = read_as_csv(train_path)
    # test_files, test_labels = read_as_csv("data/test.csv")

    # Apply the image_transforms function to train_files and test_files
    X_train = np.array(
        [image_transforms(file, label) for file, label in zip(train_files, train_labels)]
    )
    # print(X_train)
    Y_train = np.array([label_transforms(lab) for lab in train_labels])


    clf = model
    clf.fit(X_train, Y_train)

    # Save the model
    joblib.dump(clf, checkpoint_path)

if __name__=="__main__":
    configs=config_load("configs\config.yaml")
    model=ModelZoo(**configs["model"]).get_model()
 
    train(data_root=configs["data_root"],train_csv=configs["train_csv"],test_csv=configs["test_csv"],
          model=model,
          checkpoint_path=configs["checkpoint_path"])
    
    