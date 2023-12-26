import joblib
from config import MODEL_CHECKPOINT_PATH 
from utils.io import read_as_csv,save_prediction_as_csv
import os
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
from utils.preprocessing import image_transforms,label_transforms,label_to_idx_map, idx_to_label
from sklearn.metrics import classification_report
from visualization import display_grid
from train import train
from metrics.accuracy import accuracy
from metrics.confusion import confusion_metrics_calculate
from sklearn.neighbors import NearestNeighbors
from argparse import ArgumentParser



from utils.load_config import config_load

def get_prediction(model_checkpoint,test_file,out_dir):
    loaded_model = joblib.load(model_checkpoint)
    test_files, test_labels = read_as_csv("data/test.csv")
    y_test = np.array([label_transforms(lab) for lab in test_labels])
    X_test = np.array(
    [image_transforms(file, label) for file, label in zip(test_files, test_labels)]
)
    outfile=configs["evaluation"]["out_dir"]
    y_pred = loaded_model.predict(X_test)
    save_prediction_as_csv(test_files,y_test,y_pred,outfile)
    print("saved")




# # Compute accuracy
# acc = accuracy(y_test, y_pred)
# print("Accuracy:", acc)


# print(classification_report(y_test, y_pred, target_names=label_to_idx_map.keys()))

# # Compute confusion matrix
# cm = confusion_metrics_calculate(y_test, y_pred)
# print("Confusion Matrix:\n", cm)

# cm1 = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cm1)

# # Display confusion matrix

# unique_labels = label_to_idx_map.keys()
# disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=unique_labels)
# disp.plot()
# plt.show()

# # Compute precision, recall, and F1-score
# precision = precision_score(y_test, y_pred, average='macro')  # 'macro' averages the scores for each class
# recall = recall_score(y_test, y_pred, average='macro')
# f1 = f1_score(y_test, y_pred, average='macro')

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)

# # Compute specificity (True Negative Rate)
# specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
# print("Specificity (True Negative Rate):", specificity)

# # Compute Cohen's Kappa 
# cohen_kappa = cohen_kappa_score(y_test, y_pred)
# print("Cohen's Kappa:", cohen_kappa)


# DATA_DIRS='data/middle-ear-dataset'
# display_grid(DATA_DIR=DATA_DIRS,image_files=test_files,actual_labels=test_labels,
#             predicted_labels=y_pred_labels,
#              n_rows=4,
#              n_cols=3,
#              title='Otitis_media')


if __name__=="__main__":
    parser = ArgumentParser(
        prog="test",
        description="Script to test model",
    )
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    configs=config_load(args.config)
    checkpoint_path=configs["evaluation"]["model_checkpoint"]
    # data_root=configs["evaluation"]["data_root"]
    # test_csv=configs["evaluation"]["test_csv"]
    test_csv_path=os.path.join(configs["evaluation"]["data_root"],configs["evaluation"]["test_csv"])
    out_dir=configs["evaluation"]["out_dir"]
    get_prediction(model_checkpoint=checkpoint_path,test_file=test_csv_path,out_dir=out_dir)




   
