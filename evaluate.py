import joblib
from config import MODEL_CHECKPOINT_PATH 
from utils.io import read_as_csv
import matplotlib.pyplot as plt
from utils.preprocessing import image_transforms,  label_transforms
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, cohen_kappa_score
from utils.preprocessing import label_map , index_to_label
from sklearn.metrics import classification_report
from viz.visualization import display_grid
from metrics.accuracy import accuracy
from utils.load_config import config_loader
import os
from utils.io import save_prediction_as_csv


# # Load the model
# loaded_model = joblib.load(MODEL_CHECKPOINT_PATH)

# # Test filenames, labels
# test_files, test_labels = read_as_csv("data/tests.csv")

# X_test = np.array([image_transforms(file_name , label) for file_name , label in zip(test_files, test_labels)])
# Y_test = np.array([label_transforms(label) for label in test_labels])


# # Predict
# y_pred = loaded_model.predict(X_test)
# y_pred_labels = np.array([index_to_label(idx) for idx in y_pred])


def get_prediction(model_checkpoint, test_file , out_dir):
    
    
    loaded_model = joblib.load(model_checkpoint)
    image_path, test_labels = read_as_csv(test_file)
    test_files, test_labels = read_as_csv("data/tests.csv")
    Y_test = np.array([label_transforms(label) for label in test_labels])
    X_test = np.array([image_transforms(file_name , label) for file_name , label in zip(image_path, test_labels)])
    y_pred = loaded_model.predict(X_test)
    outfile = configs["evaluation"]["out_dir"]
    save_prediction_as_csv(image_path , Y_test, y_pred, outfile' )





# # Compute accuracy
# accuracy_calc = accuracy(Y_test, y_pred)
# print("Accuracy:", accuracy_calc)

# # Compute confusion matrix
# cm = confusion_matrix(Y_test, y_pred)
# print("Confusion Matrix:\n", cm)

# # Display confusion matrix
# labels = label_map.keys()
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= labels)
# # disp.plot()
# # plt.show()

# # Compute precision, recall, and F1-score
# precision = precision_score(Y_test, y_pred, average='macro')  # 'macro' averages the scores for each class
# recall = recall_score(Y_test, y_pred, average='macro')
# f1 = f1_score(Y_test, y_pred, average='macro')

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)

# # Compute specificity (True Negative Rate)
# specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
# print("Specificity (True Negative Rate):", specificity)

# # Compute Cohen's Kappa 
# cohen_kappa = cohen_kappa_score(Y_test, y_pred)
# print("Cohen's Kappa:", cohen_kappa)

 
# print(classification_report(Y_test, y_pred, target_names= labels))


# DATA_DIR= "data/middle-ear-dataset"

# display_grid(image_dir=DATA_DIR, images = test_files , actual_labels= test_labels  , predicted_label= y_pred_labels, n_rows= 4 , n_cols= 3, title= 'Otitis_media'  )


if __name__ == "__main__":
    configs = config_loader("configs/test.yaml")["evaluation"]
    get_prediction(model_checkpoint= configs["model_checkpoint"] , test_file= configs["test_data_path"] )