import joblib
from config import MODEL_CHECKPOINT_PATH 
from utils.io import read_as_csv
import os
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
from utils.preprocessing import image_transforms,label_transforms,label_to_idx_map, idx_to_label
from sklearn.metrics import classification_report
from visualization import display_grid
from train import X_train,Y_train
from metrics.accuracy import accuracy
from metrics.confusion import confusion_metrics_calculate
# Load the model
loaded_knn_model = joblib.load(MODEL_CHECKPOINT_PATH)

# Test filenames, labels
test_files, test_labels = read_as_csv("data/test.csv")
X_test = np.array(
    [image_transforms(file, label) for file, label in zip(test_files, test_labels)]
)
y_test = np.array([label_transforms(lab) for lab in test_labels])
y_test_label=np.array([ idx_to_label(p) for p in y_test])
# Predict
y_pred = loaded_knn_model.predict(X_test)

y_pred_labels=np.array([ idx_to_label(p) for p in y_pred])

#print(set(y_pred_labels))

# Compute accuracy
acc = accuracy(y_test, y_pred)
print("Accuracy:", acc)


print(classification_report(y_test, y_pred, target_names=label_to_idx_map.keys()))

# Compute confusion matrix
cm = confusion_metrics_calculate(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Display confusion matrix

unique_labels = label_to_idx_map.keys()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
# disp.plot()
# plt.show()

# Compute precision, recall, and F1-score
precision = precision_score(y_test, y_pred, average='macro')  # 'macro' averages the scores for each class
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Compute specificity (True Negative Rate)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print("Specificity (True Negative Rate):", specificity)

# Compute Cohen's Kappa 
cohen_kappa = cohen_kappa_score(y_test, y_pred)
print("Cohen's Kappa:", cohen_kappa)
DATA_DIRS='data/middle-ear-dataset'
display_grid(DATA_DIR=DATA_DIRS,image_files=test_files,actual_labels=test_labels,
            #predicted_labels=y_pred_labels,
             n_rows=4,
             n_cols=3,
             title='Otitis_media')


