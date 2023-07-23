import joblib
from utils.config import MODEL_CHECKPOINT_PATH 
from utils.io import read_as_csv
from train import X_test, Y_test
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
from utils.pre_processing import label_map, index_to_label
from viz.visualization import display_grid
import numpy as np

# Load the model
loaded_knn_model = joblib.load(MODEL_CHECKPOINT_PATH)

# Test filenames, labels
test_files, test_labels = read_as_csv("data/test.csv")

# Predict
y_pred = loaded_knn_model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(Y_test, y_pred)
#print("Accuracy:", accuracy)

# Compute confusion matrix
cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Display confusion matrix
labels = label_map.keys()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#disp.plot()
#plt.show()

# Compute precision, recall, and F1-score
precision = precision_score(Y_test, y_pred, average='macro')  # 'macro' averages the scores for each class
recall = recall_score(Y_test, y_pred, average='macro')
f1 = f1_score(Y_test, y_pred, average='macro')

#print("Precision:", precision)
#print("Recall:", recall)
#print("F1-score:", f1)

# Compute specificity (True Negative Rate)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#print("Specificity (True Negative Rate):", specificity)

# Compute Cohen's Kappa 
cohen_kappa = cohen_kappa_score(Y_test, y_pred)
#print("Cohen's Kappa:", cohen_kappa)

#print(classification_report(Y_test, y_pred, target_names=labels))


DATA_DIR ='data/middle-ear-dataset'
predicted_label = np.array([index_to_label(i) for i in y_pred])

 
display_grid(image_dir=DATA_DIR, images=test_files, labels=test_labels,predicted_label=predicted_label, n_rows=3, n_cols=3, title='asom') 
    