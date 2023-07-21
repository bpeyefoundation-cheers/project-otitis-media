import joblib
from config import MODEL_CHECKPOINT_PATH 
from utils.io import read_as_csv
from train import X_test,Y_test
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt

#load the model
loaded_knn_model = joblib.load(MODEL_CHECKPOINT_PATH)
#test filenames,labels
test_files, test_labels = read_as_csv("data/test.csv")
#predict
# print(loaded_knn_model.predict(X_test))

#print score
print("Test score:", loaded_knn_model.score(X_test, Y_test))
#predict
y_pred=loaded_knn_model.predict(X_test)
#predicted label
#print(y_pred)
#actual label
#print(Y_test)
#print accuracy score
cm=confusion_matrix(Y_test, y_pred)
print("accuracy:",accuracy_score(Y_test, y_pred))
print("confusion_matrix:\n",cm)
#display
unique_labels = sorted(set(test_labels))
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=unique_labels)
disp.plot()
plt.show()