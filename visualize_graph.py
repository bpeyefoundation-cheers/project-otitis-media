import joblib
import numpy as np
from config import MODEL_CHECKPOINT_PATH
from utils.preprocessing import image_transforms,label_transforms
from utils.io import read_as_csv
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#from evaluate import display_grid
# Load the model
loaded_knn_model = joblib.load(MODEL_CHECKPOINT_PATH)


train_files, train_labels = read_as_csv("data/train.csv")
test_files, test_labels = read_as_csv("data/test.csv")


X_test = np.array(
    [image_transforms(file, label) for file, label in zip(test_files, test_labels)]
)
X_train = np.array(
    [image_transforms(file, label) for file, label in zip(train_files, train_labels)]
)
y_test = np.array([label_transforms(lab) for lab in test_labels])
neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(X_test, y_test)

indices_of_neighbors=loaded_knn_model.kneighbors(X_test[:4],n_neighbors=3,return_distance =False)
print(indices_of_neighbors)



# def display_images(train_files, indices_of_neighbors):
#     for i, neighbors in enumerate(indices_of_neighbors):
#         print(f"Neighbors for test sample {i + 1}:")
#         fig, axs = plt.subplots(1, len(neighbors), figsize=(10, 5))
#         for j, neighbor_index in enumerate(neighbors):
#             image_path = train_files[neighbor_index]
#             image = plt.imread(image_path)
#             axs[j].imshow(image)
#             axs[j].set_title(f"Neighbor {j + 1}")
#             axs[j].axis('off')
#         plt.show()



# display_images(train_files, indices_of_neighbors[:4])
# images=[]
# labels=[]  
# for i, neighbors in enumerate(indices_of_neighbors):
#     images.append(X_test)        

#             # Adjust the spacing between subplots
#     plt.tight_layout()
#     plt.show()





fig, axs = plt.subplots(4,4,figsize=(10, 10))

for i in range(4):

    axs[i, 0].imshow(X_test[i].reshape(256,256,3))
    axs[i, 0].set_title(f"Test Image:{train_labels[i]}")
    axs[i, 0].axis('off')

    for j, idx in enumerate(indices_of_neighbors[i]):

        axs[i, j+1].imshow(X_train[idx].reshape(256,256,3))
        axs[i, j+1].set_title(f"Neighbor:{train_labels[idx]}")
        axs[i, j+1].axis('off')

plt.suptitle("Nearest Neighbors Images", fontsize=10)
plt.tight_layout()
plt.show()
