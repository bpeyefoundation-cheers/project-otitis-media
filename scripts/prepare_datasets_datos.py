import glob
import os
import pandas as pd
# from utils.io import save_as_csv

BASE_DIR="data\Datos"
TRAIN_DIR="data\Datos\Training-validation"
TEST_DIR="data\Datos\Testing"


train_images=glob.glob(f"{TRAIN_DIR}/**/*.jpg")
# print(train_images)
test_images=glob.glob(f"{TEST_DIR}/**/*.jpg")

# print(test_images)


train_labels = [image_path.split("\\")[-2] for image_path in train_images]
test_labels = [image_path.split("\\")[-2] for image_path in test_images]

# print(test_labels)

train_data = pd.DataFrame({'image_path': train_images, 'labels': train_labels})
test_data = pd.DataFrame({'image_path': test_images, 'labels': test_labels})

# print(test_data)

# Save dataframes to CSV files
train_data.to_csv(os.path.join(BASE_DIR, 'train.csv'), index=False)
test_data.to_csv(os.path.join(BASE_DIR, 'test.csv'), index=False)

