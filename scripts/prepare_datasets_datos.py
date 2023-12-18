import csv
import glob
import os

import pandas as pd

BASE_DIR="data\Datos"
TRAIN_DIR="data\Datos\Training-validation"
TEST_DIR="data\Datos\Testing"


train_images=glob.glob(f"{TRAIN_DIR}/**/*.jpg")
test_images=glob.glob(f"{TEST_DIR}/**/*.jpg")

train_labels=[image_path.split('\\')[-2]for image_path in train_images]
test_labels=[image_path.split('\\')[-2]for image_path in test_images]

# train_data = pd.DataFrame({'image_path': train_images, 'labels': train_labels})
# test_data = pd.DataFrame({'image_path': test_images, 'labels': test_labels})

# # Save dataframes to CSV files
# train_data.to_csv(os.path.join(BASE_DIR, 'train.csv'), index=False)
# test_data.to_csv(os.path.join(BASE_DIR, 'test.csv'), index=False)

# Writing to train.csv
with open(os.path.join(BASE_DIR, 'train.csv'), 'w', newline='') as train_csv_file:
    writer = csv.writer(train_csv_file)
    writer.writerow(['image_path', 'labels'])  # Writing header
    for image, label in zip(train_images, train_labels):
        writer.writerow([image, label])

# Writing to test.csv
with open(os.path.join(BASE_DIR, 'test.csv'), 'w', newline='') as test_csv_file:
    writer = csv.writer(test_csv_file)
    writer.writerow(['image_path', 'labels'])  # Writing header
    for image, label in zip(test_images, test_labels):
        writer.writerow([image, label])


