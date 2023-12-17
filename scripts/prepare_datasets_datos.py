import glob
import os
import csv
import pandas as pd

BASE_DIR="data\Ear_Domain\Datos"
TRAIN_DIR="data\Ear_Domain\Datos\Training-validation"
TEST_DIR="data\Ear_Domain\Datos\Testing"


train_images=glob.glob(f"{TRAIN_DIR}/**/*.jpg")
test_images=glob.glob(f"{TEST_DIR}/**/*.jpg")

train_labels = [image_path.split("\\")[-2] for image_path in train_images]
test_labels = [image_path.split("\\")[-2] for image_path in test_images]

print(train_images)
# def get_imagepath_labels(train_images,test_images,train_labels,test_labels):
#     train_images=glob.glob(f"{TRAIN_DIR}/**/*.jpg")
#     test_images=glob.glob(f"{TEST_DIR}/**/*.jpg")

#     train_labels = [image_path.split("\\")[-2] for image_path in train_images]
#     test_labels = [image_path.split("\\")[-2] for image_path in test_images]

#     return train_images,test_images,train_labels,test_labels


# train_data = pd.DataFrame({'image_path': train_images, 'labels': train_labels})
# test_data = pd.DataFrame({'image_path': test_images, 'labels': test_labels})

# # Save dataframes to CSV files
# train_data.to_csv(os.path.join(BASE_DIR, 'train.csv'), index=False)
# test_data.to_csv(os.path.join(BASE_DIR, 'test.csv'), index=False)



# def read_as_csv(csv_file):
#     train_image_path= []
#     train_labels= [] 
#     with open(csv_file , 'r') as f:
#         reader = csv.reader(f)
#         next(reader)
#         for row in reader:
#            train_image_path.append(row[0])
#            train_labels.append(row[1])
#     return train_image_path, train_labels


# # Example usage
# TRAIN_CSV = 'data\Ear_Domain\Datos\train.csv'
# TEST_CSV = 'data\Ear_Domain\Datos\test.csv'

# train_images, test_images, train_labels, test_labels = get_imagepath_labels(TRAIN_CSV, TEST_CSV)
