import glob
import os

import pandas as pd
# from utils.io import save_as_csv

BASE_DIR="data\Datos\Datos"
TRAIN_DIR="data\Datos\Datos\Training-validation"
TEST_DIR="data\Datos\Datos\Testing"


train_images=glob.glob(f"{TRAIN_DIR}/**/*.jpg")
test_images=glob.glob(f"{TEST_DIR}/**/*.jpg")
# print(train_images)
#print(test_images)

train_labels = [image_path.split("\\")[4] for image_path in train_images]
test_labels = [image_path.split("\\")[4] for image_path in test_images]

# # print(train_labels)

train_data = {'Image': train_images, 'Label': train_labels}
test_data = {'Image': test_images, 'Label': test_labels}

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# save_as_csv(train_images , train_labels , 'data/Datos/Datos/train.csv')
# save_as_csv(test_images , test_labels , 'data/Datos/Datos/test.csv')















