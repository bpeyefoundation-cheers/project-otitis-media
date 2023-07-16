from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.io import list_files
from os.path import join

def read_image(image_path:str) ->np.ndarray:
	image=Image.open(image_path)
	img_array=np.asarray(image)
	return img_array



# def label_to_idx(label:str): 
# 	"""convert label name to index 
# 	for example: if my dataset consists of three labels (setosa, virginica, versicolor) 
# 	this function should return 0 for setosa, 1 for virginica, 2 for versicolor 
# 	"""
# 	label_map = {'aom': 0, 'csom': 1, 'myringosclerosis': 2,'Normal':3}
# 	return label_map.get(label)
	
    

# def idx_to_label(idx:int): 
# 	""" similiar as label_to_idx but opposite I.e. take the index and return the string label """
# 	label_map = {0: 'aom', 1: 'csom', 2: 'myringosclerosis',3: 'Normal'}
# 	return label_map.get(idx)
    
if __name__ == "__main__":
    # IMAGE_PATH=r'C:\Users\Dell\Desktop\projects\project-otitis-media\data\middle-ear-dataset\csom\o13.jpg'
    # image=read_image(IMAGE_PATH)

    # remove xticks and yticks
    # plt.xticks([])
    # plt.yticks([])
    # set title

    DATA_DIR = 'data/middle-ear-dataset/aom'
    image_files = os.listdir(DATA_DIR)

# Create a 4x3 grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(10, 10))

    counter = 0
    for i in range(4):
        for j in range(3):
            if counter == 12:
                break
            file_name = image_files[counter]
            file_path = os.path.join(DATA_DIR, file_name)
            image = plt.imread(file_path)

        # Display the image in the current subplot
            axes[i, j].imshow(image)
            axes[i, j].axis('off')

            counter += 1

# Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()
