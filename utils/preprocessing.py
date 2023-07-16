from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.io import list_files,get_image_label_pairs
from os.path import join



def read_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    img_array = np.asarray(image)
    return img_array

def display_grid(DATA_DIR,image_files,labels,n_rows,n_cols,title,figsize=(10,10)):
    """Display grid of images with their labels
    """
    fig,axes=plt.subplots(n_rows,n_cols,figsize=figsize)
    fig.suptitle(title,fontsize=16)
    idx=0
    for i in range(n_rows):
        for j in range(n_cols):
            file_path=os.path.join(DATA_DIR,image_files[idx])
            img_arr=read_image(file_path)
            
			  # Display the image in the current subplot
            axes[i, j].imshow(img_arr)
            axes[i, j].axis("off")
            axes[i,j].set_title(f'Label:{labels[idx]}')
            idx+=1
            # Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()

            
            
  


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

    DATA_DIR = "data/middle-ear-dataset/aom"
    image_files = os.listdir(DATA_DIR)
    n_rows=4
    n_cols=3
    file_path,labels=get_image_label_pairs(DATA_DIR,'aom')
    # Create a 4x3 grid of subplots
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 5))
    # idx=0
    # counter = 0
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         # if counter == 12:
    #         #     break
    #         file_name = image_files[counter]
    #         file_path = os.path.join(DATA_DIR, file_name)
    #         image = plt.imread(file_path)

    #         # Display the image in the current subplot
    #         axes[i, j].imshow(image)
    #         axes[i, j].axis("off")
    #         axes[i,j].set_title(f'Label:aom{idx}')
    #         axes[i,j].set_title(f'{i*3+j}')
    #         print(i*3+j)
    #         idx+=1

            # counter += 1
			
display_grid(DATA_DIR,file_path,labels,n_rows,n_cols,'middle_ear')
    