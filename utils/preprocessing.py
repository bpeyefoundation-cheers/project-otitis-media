from PIL import Image,ImageOps
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.io import list_files,get_image_label_pairs
from os.path import join

data_root = "data\middle-ear-dataset"
label_to_idx_map = {'aom': 0, 'csom': 1, 'myringosclerosis': 2,'Normal':3}
rev_label_to_idx_map={index:label for label,index in label_to_idx_map.items() }




def label_transforms(label) -> int:
    # label_to_index
    return label_to_idx(label)

def read_image(file_path: str, mode:str,resize=(256,256), grayscale:bool = False) -> np.ndarray:
    """
    image_path:str file_path of the image which we want
    mode:str of either preprocessing or zoom.The image is either zoomed in or padded a the border
    """
    
        
    image = Image.open(file_path)
   
    if grayscale:
        image = image.convert("L")

    image = Image.open(file_path)
    # print(image.size)
    height,width=image.size
    if height==width:
         pass
    else:
        
        if mode=='zoom':
            # left=0
            # right=0
            # upper=0
            # lower=0
            if height<width:
                diff=width-height
                left=diff//2
                right=width-diff//2
                top=0
                bottom=height
            elif width<height:
                diff=height-width
                left=0
                right=width
                upper=diff//2
                lower=height-upper


            new_image=image.crop((left,right,upper,lower))
        elif mode=='padding':
        
            image=ImageOps.pad(image, size=(256,256), centering=(0.5, 0.5))
            
    new_image=image.resize(resize)
    img_array = np.asarray(new_image)
    return img_array
# Transforms
def image_transforms(file_name, label) -> np.ndarray:
    """ this function extract the exact image from the image file name and label provided
    file_name:string,label:string
    """
    file_path = os.path.join(data_root, label, file_name)
    array = read_image(file_path, "zoom", grayscale=True)
    flatten_image = array.flatten()
    return flatten_image    
            

def label_to_idx(label:str):
    """convert label name to index
	for example: if my dataset consists of three labels (aom, csom, myringosclerosis,normal)
	this function should return 0 for aom, 1 for csom, 2 for myringosclerosis,normal
	"""
    """label represent class name in otitis media and index represent the corresponding class

    Raises:
        KeyError: _description_

    Returns:
        _type_: _description_
    """
    if label not in label_to_idx_map:
        raise KeyError(f"label not define . Defined label are:{label_to_idx_map.keys()}")
    return label_to_idx_map[label]   
    
def idx_to_label(idx:int):
        """ similiar as label_to_idx but opposite I.e. take the index and return the string label """
        try:
            return rev_label_to_idx_map[idx]
        except KeyError:
            raise KeyError(f"Label not found. Try one of these: {rev_label_to_idx_map.keys()}")
    


if __name__ == "__main__":
    
    #check for label to index
    label='aom'
    label_idx=label_to_idx(label)
    print(label_idx)

    # check for index to label
    index=0
    idx_label=idx_to_label(index)
    print(idx_label)

    # DATA_DIR = "data/middle-ear-dataset/csom"
    # image_files = os.listdir(DATA_DIR)
    # n_rows=4
    # n_cols=3
    # LABEL='csom'
    # file_path,labels=get_image_label_pairs(DATA_DIR,LABEL)
    # # Create a 4x3 grid of subplots
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
			
# display_grid(DATA_DIR,file_path,labels,n_rows,n_cols,'middle_ear')
    