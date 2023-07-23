from PIL import Image , ImageOps
import numpy as np
from utils.io import read_as_csv
import matplotlib.pyplot as plt
from os.path import join
# from viz.visualization import display_grid
import os



label_map = {
     "aom" : 0 , 
     "csom" : 1 , 
     "myringosclerosis" : 2 , 
     "Normal" : 3
}

rev_label_to_index_map = {index : label for label , index in label_map.items()}


def label_to_index(label : str):
     if label not in label_map :
          raise KeyError("label not defined")
     return label_map[label]
     

def index_to_label(index : int):
     if index not in rev_label_to_index_map :
          raise KeyError("index not defined")
     return rev_label_to_index_map[index]



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

#transforms

data_root = "data\middle-ear-dataset"
def image_transforms(file_name , label) -> np.ndarray :
    file_path = os.path.join(data_root , label, file_name)
    array = read_image(file_path, mode="zoom")
    flatten_image = array.flatten()
    return flatten_image

def label_transforms( label) -> int :
    #label to index 
    return label_to_index(label)


if __name__ == "__main__" :
    

    DATA_DIR= "data/middle-ear-dataset"
    n_rows=3
    n_cols= 3
    LABEL= "middle-ear-dataset"
    image_path, labels= read_as_csv("data/tests.csv")
    
    #get_image_label_pairs(DATA_DIR, LABEL)
    
    # display_grid(DATA_DIR, image_path , labels , n_rows, n_cols ,LABEL)
  




    
