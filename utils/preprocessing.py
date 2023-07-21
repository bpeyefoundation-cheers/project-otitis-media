from PIL import Image,ImageOps
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.io import list_files,get_image_label_pairs
from os.path import join




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

def display_grid(DATA_DIR,image_files,labels,n_rows,n_cols,title,figsize=(10,10)):
    """Display grid of images with their labels
    """
    fig,axes=plt.subplots(n_rows,n_cols,figsize=figsize)
    fig.suptitle(title,fontsize=16)
    idx=0
    for i in range(n_rows):
        for j in range(n_cols):
            file_path=os.path.join(DATA_DIR,image_files[idx])
            img_arr=read_image(file_path,mode='padding')
            
			  # Display the image in the current subplot
            axes[i, j].imshow(img_arr)
            axes[i, j].axis("off")
            axes[i,j].set_title(f'Label:{labels[idx]}')
            idx+=1
            # Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()

            
            
 
label_to_idx_map = {'aom': 0, 'csom': 1, 'myringosclerosis': 2,'Normal':3}
rev_label_to_idx_map={index:label for label,index in label_to_idx_map.items() }

def label_to_idx(label:str):
        """convert label name to index
	for example: if my dataset consists of three labels (aom, csom, myringosclerosis,normal)
	this function should return 0 for aom, 1 for csom, 2 for myringosclerosis,normal
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

# def func(default=None):
#     try:
#         return label_to_idx_map[label]
#     except KeyError:
#          return default

if __name__ == "__main__":
    # IMAGE_PATH=r'C:\Users\Dell\Desktop\projects\project-otitis-media\data\middle-ear-dataset\csom\o13.jpg'
    # image=read_image(IMAGE_PATH)

    # remove xticks and yticks
    # plt.xticks([])
    # plt.yticks([])
    # set title

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
    