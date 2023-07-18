from PIL import Image , ImageOps
from os.path import join
from utils.preprocessing import read_image
import matplotlib.pyplot as plt
from collections import OrderedDict


def display_grid(image_dir:str, images:list, labels:list, n_rows:int, n_cols:int, title:str, fig_size:tuple=(10,10)):
      """display grid of images with their labels"""
    #   image_paths= list_files(image_dir)
    #   no_of_images= n_rows * n_cols
    #   new_image_list= image_paths[0 : no_of_images]
      fig , ax = plt.subplots(n_rows, n_cols , figsize=fig_size) 
      fig.suptitle(title)
      index = 0
      for i in range(n_rows):
        for j in range(n_cols):
                
                # image= new_image_list[index]
                data_path= join(image_dir, labels[index], images[index])
               
                image_array = read_image(data_path , 'zoom')
                ax[i][j].imshow(image_array)
                
                ax[i,j].axis('off')
                ax[i,j].set_title(labels[index])
                
                
                index += 1
      plt.show()


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

index_value = label_to_index('aom')
print(index_value)

label_value = index_to_label(3)
print(label_value)