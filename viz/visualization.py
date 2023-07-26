from PIL import Image , ImageOps
from os.path import join
from utils.preprocessing import read_image
import matplotlib.pyplot as plt
from collections import OrderedDict


def display_grid(image_dir:str, images:list, actual_labels:list ,  n_rows:int, n_cols:int, title:str, predicted_label: list = None , fig_size:tuple=(10,10)):
      """display grid of images with their labels"""
    #   image_paths= list_files(image_dir)
    #   no_of_images= n_rows * n_cols
    #   new_image_list= image_paths[0 : no_of_images]
      fig , ax = plt.subplots(n_rows, n_cols , figsize=fig_size) 
      fig.suptitle(title)
      fig.tight_layout(pad=4)
      index = 0
      for i in range(n_rows):
        for j in range(n_cols):
                
                # image= new_image_list[index]
                data_path= join(image_dir, actual_labels[index], images[index])
               
                image_array = read_image(data_path , 'zoom')
                ax[i][j].imshow(image_array)
                
                ax[i,j].axis('off')
                if predicted_label.any == None :
                     ax[i,j].set_title(f'True:{actual_labels[index]}')
                else :
                     ax[i,j].set_title(f'True:{actual_labels[index]} \n predicted:{predicted_label[index]}')
                
                
                index += 1
      plt.show()


