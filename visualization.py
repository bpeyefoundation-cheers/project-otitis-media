import matplotlib.pyplot as plt
from utils.preprocessing import read_image
from os.path import join
import os
def display_grid(DATA_DIR,image_files,actual_labels,n_rows,n_cols,title,figsize=(8,8),predicted_labels=None):
    """Display grid of images with their labels
    """
    fig,axes=plt.subplots(n_rows,n_cols,figsize=figsize)
    fig.suptitle(title,fontsize=10)
    fig.tight_layout(pad=10)
    idx=0
    for i in range(n_rows):
        for j in range(n_cols):
            file_path=os.path.join(DATA_DIR,actual_labels[idx], image_files[idx])
            img_arr=read_image(file_path,mode='padding')
            
			  # Display the image in the current subplot
            axes[i, j].imshow(img_arr)
            axes[i, j].axis("off")
            if predicted_labels is not None :
                axes[i,j].set_title(f'True:{actual_labels[idx]}\npredicted:{predicted_labels[idx]}',fontsize=8)
                
            else:
                axes[i,j].set_title(f'True:{actual_labels[idx]}',fontsize=8)
        idx+=1

            # Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()
    
    
