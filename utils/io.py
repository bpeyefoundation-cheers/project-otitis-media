import os
def list_files(dir:str , file_extension:str) ->list:
    
    if not os.path.isdir(dir):
        return None
    
    files = os.listdir(dir)
    return files