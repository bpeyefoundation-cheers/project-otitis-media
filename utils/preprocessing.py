def label_to_idx(label:str): 
	"""convert label name to index 
	for example: if my dataset consists of three labels (setosa, virginica, versicolor) 
	this function should return 0 for setosa, 1 for virginica, 2 for versicolor 
	"""
	label_map = {'aom': 0, 'csom': 1, 'myringosclerosis': 2,'Normal':3}
	return label_map.get(label)
	
    

def idx_to_label(idx:int): 
	""" similiar as label_to_idx but opposite I.e. take the index and return the string label """
	label_map = {0: 'aom', 1: 'csom', 2: 'myringosclerosis',3: 'Normal'}
	return label_map.get(idx)
    
label = 'aom'
label_idx = label_to_idx(label)
print(label_idx)  # Output: 1

idx = 2
idx_label = idx_to_label(idx)
print(idx_label)  # Output: 'versicolor'