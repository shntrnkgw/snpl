# coding=utf-8
'''
Created on 2018/02/24

@author: snakagawa
'''

import numpy as np
from snpl import __version__

class BaseImage(object):
    '''
    A base class for n-dimensional data (n > 1). 
    
    Difference from BaseData:
    - Multiple n-d data are now handled as "layers" instead of columns. 
    - All layers should have identical dimension and shape (data type may be varied). 
    - No filtering functions.
    - No order of layers; layers are stored in a dictionary. 
    
    *NOTE* There is no "GenericImage" class because a generic format of image data 
    does not exist, unlike the situation in 1-d data where CSV format is 
    frequently used in many applications. 
    '''
    
    def __init__(self):
        
        self.layers = {}
    
    #-----#
    # Get #
    #-----#
    def get_layer(self, key):
        return self.layers[key]
    
    #---------#
    # Editing #
    #---------#
    def append_layer(self, key, arr):
        if self.layers:
            for l in self.layers.values():
                if len(l.shape) == len(arr.shape): # check dimensionality
                    for size1, size2 in zip(l.shape, arr.shape):
                        if size1 != size2:         # check size in each dimension
                            raise ValueError("Cannot append data with different shape. ")
                else:
                    raise ValueError("Cannot append data with different dimension. ")
        
        self.layers[key] = arr
    
    def pop_layer(self, key):
        return self.layers.pop(key)
    

class NpzImage(BaseImage):
    
    def __init__(self, fp=None):
        BaseImage.__init__(self)
        
        h = {}
        layers = {}
        
        
        if fp:
            with np.load(fp, allow_pickle=True) as z:
                for key, arr in z.items():
                    if key == "h":
                        h = arr
                    else:
                        layers[key] = arr
            
            h = h[()]
        else:
            h = {}
            layers = {}
                
        self.h = {k: v for k, v in h.items()}
        self.layers = layers
    
    def write(self, fp, compress=False):
        h = {k: v for k, v in self.h.items()}
        h["version"] = __version__
        if compress:
            np.savez_compressed(fp, h=h, **self.layers)
        else:
            np.savez(fp, h=h, **self.layers)

    def append_history(self, string):
        try:
            self.h["history"].append(string)
        except KeyError:
            self.h["history"] = [string, ]
                

if __name__ == '__main__':
    
    im = NpzImage()
    im.h = {"number": 100.0, "string": "wow", "bool": True, "list": [1.0, 2.0]}
    im.append_layer("one", np.array( [[1.0, 2.0], [3.0, 4.0]] ) )
    im.append_layer("two", np.array( [[5.0, 6.0], [7.0, 8.0]] ) )
    
    im.write("test/npzimage.npz")
    
    im2 = NpzImage("test/npzimage.npz")
    print(im2.h)
    print(im2.layers["one"])
    print(im2.layers["two"])
    