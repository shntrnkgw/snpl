# coding=utf-8
"""I/O interface for a generic multilayer array collection
"""

import numpy as np
from snpl import __version__    

class NpzImage:
    """I/O interface for NpzImage file. 
    
    NpzImage is a convenient file format to store 
    multi-layered multi-dimensional arrays with a metadata header. 
    Multiple ``numpy.ndarray`` objects with the same shape can be stored 
    as "layers", which can be specified by a "key". 
    All layers should have identical dimension and shape, but data type may be varied. 

    Args:
        fp (str or file-like): Path or file-like object of the source file. If None, an empty object is created. 

    Examples:
        Creation

        >>> im = snpl.image.NpzImage()
        
        Adding headers
        
        >>> im.h["number"] = 100.0
        >>> im.h["string"] = "wow",
        >>> im.h["bool"] = True
        >>> im.h["list"] = [1.0, 2.0]

        Adding layers

        >>> im.append_layer("one", np.array( [[1.0, 2.0], [3.0, 4.0]] ) )
        >>> im.append_layer("two", np.array( [[5.0, 6.0], [7.0, 8.0]] ) )
        
        Save to a file

        >>> im.save("npzimage.npz")

        Load from a file

        >>> im2 = snpl.image.NpzImage("npzimage.npz")
        >>> print(im2.h)
        {'number': 100.0, 'string': 'wow', 'bool': True, 'list': [1.0, 2.0], 'version': '0.3.0'}
        >>> print(im2.layers["one"])
        [[1. 2.]
        [3. 4.]]
        >>> print(im2.layers["two"])
        [[5. 6.]
        [7. 8.]]
        >>> print(im2.h["string"])
        wow
    """
    
    def __init__(self, fp=None):
        """Initializer
        """
        
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
    
    def save(self, fp, compress=False):
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
    pass