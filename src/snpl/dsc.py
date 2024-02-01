# coding="utf-8"
'''I/O interface utilities for DSC data. 
'''

import hicsv
import numpy as np

def load_ProteusExpDat(src, sep=","):
    '''Loads a CSV file exported by Netzsch Proteus DSC software into a ``hicsv`` object. 

    This function loads a CSV file (typically has the file name starts with `ExpDat_`) into a single ``hicsv`` object. 

    Args:
        src (str): Path to the CSV file. Mandatory. 
        sep (str): Separator. Defaults to ``,`` (comma). Optinal. 

    Returns:
        an ``hicsv.hicsv`` object. 

    Example:
        >>> from snpl import dsc
        >>> d = dsc.load_ProteusExpDat("ExpDat_file_name.csv")
        >>> temps, heatflows = d.ga("Temp./°C", "DSC/(mW/mg)")
    '''
    
    keys = []
    rows = []
    header = {}
    with open(src, "r", encoding="cp1252") as f:
        lines = f.readlines()
        
    for l in lines:
        l_ = l.strip()
        
        if not l_:
            continue
        elif l_.startswith("##"):
            # column headers
            keys = l_.replace("##", "").split(sep)
        elif l_.startswith("#"):
            # header data
            header_key, header_value = l_.split(sep, 1)
            header_key = header_key.strip("#: ")
            header[header_key] = header_value
        else:
            # lines other than the column header and the header data are data table
            rows.append([float(v) for v in l_.split(sep)])
    
    rows = np.array(rows)
    columns = rows.T

    out = hicsv.hicsv()
    for k, col in zip(keys, columns):
        out.append_column(k, col)
    
    out.h.update(header)

    return out

if __name__ == "__main__":
    pass