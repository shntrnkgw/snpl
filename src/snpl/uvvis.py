# coding=utf-8
"""I/O interfaces for UV-vis spectrum files
"""

import hicsv
import datetime
import numpy as np

def load_HitachiTXT(fp, encoding="shift-jis"):
    """Loader for a text file exported from Hitachi UV-vis spectrometer software. 

    This function loads a text file exported from Hitachi UV-vis spectrometer
    and converts it into a hicsv format. 

    Args:
        fp (str or file-like): Path or file-like object of the source .TXT file. 
        encoding (str): Encoding of the source file. Defaults to shift-jis. 

    Returns:
        A ``hicsv.hicsv`` object. Contains columns for wavelength and absorbance. 
        The header only includes the time of measurement. 

    Examples:
        >>> c = snpl.uvvis.load_HitachiTXT("HitachiTest.TXT")
        >>> print(c.ga("nm"))
        [500. 498. 496. ... 204. 202. 200.]
        >>> print(c.ga("Abs"))
        [-1.500e-02 -1.500e-02 -1.500e-02 ...  4.519e+00  4.322e+00  4.250e+00]
        >>> print(c.h["datetime"])
        2020-08-13T21:24:32
        >>> print(c.h["timestamp"])
        1597321472.0
        >>> c.save("HitachiTest_out.txt")
    
    """
    
    if isinstance(fp, str):
        # regard fp as a path
        with open(fp, "r", newline="", encoding=encoding) as f:
            lines = f.readlines()
    else:
        # regard fp as a file-like obj
        lines = fp.readlines()
    
    # extract date
    for i, l in enumerate(lines):
        if l.startswith("分析日時"):
            break    
    
    l = lines[i].strip()
    elems = l.split("\t")
    elems = elems[1].split(",")
    dt_string = " ".join([elems[0].strip(), elems[1].strip()])
    
    dt = datetime.datetime.strptime(dt_string, "%H:%M:%S %m/%d/%Y")
    
    timestring = dt.isoformat()
    timestamp = dt.timestamp()
    
    for i, l in enumerate(lines):
        if l.startswith("ﾃﾞｰﾀﾘｽﾄ"):
            break
    
    lines = [l.strip() for l in lines[i+1:]]
    
    keys = lines.pop(0).split("\t") # column headers
    
    
    
    rows = []
    for l in lines:
        if not l:
            continue
        else:
            elems = l.split("\t")
            try:
                values = [float(e) for e in elems]
            except ValueError:
                pass
            else:
                rows.append(values)
    
    rows = np.array(rows)
    cols = np.transpose(rows)
    
    out = hicsv.hicsv()
    for key, col in zip(keys, cols):
        out.append_column(key, col)

    out.h["timestamp"] = timestamp
    out.h["datetime"] = timestring
    
    return out

if __name__ == '__main__':
    pass