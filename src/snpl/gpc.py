# coding=utf-8
'''I/O interface utilities for GPC data. 
'''

import hicsv
import io
import numpy as np
import glob
import os.path
import json

def load_ExportOmniSECReveal(src_curves, src_measinfo="guess", srcs_peakres="guess"):
    '''Loads text files exported by ``__ExportOmniSEC.py`` into a ``hicsv`` object. 

    OmniSEC (the newer software for Reveal detector) has only a very limited 
    functionality to export the chromatography data. 
    The original author of this code (SN) developed a script ``__ExportOmniSEC.py``
    to semi-automatically export the data in a handy text format. 

    For the sake of generality, the exported data are separated into several files. 
    The main file containing the raw curve data, the file containing the metadata 
    of the measurement (``*_MeasInfo.txt``), and the analysis results 
    (``*_PeakResults_*.txt``). 

    This function loads these files and combine them into a single ``hicsv`` object. 

    Args:
        src_curves: Path to the curve data file. Mandatory. 
        src_measinfo: Path to the MeasInfo file. If it is set to "guess", the function
            infers the path from ``src_curves``. 
        srcs_peaks: Path(s) to the PeakResults file. If it is set to "guess", 
            the function infers the path(s) from ``src_curves``. 
            Otherwise, if it is a single string representing a path, 
            it is treated as the only associated PeakResults file. 
            If it is a list of strings, it is treated as the list of 
            multiple PeakResults files associated to the given curve data. 

    Returns:
        an ``hicsv.hicsv`` object. 

    Example:
        >>> from snpl import gpc
        >>> d = gpc.load_ExportOmniSECReveal("2022-07-14-15-30-59-20220714_SN628A.txt")
        >>> print(d.h.keys())
        dict_keys(['MeasInfo', 'PeakResults 20220714-abs-2'])
    '''

    # raw data
    out = hicsv.txt2hicsv(src_curves, key_line=0)
    
    srcd, srcf = os.path.split(src_curves)
    srcf, srce = os.path.splitext(srcf)
    
    # find measinfo
    if src_measinfo == "guess":
        src_meas = os.path.join(srcd, srcf + "_MeasInfo.txt")
    else:
        src_meas = src_measinfo

    try:
        with open(src_meas, "r") as fp:
            measinfo = json.load(fp)
    except (IOError, OSError):
        print("Cannot find/open", src_meas)
    else:
        out.h["MeasInfo"] = measinfo
    
    # find PeakResults
    if srcs_peakres == "guess":
        srcs_res = glob.glob(os.path.join(srcd, srcf + "_PeakResults_*.txt"))
    elif isinstance(srcs_peakres, str):
        # assume single string input
        srcs_res = [srcs_peakres, ]
    else:
        # assume the list input
        srcs_res = srcs_peakres

    for src_res in srcs_res:
        try:
            with open(src_res, "r") as fp:
                peakres = json.load(fp)
        except (IOError, OSError):
            print("Cannot find/open", src_res)
        else:
            b, meth = src_res.rsplit("_PeakResults_", 1)
            meth, e = meth.rsplit(".", 1)
            out.h["PeakResults " + meth] = peakres        
    
    return out

def load_OmniSECExported(fp):
    '''Loads the exported text file from the OmniSEC software (older one)
    
    '''
    
    if isinstance(fp, str):
        with open(fp, "r", newline="") as f:
            lines = f.readlines()
    else:
        lines = fp.readlines()
        
    # divide into blocks
    lines = [l.strip("\r\n ") for l in lines]
    
    blocks = []
    
    block = []
    for l in lines:
        if l == "":
            blocks.append(block)
            block = []
        else:
            block.append(l)
    blocks.append(block)
    
        
    # TODO: parse header part
    
    # load data part
    with io.StringIO("\n".join(blocks[-1])) as f:
        c = hicsv.txt2hicsv(f, sep="\t", key_line=0)
        for key, col in zip(c.keys, c.cols):
            if col[0] == "":
                newcol = np.array([_float_blank_nan(e) for e in col])
                c.replace_column(key, newcol)
    
    return c

def _float_blank_nan(string):
    '''Utility function to convert empty string to nan and non-empty string to float. 
    '''
    if string.strip() == "":
        return np.nan
    else:
        return float(string)


def load_LabSolutionsAscii(fp, sep="\t", target="[LC Chromatogram"):
    '''Loads the exported ASCII file from Shimadzu LabSolutions software. 

    LabSolutions ASCII file has some header part at the beginning 
    followed by multiple "blocks" containing time series data 
    for different channels (RI, UV, pressure, etc). 
    This function extracts one of the channel (specified by ``target`` argument)
    and returns it as a convenient ``hicsv`` object. 

    Args:
        fp: path to the file or file-like object. 
        sep: delimiter used in the ASCII file. Usually a tab. 
        target: the beginning of the block of interest. 

    Returns:
        a ``hicsv.hicsv`` object. 

    
    '''
    if isinstance(fp, str):
        with open(fp, "r", newline="") as f:
            lines = f.readlines()
    else:
        lines = fp.readlines()
    
    # divide into blocks
    lines = [l.strip() for l in lines]
    
    blocks = []
    
    block = []
    for l in lines:
        if l == "":
            blocks.append(block)
            block = []
        else:
            block.append(l)
    blocks.append(block)
    
    b_ch = []
    # search for blocks
    for b in blocks:
        if b:
            if b[0].startswith(target):
                b_ch = b
                break
    
    # b_mw = []
    # for b in blocks:
    #     if b:
    #         if b[0].startswith("[Molecular Weight Distribution Table"):
    #             b_mw = b
    #             break
    
    ret = hicsv.hicsv()
        
    if isinstance(fp, str):
        ret.h["path"] = fp
        
    if b_ch:
        keys, cols = _read_LabSolutionsDataBlock(b_ch, sep)
        
        for k, c in zip(keys, cols):
            ret.append_column(k, c)
            
            
    # if b_mw:
    #     keys, cols = _read_LabSolutionsDataBlock(b_mw, sep)
    # 
    #     keys[keys.index("R.Time")] = "MW R.Time"
    #     
    #     for k, c in zip(keys, cols):
    #         ret.set_column(k, c)
        
    return ret

def _read_LabSolutionsDataBlock(lines, sep="\t"):
    
    i_data = 0
    for i, l in enumerate(lines):
        try:
            float(l.split(sep)[0].strip())
        except ValueError:
            pass
        else:
            i_data = i
            break
        
    head = lines[:i_data]
    tail = lines[i_data:]
    
    mult = 1.0
    for l in head:
        if l.startswith("Intensity Multiplier"):
            mult = float(l.strip().split(sep)[1])
            break
        
    keys = head[-1].strip().split(sep)
        
    rows = []
    
    ncols = len(keys)
    
    for l in tail:
        row = l.strip().split(sep)
        if len(row) != ncols:
            pass
        else:
            row = [float(r) for r in row]
            rows.append(row)
        
    cols = list(zip(*rows))
    cols = [np.array(col) for col in cols]
    
    if "Intensity" in keys:
        i_int = keys.index("Intensity")
        cols[i_int] = cols[i_int]*mult
    
    
    return keys, cols


def load_TosohAscii(fp):
    '''Loads a legacy Tosoh ascii file into a ``hicsv`` object. 
    '''

    if isinstance(fp, str):
        # regard fp as a path
        with open(fp, "r", newline="") as f:
            lines = f.readlines()
    else:
        # regard fp as a file-like obj
        lines = fp.readlines()
    
    tcsv = hicsv.hicsv()
    ts = []
    ys = []
    for line in lines:
        l = line.strip()
        
        if not l:
            pass
        elif l.startswith("#"):
            pass
        else:
            elems = l.split("\t")
            ts.append(float(elems[0].strip()))
            ys.append(float(elems[1].strip()))

    tcsv.append_column("t", np.array(ts))
    tcsv.append_column("y", np.array(ys))
            
    return tcsv



def time2mw(t, polycoeffs):
    logmw = np.poly1d(polycoeffs)
    return np.power(10.0, logmw(t))

def mw2time(mw, polycoeffs):
    logmw = np.log10(mw)
    p = [e for e in polycoeffs]
    p[-1] = p[-1] - logmw
    roots = np.roots(p)
    
    return np.real(roots[np.isreal(roots)][0])


if __name__ == '__main__':
    pass