# coding=utf-8
"""Utility functions
"""

import numpy as np
import re
import datetime
import time

import os.path

def natural_sort(li, key=lambda s:s):
    """ Sorts the list into natural alphanumeric order.

    Performs an inplace natural sort on a list of strings. 

    Args:
        li (list): List to be sorted. 
        key (callable): Function that takes a single argument and returns a key to be used for sorting. 
            Defaults to ``lambda s:s`` (the argument is used as the key). 
    
    Returns:
        None

    Examples:
        >>> a = ["f2", "f0", "f10", "f1"]
        >>> a.sort()
        >>> print(a)
        ['f0', 'f1', 'f10', 'f2']
        >>> natural_sort(a)
        >>> print(a)
        ['f0', 'f1', 'f2', 'f10']
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text 
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    li.sort(key=sort_key)

def swap_ext(fpath, new_ext):
    name, ext = os.path.splitext(fpath)
    return name + os.path.extsep + new_ext

def attach_prefix(fpath, prefix):
    dpath, fname = os.path.split(fpath)
    return os.path.join(dpath, prefix + fname)

def attach_postfix(fpath, postfix):
    name, ext = os.path.splitext(fpath)
    return name + postfix + ext

def modify_path(d, ext=None, pref=None, postf=None):
    p = d
    if ext != None:
        p = swap_ext(p, ext)
    if pref != None:
        p = attach_prefix(p, pref)
    if postf != None:
        p = attach_postfix(p, postf)
    
    return p

def datetime2epochsecond(dt):
    return time.mktime(dt.timetuple())

def epochsecond2datetime(es):
    return datetime.datetime.fromtimestamp(es)

def splitnull(c_string):
    '''
    convert null terminated string to a string
    '''
    buf = c_string.split("\0")
    return buf[0]

if __name__ == '__main__':
    a = ["f2", "f0", "f10", "f1"]
    a.sort()
    print(a)
    natural_sort(a)
    print(a)