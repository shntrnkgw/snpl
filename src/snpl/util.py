# coding=utf-8
'''
Created on 2018/02/18

@author: snakagawa
'''

import numpy as np
import re
import datetime
import time

import os.path

BOLTZMANN_CONST = 1.38064852e-23 # m^2 kg s^-2 K-1

def natural_sort(li, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text 
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    li.sort(key=sort_key)

def clean(xs, ys):
    not_nan = np.logical_and(np.logical_not(np.isnan(xs)), np.logical_not(np.isnan(ys)))
    not_inf = np.logical_and(np.logical_not(np.isinf(xs)), np.logical_not(np.isinf(ys)))
    ret = np.logical_and(not_nan, not_inf)
    return xs[ret], ys[ret]

def rotate(azimuthal_angles, azimuthal_profile, rotation):
    
    rotated_angles = []
    
    for angle in azimuthal_angles:
        new_angle = angle + rotation
        if new_angle > 360.0:
            new_angle = new_angle - 360.0
        elif new_angle < 0.0:
            new_angle = new_angle + 360.0
        
        rotated_angles.append(new_angle)
        
    zipped = zip(rotated_angles, azimuthal_profile)
    rotated_angles, rotated_profile = zip(*sorted(zipped, key=lambda pair: pair[0]))
    
    if isinstance(azimuthal_angles, np.ndarray) and isinstance(azimuthal_profile, np.ndarray):
        return np.array(rotated_angles), np.array(rotated_profile)
    else:
        return rotated_angles, rotated_profile

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

if __name__ == '__main__':
    pass