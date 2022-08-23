# coding=utf-8
'''
Created on 2018/02/18

@author: snakagawa
'''

import numpy as np
import hicsv

import olefile
import struct

import os.path

KEYS = { 6: "gap [um]", 
        23: "time [s]",
        25: "sweep number?",
        26: "temperature [C]",
        27: "torque [N m]",
        29: "velocity [rad/s]",
        32: "osc. torque [N m]",
        34: "displacement [rad]",
        35: "displacement [rad]",
        39: "frequency [Hz]",
        41: "raw phase [rad]",
        50: "normal force [N]"}

KEYS_GEOMETRY = {1001: "shear rate factor", 
                 1002: "shear stress factor [m-3]", 
                 1003: "measurement system factor [m-3]", 
                 1004: "fluid density factor [m5]", 
                 1005: "normal force factor [m-2]", 
                 1051: "geometry intertia [N m s2]", 
                  100: "backoff distance [um]", 
                    2: "gap? [um]", 
                    1: "geometry diameter [m]"}

# KEYS = {
#     0x27: "frequency [Hz]", 
#     0x17: "time [s]", 
#     0x1a: "temperature [C]", 
#     0x19: "index", 
#     0x20: "oscillation torque [N m]", 
#     0x29: "raw phase [rad]", 
#     0x22: "displacement [rad]", 
#     0x23: "displacement [rad]", 
#     0x1b: "torque [N m]",
#     0x1d: "velocity [rad s-1]", 
#     0x32: "normal force [N]", 
#     0x06: "gap [um]"
#     }

GTYPE_CP = bytearray([255, 255, 4, 0, 3, 0])
GTYPE_PP = bytearray([255, 255, 0, 0, 2, 0])

def load_RheologyAdvantageData(fp, encoding="utf-16"):
    
    if isinstance(fp, str):
        src = os.path.realpath(fp)
    else:
        src = os.path.realpath(fp.name)
    
    with olefile.OleFileIO(fp) as f:
        
        # geometry information
        geo = {}
        with f.openstream(['Embedding 2', 'Geometry']) as s:
            s.seek(6)
            
            offset, = struct.unpack("i", s.read(4))
            geo["Geometry name"] = s.read(offset).decode(encoding).strip("\u0000")
                                    
            offset, = struct.unpack("i", s.read(4))
            geo["Geometry description"] = s.read(offset).decode(encoding).strip("\u0000")
            
            gsig = s.read(6) # This seems to indicate geometry type? 
            
            if gsig == GTYPE_PP:
                geo["Geometry type"] = "PP"
                geo["Diameter [m]"],              = struct.unpack("d", s.read(8))
                geo["Gap [um]"],                  = struct.unpack("d", s.read(8))
                geo["Backoff distance [um]"],     = struct.unpack("d", s.read(8))
                geo["Geometry inertia [N m s2]"], = struct.unpack("d", s.read(8))
                geo["Geometry compliance [rad N-1 m-1]"], = struct.unpack("d", s.read(8))
            
            elif gsig == GTYPE_CP:
                geo["Geometry type"] = "CP"        
                geo["Angle [s]"],                 = struct.unpack("d", s.read(8)) 
                geo["Diameter [m]"],              = struct.unpack("d", s.read(8))
                geo["Truncation gap [um]"],       = struct.unpack("d", s.read(8))
                geo["Backoff distance [um]"],     = struct.unpack("d", s.read(8))
                geo["Geometry inertia [N m s2]"], = struct.unpack("d", s.read(8))
                geo["Gemetry compliance [rad N-1 m-1]"], = struct.unpack("d", s.read(8))
            else:
                geo["Geometry type"] = "unknown"
                
        num = 0
        cs = []
        while True:
            path = ["Results", "Results{0:d}".format(num)]
            if not f.exists(path):
                break
            
            c = snpl.data.CSV()
            
            with f.openstream(path) as s:
                s.seek(6)
                    
                offset, = struct.unpack("i", s.read(4))
                            
                step = s.read(offset).decode(encoding).strip("\u0000")
                            
                s.seek(16, 1)
                
                num_blocks, = struct.unpack("i", s.read(4))
                
                # begin block
                ucount = 0
                for i_block in range(num_blocks):
                    try:
                        keyb = s.read(1)[0]
                    except IndexError:
                        break
                    
                    try:
                        key = KEYS[keyb]
                    except KeyError:
                        key = "unknown_{0:d}".format(ucount)
                        ucount += 1
                    
                    hb = s.read(7)
                    
                    lb = s.read(4)
                    
                    N, = struct.unpack("i", lb)
                    
                    s.seek(N, 1)
                    
                    arr = struct.unpack("{0:d}f".format(N), s.read(N*4))
                    
                    c.set_column(key, arr)
                
                num += 1
            
            c.h["Step name"] = step
            c.h["Step number"] = num
            c.h["Source file path"] = src
            c.h.update(geo)
            
            cs.append(c)
    return cs


# def calc_RheoAdvantage_CompModulus():

def calc_RheologyAdvantageSteadyViscosity(fp):
    
    c = snpl.data.CSV(fp)
    
    if c.h["Geometry type"] == "CP":
        R_m = c.h["Diameter [m]"]/2.0
        alpha_rad = np.deg2rad(c.h["Angle [s]"]/3600.0)
        
        Fsig = 3.0/(2.0*np.pi*R_m**3)
        Fgam = 1.0/(np.tan(alpha_rad))
        
    elif c.h["Geometry type"] == "PP":
        R_m = c.h["Diameter [m]"]/2.0
        h_m = np.deg2rad(c.h["Gap [um]"]/1e6)
        
        Fsig = 2.0/(np.pi*R_m**3)
        Fgam = R_m/h_m
    
    else:
        assert False
        
    sig_Pa = c.ga("torque [N m]")*Fsig
    srate_persec = c.ga("velocity [rad/s]")*Fgam
    
    visc_Pas = sig_Pa/srate_persec
    
    c.set_column("shear stress [Pa]", sig_Pa)
    c.set_column("shear rate [/s]", srate_persec)
    c.set_column("viscosity [Pa s]", visc_Pas)
    
    return c



def load_RheologyAdvantageExportedSingleStep(fp, encoding="shift-jis"):
    
    if isinstance(fp, str):
        with open(fp, "r", newline="", encoding=encoding) as f:
            lines = f.readlines()
    else:
        lines = fp.readlines()

    # get step names
    stepnames = []
    for l in lines:
        if l.startswith("Step name\t"):
            stepname = l.strip().split("\t")[1]
            stepnames.append(stepname)
    
    # find the step in the stepnames, found first
    i0 = 0
    for i, l in enumerate(lines):
        if l.strip() in stepnames:
            i0 = i
            break
        
    keys = lines[i0+2].strip().split("\t")
    units = lines[i0+3].strip().split("\t")
    
    kus = []
    
    for k, u in zip(keys, units):
        if u:
            kus.append("{0} [{1}]".format(k.replace("'", "′"), u))
        else:
            kus.append(k)
    
    out = snpl.data.CSV()
    for key in kus:
        out.append_column(key, [])
    
    for l_ in lines[i0+5:]:
        l = l_.strip()
        if not l:
            break
        
        out.append_row([float(v) for v in l.split("\t")])
    
    return out


def load_Physica501CSV(fp, sep=",", encoding="shift-jis"):
    
    if isinstance(fp, str):
        with open(fp, "r", newline="", encoding=encoding) as f:
            lines = f.readlines()
    else:
        lines = fp.readlines()

    lines = [l.replace(";", ",").strip() for l in lines]
    
    start_indices = []
    
    for i, l in enumerate(lines):
        if l.startswith("Data Series Information"):
            start_indices.append(i)
    
    blocks = []
    if len(start_indices) == 0 or len(start_indices) == 1:
        blocks.append(lines)
    else:
        for j in range(len(start_indices)):
            try:
                blocks.append(lines[start_indices[j]:start_indices[j+1]])
            except IndexError:
                blocks.append(lines[start_indices[j]:])   
    
    csvs = []
    for bid, b in enumerate(blocks):        
        i_data = 0
        for i, l in enumerate(b):
            if not l:
                continue
            elif l[0].isdigit():
                i_data = i - 2
                break
        
        hlines = b[:i_data]
        dlines = b[i_data:]
        
        # print(hlines[0], hlines[-1])
        # print(dlines[0], dlines[-1])
        
        units = dlines[1]
        dlines = [ [e for e in l.split(sep) if e] for l in dlines]
        
        csv = snpl.data.CSV()
        keys = dlines[0]
        
        for k in keys:
            csv.append_column(k, [])
        
        for l in dlines[2:]:
            elems = []
            if not l:
                continue
            for e in l:
                try:
                    elems.append(float(e.replace(",", "")))
                except ValueError:
                    elems.append(np.nan)
            
            try:
                csv.append_row(elems)
            except AssertionError:
                print(elems)
        
        csv.h["units"] = units
        
        
        # parse header
        name = "{0:d}".format(bid)
        for l in hlines:
            if l.startswith("Name"):
                name = l.strip().split(sep)[-1]
        
        csv.h["Name"] = name
        
        csvs.append(csv)
    
    return csvs
        


def load_RheoCompassText(fp, sep=",", encoding="utf-16"):
    '''
    Not compatible with the tests having multiple intervals!
    '''
    
    
    if isinstance(fp, str):
        with open(fp, "r", newline="", encoding=encoding) as f:
            lines = f.readlines()
    else:
        lines = fp.readlines()

    lines = [l.replace(";", ",") for l in lines]
    
    start_indices = []
    
    for i, l in enumerate(lines):
        if l.startswith("Test:"):
            start_indices.append(i)
    
    blocks = []
    if len(start_indices) == 0 or len(start_indices) == 1:
        blocks.append(lines)
    else:
        for j in range(len(start_indices)):
            try:
                blocks.append(lines[start_indices[j]:start_indices[j+1]])
            except IndexError:
                blocks.append(lines[start_indices[j]:])   
        
    csvs = []
    for bid, b in enumerate(blocks):        
        # get test name
        name_test = b[0].strip().split(sep)[1]
        # print("Test name", name_test)
        
        # search for Interval
        i_interv = 0
        for i, l in enumerate(b):
            if l.startswith("Interval:"):
                i_interv = i
                break
        
        # get keys
        keys = b[i_interv+1].split(sep)
        keys = [k.strip() for k in keys]
        # print("keys", keys)
        
        # search for the start point of the data
        i_data = 0
        for i, l in enumerate(b):
            if not l:
                continue
            elif l[0].isdigit():
                i_data = i
                break
        
        if i_data == 0: # for empty data block
            hlines = b
            dlines = []
        else:
            # split header and data lines
            hlines = b[:i_data]
            dlines = b[i_data:]
                
        # get units
        units = hlines[-1].split(sep)
        units = [u.strip() for u in units]
        
        # print("units", units) 
        
        
        
        keys_comp = []
        for k, u in zip(keys, units):
            if u:
                keys_comp.append(k + " " + u)
            else:
                keys_comp.append(k)
        
        # get unique column
        keys_comp = np.array(keys_comp)
        keys_unique = np.unique(keys_comp)
        
        column_mapping = [keys_comp.tolist().index(uk) for uk in keys_unique]
        
        csv = snpl.data.CSV()
        for k in keys_unique:
            csv.append_column(k, [])
        
        dlines = [ [e.strip() for e in l.split(sep) if l.strip()] for l in dlines]
        
        for l in dlines:
            elems = []
            if not l:
                continue
            for i in column_mapping:
                try:
                    elems.append(float(l[i].replace(",", "")))
                except ValueError:
                    elems.append(np.nan)

            try:
                csv.append_row(elems)
            except AssertionError as er:
                print(str(er))
        
        csv.h["Test name"] = name_test
        
        
        csvs.append(csv)
    
    return csvs



def load_Physica501Table(fp):
    tab = Physica501Table(fp)
    
    return tab.get_CSV()
    

class Physica501Table(object):
    
    def __init__(self, fp="", delay_sec=0.0):
        
        if fp:
            if isinstance(fp, str):
                with open(fp, "r", newline="", encoding="shift-jis") as f:
                    lines = f.readlines()
            else:
                lines = fp.readlines()

            
            # search for header line
            i_header = 0
            for i, line in enumerate(lines):
                if line.startswith("Meas. Pts."):
                    i_header = i
                    break
            
            headers = lines[i_header].strip().split("\t")
            units = lines[i_header+1].strip().split("\t")
            if lines[i_header+1].startswith("\t"):
                units.insert(0, "")
            
            di = {}
            for header in headers:
                di[header] = []
            
            for l in lines[i_header+2:]:
                
                data_str = l.strip().split("\t")
                if len(data_str) != len(headers):
                    break

                pts = int(data_str[0].replace(",", ""))
                for datum_str, header in zip(data_str, headers):
                    try:
                        v = float(datum_str.replace(",", "").replace("\"", ""))
                    except ValueError:
                        v = np.nan
                    di[header].append(v)
            
            for header in headers:
                di[header] = np.array(di[header])
            
            # condering delay if header Time is included. 
            try:
                di["Time"] = di["Time"] + delay_sec
            except KeyError:
                pass
            
            self.headers = headers
            self.units = units
            self.dict = di
            self.pts = pts
            self.delay_sec = delay_sec
        else:
            self.headers = []
            self.units = []
            self.dict = {}
            self.pts = []
            self.delay_sec = delay_sec
        
    def get_headers(self): 
        return self.headers
    
    def get_unit(self, header):
        return self.units[self.headers.index(header)]
    
    def get_array(self, header):
        return self.dict[header]
    
    def get_CSV(self):
        t = snpl.data.CSV()
        
        t.h["units"] = ",".join(self.units)
        
        for k in self.headers:
            t.append_column(k, [])
        
        columns = [self.dict[k] for k in self.headers]
                
        for row in zip(*columns):
            t.append_row(row)        
        
        return t

if __name__ == '__main__':
    # r = load_Physica501CSV("./test/rheo/20199725_Someya_good.txt", sep="\t")
    # print(len(r))
    
    # rc = load_RheoCompassText("./test/rheo/RheoCompass_20191225_someya.txt", sep="\t")
    # [d.write("./test/rheo/" + d.h["Test name"] + ".txt") for d in rc]
    
    # cs = load_RheoAdvantage("test/rheo/RheologyAdvantage/20200311_JS100_PP20_20C_jump_70C-0013f.rsl")
    # cs = load_RheologyAdvantageData("test/rheo/RheologyAdvantage/20200313_paraffin_35C-0000f.rsl")
    # 
    # for c in cs:
    #     with open("test/rheo/RheologyAdvantage/test_{0}.txt".format(c.h["Step number"]), "w", encoding="utf-8") as f:
    #         c.write(f)
            
    "test/rheo/RheologyAdvantage/test_1.txt"
    