# coding=utf-8
'''I/O interface utilities for AFM (atomic force microscope) images. 
'''

import numpy as np
import datetime
import os.path

MAGIC_LINE = "Gwyddion Simple Field 1.0"

HFIELDS = ("XRes", "YRes", "XReal", "YReal", "XOffset", "YOffset", "Title", "XYUnits", "ZUnits")

class GwyddionSimpleField(object):
    '''Read/write interface for Gwyddion Simple Field format. 

    See http://gwyddion.net/documentation/user-guide-en/gsf.html for specifications. 

    Args:
        fp: Source file (path or file-like object). 
            If None, create an empty object. 

    Attributes:
        h (dict): header. 
            - "XRes": pixel number along the x axis. 
            - "YRes": pixel number along the y axis. 
            - "XReal": real length of the image along the x axis (in meter). 
            - "YReal": real length of the image along the y axis (in meter). 
            - "XOffset": offset to the x coordinates. 
            - "YOffset": offset to the y coordinates. 
            - "Title": title string. 
            - "XYUnits": string describing the unit. (usually m (meter)). 
            - "ZUnits": string describing the unit. (usually m (meter)). 
        x (numpy.ndarray): 1-d array of the x-coordinates along x axis. 
        y (numpy.ndarray): 1-d array of the y-coordinates along y axis. 
        zmat (numpy.ndarray): 2-d array of the z data (height, phase angle, etc)
        xmat (numpy.ndarray): 2-d array of the x coordinates. 
        ymat (numpy.ndarray): 2-d array of the y coordinates. 
        path (str): path to the source file, if any. 

    Example:
        >>> d = afm.GwyddionSimpleField("test.gsf")
        >>> d.save("save_test.gsf")
    '''
    
    def __init__(self, fp=None):
        
        # default members
        self.h = {"XRes": 1, "YRes": 1, "XReal": 1.0, "YReal": 1.0, "XOffset": 0.0, "YOffset": 0.0, }
        self.x = np.zeros(1)
        self.y = np.zeros(1)
        self.zmat = np.zeros((1, 1))
        self.xmat = np.zeros((1, 1))
        self.ymat = np.zeros((1, 1))
        self.path = ""
        
        if fp:
            if isinstance(fp, str):
                with open(fp, "rb") as f:
                    self.load_from_binary(f.read())
                    self.path = fp
            else:
                self.load_from_binary(fp.read())
        else:
            pass
            
    def load_from_binary(self, bindata):        
        # search for the first linefeed
        for i, c in enumerate(bindata):
            if c == 0x0a:
                break
        # see if the first line matches with the magic line
        if bindata[:i].decode("utf-8") != MAGIC_LINE:
            raise ValueError("Not a valid Gwyddion Simple Field file")
            
        # search for the first null byte
        for i, c in enumerate(bindata):
            if c == 0x00:
                break
        
        header_end = i - i%4 + 4 # end of the header section
        
        # parse header
        hlines = bindata[:header_end].decode("utf-8").strip("\0").split("\n")
        h = {}
        for l in hlines[1:]:
            elements = l.split("=", 1)
            if len(elements) < 2:
                continue
            key = elements[0].strip()
            value = elements[1].strip()
            if key in ("XRes", "YRes"):
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
                
            h[key] = value
        
        xres = h["XRes"]
        yres = h["YRes"]
        
        # d = bindata[header_end:header_end+xres*yres*4]
        d = bindata[header_end:]
        flattened = np.frombuffer(d, dtype="<f4")
        mat = np.reshape(flattened, (yres, xres))
        
        # physical dimension of the image
        if "XReal" in h and "YReal" in h:
            xreal = h["XReal"]
            yreal = h["YReal"]
        else:
            xreal = 1.0
            yreal = 1.0
        
        # physical offset of the image
        if "XOffset" in h and "YOffset" in h:
            xoff = h["XOffset"]
            yoff = h["YOffset"]
        else:
            xoff = 0.0
            yoff = 0.0
        
        # calculating the coordinates
        x = np.linspace(xoff, xoff + xreal, xres + 1) # this gives the edges of each pixel
        x = (x[:-1] + x[1:])/2.0               # the position of each pixel is the middle of adjacent edges
                
        y = np.linspace(yoff, yoff + yreal, yres + 1) # this gives the edges of each pixel
        y = (y[:-1] + y[1:])/2.0               # the position of each pixel is the middle of adjacent edges
        
        
        self.x = x # 1d array
        self.y = y # 1d array
        self.xmat, self.ymat = np.meshgrid(x, y) # convert to 2d matrix
        self.zmat = mat
        self.h.update(h)
    
    def save(self, fp):
        '''Saves the image in GSF format. 

        The saved file can be opened by Gwyddion (hopefully)

        Args:
            fp (str or file-like): destination
        '''
        
        headers = [MAGIC_LINE]
        
        for key in HFIELDS:
            try:
                value = self.h[key]
                if isinstance(value, str):
                    string = value
                elif isinstance(value, float) or isinstance(value, int):
                    string = repr(value)
                else:
                    raise ValueError("Header values must be string, float, or int (key \"{0}\")".format(key))
                headers.append(key + " = " + string)
            except KeyError:
                pass
        
        for key, value in self.h.items():
            if key not in HFIELDS:
                if isinstance(value, str):
                    string = value
                elif isinstance(value, float) or isinstance(value, int):
                    string = repr(value)
                else:
                    raise ValueError("Header values must be string, float, or int (key \"{0}\")".format(key))
                headers.append(key + " = " + string)
        
        header = "\n".join(headers)
        header = header.encode()  # convert to bytes
        header = bytes([ch for ch in header if ch != 0x00]) # No null character allowed in the header section
                
        nullpad = 4 - len(header) % 4 # the number of null bytes to fill the header part
        header = header + bytes([0x00]*nullpad) # add necessary null bytes
        
        z = np.ravel(self.zmat)    
        data = z.astype(np.float32).tobytes()
        
        if isinstance(fp, str):
            with open(fp, "wb") as f:
                f.write(header + data)
        else:
            fp.write(header + data)
            

VERSION_OFFSET   = 0x10
ENDFILE_OFFSET   = 0x14
DATASTART_OFFSET = 0x18
DATETIME_OFFSET  = 0x1c # added by SN
XRES_OFFSET      = 0x57a
YRES_OFFSET      = 0x57c
XSCALE_OFFSET    = 0x98
YSCALE_OFFSET    = 0xa0
ZSCALE_OFFSET    = 0xa8
ZOFFSET_OFFSET   = 0xe0

def _load_single_xq(bins, ztype="topography"):
    
    version = int(np.frombuffer(bins[VERSION_OFFSET:VERSION_OFFSET+4], "<i4")[0])
    endfile = int(np.frombuffer(bins[ENDFILE_OFFSET:ENDFILE_OFFSET+4], "<i4")[0])
    datastart = int(np.frombuffer(bins[DATASTART_OFFSET:DATASTART_OFFSET+4], "<i4")[0])
    
    xres = int(np.frombuffer(bins[XRES_OFFSET:XRES_OFFSET+2], "<i2")[0])
    yres = int(np.frombuffer(bins[YRES_OFFSET:YRES_OFFSET+2], "<i2")[0])
    
    xscale = np.frombuffer(bins[XSCALE_OFFSET:XSCALE_OFFSET+8], "<f8")[0]
    xscale = xscale*1e-9 # nm -> m
    
    yscale = np.frombuffer(bins[YSCALE_OFFSET:YSCALE_OFFSET+8], "<f8")[0]
    yscale = yscale*1e-9 # nm -> m
    
    zscale = np.frombuffer(bins[ZSCALE_OFFSET:ZSCALE_OFFSET+8], "<f8")[0]
    if ztype == "topography":
        zscale = zscale*1e-9 # nm -> m
    elif ztype == "phase":
        pass
    else:
        pass
    
    zoffset = np.frombuffer(bins[ZOFFSET_OFFSET:ZOFFSET_OFFSET+8], "<f8")[0] # ?
    
    ymdhms = np.frombuffer(bins[DATETIME_OFFSET:DATETIME_OFFSET+12], "<u2")
    timestamp = datetime.datetime(*ymdhms).timestamp()
    
    # for i, v in enumerate(np.frombuffer(bins[:datastart], "<u2")):
    #     print(i, v)
    
    # print(version, endfile, datastart, xres, yres, xscale, yscale, zscale, zoffset)
    
    dataend = datastart + xres*yres*2
    
    zmat = np.frombuffer(bins[datastart:dataend], "<u2").astype(np.float)
    zmat = (zmat + zoffset)*zscale # may be wrong
    zmat = np.reshape(zmat, (yres, xres))
    
    bins_residue = bins[dataend:]
    
    xreal = xscale*(xres + 1)
    yreal = yscale*(yres + 1)
    
    xu = np.linspace(0.0, xreal, xres+1)
    xu = (xu[:-1] + xu[1:])/2.0

    yu = np.linspace(0.0, yreal, yres+1)
    yu = (yu[:-1] + yu[1:])/2.0
    
    xmat, ymat = np.meshgrid(xu, yu)
    
    # pyplot.gca().set_aspect("equal")
    # pyplot.pcolormesh(xmat, ymat, zmat)
    # pyplot.show()
    
    d = GwyddionSimpleField()
    d.h["XRes"] = xres
    d.h["YRes"] = yres
    d.h["XReal"] = xreal
    d.h["YReal"] = yreal
    d.h["XYUnits"] = "m"
    if ztype == "topography":
        d.h["ZUnits"] = "m"
    elif ztype == "phase":
        d.h["ZUnits"] = "deg"
    d.h["Timestamp"] = timestamp
    d.x = np.copy(xu)
    d.y = np.copy(yu)
    d.xmat = np.copy(xmat)
    d.ymat = np.copy(ymat)
    d.zmat = np.copy(zmat)
    
    return d, bins_residue

def load_xq(fp, ztype="auto"):
    '''Loads a .xqdx or .xqpx files from Hitachi AFM. 

    Args:
        fp: file path or file-like object. The object must be in the binary read mode. 
        ztype: type of the z axis. if "auto", it is inferred from the extension. 
            ``.xqdx`` = height, ``.xqpx`` = phase. Or it can be either of "topography"
            or "phase". 

    Returns:
        a ``GwyddionSimpleField`` object. 
        
    '''
    if isinstance(fp, str):
        with open(fp, "rb") as f:
            bins = f.read()
    else:
        bins = fp.read()
    
    if ztype == "auto":
        assert isinstance(fp, str)
        ext = os.path.splitext(fp)[-1]
        if ext in (".xqdx", ):
            zt = "topography"
        elif ext in (".xqpx", ):
            zt = "phase"
        else:
            assert False
    else:
        zt = ztype
    
    d, bins = _load_single_xq(bins, zt)
    # d.write(fp + ".gsf")
    return d


def row_background_polynominal(xmat, ymat, zmat, polydeg=1, mask=None, preserve_height=True):
    '''
    @param mask: truth array with the same shape as zmat
    '''
    
    zbmat = np.zeros_like(zmat)
    if isinstance(mask, np.ndarray):
        # check dimension
        if zmat.shape != mask.shape:
            assert False
        else:
            ma = mask
    else:
        ma = np.logical_not(np.isnan(zmat)) # mask only nans
            
    
    for i in range(len(zmat)):
        m = ma[i]
        x = xmat[i]
        z = zmat[i]
        
        # polynominal fitting for masked line data
        xm = x[m]
        zm = z[m]
        poly = np.poly1d(np.polyfit(xm, zm, deg=polydeg))
        
        # evaluating polynominal at all points (no matter if it is masked or not)
        zp = poly(x)
        
        zbmat[i] = zp
    
    if preserve_height:
        return zbmat - np.mean(zbmat) # subtract the mean to preserve overall height
    else:
        return zbmat


if __name__ == '__main__':
    pass