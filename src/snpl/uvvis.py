# coding=utf-8
'''
Created on Aug 26, 2020

@author: snakagawa
'''

from snpl import data
import datetime

def load_HitachiAscii(fp, encoding="shift-jis"):
    
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
    
    timestamp = dt.timestamp()
    
    for i, l in enumerate(lines):
        if l.startswith("ﾃﾞｰﾀﾘｽﾄ"):
            break
    
    lines = [l.strip() for l in lines[i+1:]]
    
    head = lines.pop(0).split("\t") # column headers
    
    out = data.CSV()
    for h in head:
        out.append_column(h, [])
    
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
                out.append_row(values)
    
    out.h["timestamp"] = timestamp
    
    return out

if __name__ == '__main__':
    c = load_HitachiAscii("test/uvvis/HitachiAsciiTest.TXT")
    print(c.ga(0, 1))