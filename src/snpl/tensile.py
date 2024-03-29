# coding=utf-8
"""I/O interfaces for tensile test data
"""

import hicsv
import csv
import numpy as np

def _str2number(string):
    if string.isdigit():
        return int(string)
    else:
        try:
            return float(string)
        except ValueError:
            return string


def _load_Trapezium_from_blocks(blocks):
    """
    takes the "blocks" and returns a list of CSVs.  
    a block contains consecutive rows in the original Trapezium file. 
    """
    # header
    keys = blocks[0][0]
    values = blocks[0][1]
    values = [_str2number(v) for v in values]
    
    h = {k: v for k, v in zip(keys, values)}
    
    # skip to the sample geometry
    # the sample geometry block should have at least 5 rows
    # this is to deal with Trapezium 2 cycle file 
    # that has an additional block next to the header block
    skip = 0
    for i, block in enumerate(blocks[1:]):
        skip = i
        if len(block) < 5:
            continue
        else:
            break
        
    # sample names
    key_name = blocks[skip+1][2][0]
    names = [row[0] for row in blocks[skip+1][4:]]
    
    # sample geometry
    keys  = blocks[skip+1][2][1:]
    units = blocks[skip+1][3][1:]
    keys_testpiece = [k + " [{0}]".format(u) for k, u in zip(keys, units)]
    values_testpiece = []
    for row in blocks[skip+1][4:]:
        values_testpiece.append([_str2number(v) for v in row[1:]])
    
    
    num_batches = len(blocks[skip+1][4:])
        
    keys  = blocks[skip+2][0][1:]
    units = blocks[skip+2][2][1:]
    keys_property = [k + " [{0}]".format(u) for k, u in zip(keys, units)]

    values_property = []
    for row in blocks[skip+2][4:4+num_batches]:
        values_property.append([_str2number(v) for v in row[1:]])
    
    csvs = []
    for i in range(num_batches):
        c = hicsv.hicsv()
        c.h.update(h)
        
        c.h[key_name] = names[i]
        
        for k, v in zip(keys_testpiece, values_testpiece[i]):
            c.h[k] = v
                
        if keys_property:
            for k, v in zip(keys_property, values_property[i]):
                c.h[k] = v
        
        datablock = blocks[skip+3+i]
        
        # get valid columns
        valid_columns = [j for j in range(len(datablock[1])) if datablock[1][j]]
        
        keys = [datablock[1][j] for j in valid_columns]
        units = [datablock[2][j] for j in valid_columns]
        keys = [k + " [{0}]".format(u) for k, u in zip(keys, units)]
                
        rows = []
        for row in datablock[3:]:
            valid_row = [row[j] for j in valid_columns]
            rows.append([_str2number(s) for s in valid_row])
        
        cols = list(zip(*rows))

        for k, col in zip(keys, cols):
            c.append_column(k, np.array(col))
        
        c.h["Batch ID"] = i
        
        csvs.append(c)
    
    return csvs


def load_TrapeziumCSV(fp, encoding="shift-jis", dialect="excel"):
    '''Loads a CSV file exported by Trapezium software. 

    Works both for Trapezium 3 (for AGS-X) and Trapezium 2 (for EZ-L). 

    Args:
        fp (str or file-like): Path or file-like object of the source CSV file. 
        encoding (str): Encoding of the source file. Defaults to shift-jis. 
        dialect (str): Dialect of the CSV to be read. Defaults to ``excel`` (comma-delimited).
            Setting to ``excel-tab`` enables loading a tab-delimited values (TSV).  

    Returns:
        A list of ``hicsv.hicsv`` objects for each test pieces (batches). 

    Note:
        This function is compatible with the data collected with "Single" and "Control" programs 
        in the Trapezium software, but not with "Cycle" program. 
        For the data collected with "Cycle" program, use ``load_TrapeziumCycleCSV``. 

    Examples:
        >>> ds = tensile.load_TrapeziumCSV("trapezium_csv.csv")
        >>> 
        >>> for d in ds:
        >>>     d.save("piece " + c.h["名前"] + ".txt") # save each batch using the batch name as the file name
    '''
    if isinstance(fp, str):
        with open(fp, "r", newline="", encoding=encoding) as f:
            lines = f.readlines()
    else:
        lines = fp.readlines()
    
    reader = csv.reader(lines, dialect=dialect)
    
    blocks = []
    block = []
    for row in reader:
        if not row or all(v.strip()=="" for v in row):
            if block:
                blocks.append(block)
                block = []
            else:
                pass
        else:
            block.append(row)
    
    # this is to ensure that the last block is properly added to blocks. 
    if block not in blocks and block:
        blocks.append(block)
        
    return _load_Trapezium_from_blocks(blocks)

def load_TrapeziumCycleCSV(fp, encoding="shift-jis", initial_cycle_id=0, dialect="excel"):
    """Loads a cycle CSV file exported by Trapezium software. 

    To discriminate between the cycles, each cycle will be given a unique and consequtive 
    "Cycle ID", usually starting from zero. 

    Args:
        fp (str or file-like): Path or file-like object of the source CSV file. 
        encoding (str): Encoding of the source file. Defaults to shift-jis. 
        initial_cycle_id (int): Index of the first cycle. 
            This integer number will be used as the "Cycle ID" of the first cycle. 
        dialect (str): Dialect of the CSV to be read. Defaults to ``excel`` (comma-delimited).
            Setting to ``excel-tab`` enables loading a tab-delimited values (TSV).  

    Returns:
        A list of ``hicsv.hicsv`` objects for each cycle. 

    Note:
        This function is not compatible with the data collected with "Single" and "Control" programs 
        in the Trapezium software. For these cases, use ``load_TrapeziumCSV``. 

    Examples:
        >>> ds = tensile.load_TrapeziumCSV("trapezium_cycle.csv")
        >>> 
        >>> for d in ds:
        >>>     d.save("piece " + c.h["Cycle ID"] + ".txt") # save each cycle using the cycle id as the file name
    """
    if isinstance(fp, str):
        with open(fp, "r", newline="", encoding=encoding) as f:
            lines = f.readlines()
    else:
        lines = fp.readlines()
    
    reader = csv.reader(lines, dialect=dialect)
    
    blocks = []
    block = []
    for row in reader:
        if not row or not any(row):
            blocks.append(block)
            block = []
        else:
            block.append(row)
    
    # header
    keys = blocks[0][0]
    values = blocks[0][1]
    values = [_str2number(v) for v in values]
    
    h = {k: v for k, v in zip(keys, values)}
    
    # cycle number
    # this may not be the actual number of cycles
    # if the measurement was terminated before completing 
    # all cycles
    num_cycles = 0
    for k in ("サイクル回数", "ｻｲｸﾙ回数", "Number of Cycles"):
        try:
            num_cycles = h[k]
        except KeyError:
            pass
        else:
            break
        
    # sample geometry
    keys = blocks[1][2][1:]
    units = blocks[1][3][1:]
    keys_testpiece = [k + " [{0}]".format(u) for k, u in zip(keys, units)]
    values_testpiece = [_str2number(v) for v in blocks[1][4][1:]]
    
    # num_batches = len(blocks[1][4:])
        
    keys = blocks[2][0][1:]
    units = blocks[2][2][1:]
    keys_property = [k + " [{0}]".format(u) for k, u in zip(keys, units)]

    values_property = []
    for row in blocks[2][4:4+num_cycles]:
        values_property.append([_str2number(v) for v in row[1:]])
            
    csvs = []
    for i in range(num_cycles):
        c = hicsv.hicsv()
        c.h.update(h)
        for k, v in zip(keys_testpiece, values_testpiece):
            c.h[k] = v
        
        if keys_property:
            for k, v in zip(keys_property, values_property[i]):
                c.h[k] = v

        # IndexError can be raised
        # if num_cycles does not match the actual 
        # cycle number recorded in the file
        try:
            datablock = blocks[3+i]
        except IndexError:
            break
        
        keys = datablock[1]
        units = datablock[2]
        keys = [k + " [{0}]".format(u) for k, u in zip(keys, units)]
        
        rows = []
        for row in datablock[3:]:
            rows.append([_str2number(s) for s in row])
        
        cols = list(zip(*rows))

        for k, col in zip(keys, cols):
            c.append_column(k, np.array(col))
        
        c.h["Cycle ID"] = i + initial_cycle_id
        
        csvs.append(c)
    
    return csvs

if __name__ == '__main__':
    pass