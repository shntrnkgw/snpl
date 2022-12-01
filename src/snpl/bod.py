# coding="utf-8"

import hicsv
import numpy as np

def load_OxiTopCSV(fp, sep=";"):
    """Loads a CSV file from OxiTop into a handy ``hicsv`` object. 
    
    Args:
        fp: path to the file or file-like object. 
        sep: delimiter. Defaults to semicolon. 

    Returns:
        a ``hicsv.hicsv`` object. 

    Examples:
        >>> d = load_OxiTopCSV("oxitop_raw_data.CSV")
        >>> print(d.keys)
        ['No.', 'Time [sec]', 'hPa']
        >>> t, deltaP = d.ga("Time [sec]", "hPa")
    """

    if isinstance(fp, str):
        with open(fp, "r") as f:
            lines = f.readlines()
    else:
        lines = fp.readlines()

    # get header lines

    i_headerend = 0

    for i, l in enumerate(lines):
        if l.strip() == "":
            i_headerend = i
            break
    
    if i_headerend == 0:
        raise ValueError("cannot find the end point of the header part")

    lines_h = lines[:i_headerend]
    lines_t = lines[i_headerend:]

    ################
    # parse header #
    ################
    h = {}
    cells_h = []
    for l in lines_h:
        # split by `sep` and join into one big list
        cells_h.extend(l.strip().split(sep))

    pairs = []
    pair = []
    is_previous_blank = False
    for cell in cells_h:
        # case 1: current is blank & previous is blank     => do nothing
        # case 2: current is blank & previous is not blank => end of the previous `pair`. add the `pair` to `pairs` and erase `pair`
        # case 3: current is not blank & previous is blank => beginning of a new `pair`
        # case 4: current is not blank & previous is not blank => add to the current `pair`
        if not cell:
            if is_previous_blank:
                is_previous_blank = True
                continue
            else:
                pairs.append(pair)
                pair = []
                is_previous_blank = True
                continue
        else:
            pair.append(cell)
            is_previous_blank = False
            continue
    
    h = {}
    count_nokey = 0
    for pair in pairs:
        if len(pair) == 1:
            h["Keyless value {0}".format(count_nokey)] = pair[0]
            count_nokey = count_nokey + 1
        elif len(pair) == 2:
            h[pair[0][:-1]] = pair[1] # [:-1] to remove the trailing colon
    
    ###############
    # parse table #
    ###############

    lines_t = [l.strip() for l in lines_t] # remove line feed
    lines_t = [l for l in lines_t if l] # remove blank lines

    rows = [l.split(sep) for l in lines_t]

    keys = []
    i_start_values = 0
    # first, scan for the non-table part at the beginning
    for i, row in enumerate(rows):
        if not row[0]: # if the 0th cell in the row is blank, it is a header row
            h[row[1]] = row[2]
        else: # the first row with non-blank 0th cell is the key row
            keys = row[:-1]
            i_start_values = i + 1
            break # get out of the loop as soon as the above row is found
    
    cols = list(zip(*[row[:-1] for row in rows[i_start_values:]]))
    cols = [np.array([float(e) for e in c]) for c in cols]

    out = hicsv.hicsv()
    out.h.update(h)
    for k, a in zip(keys, cols):
        out.append_column(k, a)
    
    return out

if __name__ == "__main__":
    pass