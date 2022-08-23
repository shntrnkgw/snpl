# coding=utf-8
"""A thin wrapper to matplotlib.pyplot and some utility functions

Examples:
    Here are particularly useful functions. 

    Basic plotting

    >>> import snpl           # this will set the style
    >>> import numpy as np
    >>> snpl.limx(0.0, 100.0) # unlike pyplot.xlim(), a margin will be added automatically
    >>> snpl.limy(0.0, 1.0)
    >>> snpl.labelx("$x$") # latex math style is available
    >>> snpl.labelx("$y$")
    >>> x = np.linspace(10.0, 90.0, 100)
    >>> y = np.random.rand(100)
    >>> snpl.plot(x, y, ls="", marker="o", color="g", label="data")
    >>> snpl.legend(fontsize=8, loc="upper right")
    >>> snpl.savefig("test.pdf")
    >>> snpl.delete_lines() # this erases all the lines and texts on the current Axes while keeping other things intact

    Data cropping - ``snpl.crop()``

    >>> import numpy as np
    >>> import snpl
    >>> np.random.seed(0)
    >>> x = np.linspace(0.0, 100.0, 5)
    >>> y = np.random.rand(5)
    >>> z = np.random.rand(5)
    >>> print(x, y, z)
    [  0.  25.  50.  75. 100.] [0.5488135  0.71518937 0.60276338 0.54488318 0.4236548 ] [0.64589411 0.43758721 0.891773   0.96366276 0.38344152]
    >>> xc, yc, zc = snpl.crop(x, [20.0, 50.0], y, z) # x, y, z will be cropped based on the range of x (20.0 ~ 50.0)
    >>> print(xc, yc, zc)
    [25. 50.] [0.71518937 0.60276338] [0.43758721 0.891773  ]    

    Gradient color generator - ``snpl.get_colors()``

    >>> import numpy as np
    >>> import snpl
    >>> x = np.linspace(0.0, 1.0, 10)
    >>> n = 10
    >>> cs = snpl.get_colors(n)
    >>> for i in range(n):
    >>>     snpl.plot(x, np.random.rand(len(x))+i, ls="-", marker="o", color=cs[i], label=str(i))
    >>> snpl.legend()
    >>> snpl.show()

    Color manipulation - ``snpl.tint()``

    >>> import numpy as np
    >>> import snpl
    >>> x = np.linspace(0.0, 1.0, 10)
    >>> c = "g"
    >>> snpl.plot(x, np.random.rand(len(x))+0.0, ls="-", marker="o", color=snpl.tint(c, 0.0), label="0.0")
    >>> snpl.plot(x, np.random.rand(len(x))+1.0, ls="-", marker="o", color=snpl.tint(c, 0.2), label="0.2")
    >>> snpl.plot(x, np.random.rand(len(x))+2.0, ls="-", marker="o", color=snpl.tint(c, 0.5), label="0.5")
    >>> snpl.plot(x, np.random.rand(len(x))+3.0, ls="-", marker="o", color=snpl.tint(c, 0.8), label="0.8")
    >>> snpl.legend()
    >>> snpl.show()

    Access the ``pyplot`` instance

    >>> import snpl
    >>> x = [0.0, 1.0, 2.0]
    >>> y = [3.0, 4.0, 5.0]
    >>> snpl.pyplot.bar(x, y) # you have full access to pyplot functionality
    >>> snpl.show()

"""
__version__ = "0.3.0"
__author__ = "NAKAGAWA Shintaro"

import sys

import numpy as np
from matplotlib import colors, cm, pyplot, ticker, axes, rc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.mathtext import _mathtext


from snpl import afm, fit, gpc, image, tensile, util

#----------------#
# Runtime config #
#----------------#

import importlib.resources
pyplot.style.use(importlib.resources.path("snpl", "matplotlibrc_snpl"))


# This is to make the position of the super/subscripts more natural. 
# Thanks to: https://qiita.com/ogose/items/d110aa090102079fe73f
_mathtext.FontConstantsBase.sup1 = 0.4

#---------------------------------------------------#
# Utility functions to change rc parameters at once #
#---------------------------------------------------#
def figsize(wid_in=None, hei_in=None, ax=None):
    """Sets the figure size

    Args:
        wid_in: width in inches. If None, inherit the current value. Defaults to None. 
        hei_in: height in inches. If None, inherit the current value. Defaults to None. 
        ax: `Axes` object. The size of the figure containing this `Axes` object will be modified. 
            If None, the current figure in `pyplot` will be modified. Defaults to None. 
    
    Returns:
        Current figure width & height in tuple (width, height). 
    """
    ax_ = _get_proper_axis(ax)
    fig_ = ax_.get_figure()
    current = fig_.get_size_inches()
    
    if wid_in:
        wid_ = wid_in
    else:
        wid_ = current[0]
    
    if hei_in:
        hei_ = hei_in
    else:
        hei_ = current[1]
    
    fig_.set_size_inches(wid_, hei_)
    
    return wid_, hei_

def markersize(ms=None, mew=None):
    """Sets the default marker size through matplotlib.rc

    Args:
        ms: Markersize in points. If None, no change will be made. 
            It also sets the errorbar cap size (half of the marker size)
        mew: Marker edge width in points. If None, no change will be made. 
    
    Note:
        This just sets the default value. 
        Styles can be modified for individual plots in `snpl.plot()` etc. 
    
    """
    if ms != None:
        rc("lines", markersize=ms)
        rc("errorbar", capsize=ms/2.0)
    
    if mew != None:
        rc("lines", markeredgewidth=mew)

def ticklabelpadding(val):
    """Sets the default tick label padding through matplotlib.rc

    The tick label padding is the space between the tick 
    and the tick label (numbers on the axis). 
    This method applies for major and minor ticks on both x and y axes. 
    
    Args:
        val: padding in points. 
    """
    rc("xtick.major", pad=val)
    rc("xtick.minor", pad=val)
    rc("ytick.major", pad=val)
    rc("ytick.minor", pad=val)

def linewidth(lw):
    """Sets the default line width through matplotlib.rc

    Default values of 
    - plot line width
    - axes frame line width, 
    - tick mark line width
    - tick mark length (major ticks: lw*4, minor ticks: lw*2)
    will be changed. 

    Args:
        lw: line width in points. 
    """
    rc("lines", linewidth=lw)
    rc("axes", linewidth=lw)
    rc("xtick.major", width=lw, size=lw*4)
    rc("xtick.minor", width=lw, size=lw*2)
    rc("ytick.major", width=lw, size=lw*4)
    rc("ytick.minor", width=lw, size=lw*2)

#-------------#
# Appearances #
#-------------#
def labelx(label, ax=None, **kwargs):
    """Sets the label for x axis for the current axes. 

    Args:
        ax: Axes object to which the label will be set. 
            If `None`, the current Axes in `pyplot`. 
        kwargs: Arguments to `set_xlabel()`. 
    """
    ax_ = _get_proper_axis(ax)
    
    return ax_.set_xlabel(label, **kwargs)

def labely(label, ax=None, **kwargs):
    """Sets the label for y axis for the current axes. 

    Args:
        ax: Axes object to which the label will be set. 
            If `None`, the current Axes in `pyplot`. 
        kwargs: Arguments to `set_ylabel()`. 
    """
    ax_ = _get_proper_axis(ax)
    
    return ax_.set_ylabel(label, **kwargs)

def title(text, ax=None, **kwargs):
    """Sets the label for y axis for the current axes. 

    Args:
        ax: Axes object to which the label will be set. 
            If `None`, the current Axes in `pyplot`. 
        kwargs: Arguments to `set_title()`. 
    """
    ax_ = _get_proper_axis(ax)
    
    return ax_.set_title(text, **kwargs)

def formatx(empty=False, scientific=True, ax=None):
    """Modifies the axis tick label formatting for x axis. 

    Args:
        empty: If `True`, no tick labels. 
        scientific: If `False`, turn off the scientific notation
            (e.g. 10^3)
        ax: Axes object to which the label will be set. 
            If `None`, the current Axes in `pyplot`. 
    """
    ax_ = _get_proper_axis(ax)
    
    if empty:
        ax_.get_xaxis().set_ticklabels([])
    elif scientific:
        ax_.ticklabel_format(axis="x", style="scientific")
    else:
        ax_.ticklabel_format(axis="x", style="plain")

def formaty(empty=False, scientific=True, ax=None):
    """Modifies the axis tick label formatting for y axis. 

    Args:
        empty: If `True`, no tick labels. 
        scientific: If `False`, turn off the scientific notation
            (e.g. 10^3)
        ax: Axes object to which the label will be set. 
            If `None`, the current Axes in `pyplot`. 
    """
    ax_ = _get_proper_axis(ax)
    
    if empty:
        ax_.get_yaxis().set_ticklabels([])
    elif scientific:
        ax_.ticklabel_format(axis="y", style="scientific")
    else:
        ax_.ticklabel_format(axis="y", style="plain")

def remove_borders(*args, ax=None):
    """Removes axes frame on the specified sides. 

    Args:
        args: One or more strings from ("top", "left", "right", "bottom")
            that specifies the border(s) to be removed. 
        ax: Axes object to which the label will be set. 
            If `None`, the current Axes in `pyplot`. 
    """
    ax_ = _get_proper_axis(ax)
    sides = ("top", "left", "right", "bottom")
    
    sides_remove = []    
    for s in args:
        if s in sides:
            sides_remove.append(s)
        else:
            sys.stderr.write("Unrecognized side ignored: {0}".format(s))
    
    sides_remain = [s for s in sides if s not in sides_remove]
    
    if not sides_remove:
        return
    
    [ax_.spines[s].set_visible(False) for s in sides_remove]
    
    if ("top" in sides_remain) and ("bottom" in sides_remain):
        ax_.xaxis.set_ticks_position("both")
    elif ("top" in sides_remain):
        ax_.xaxis.set_ticks_position("top")
    elif ("bottom" in sides_remain):
        ax_.xaxis.set_ticks_position("bottom")
    else:
        ax_.xaxis.set_ticks_position("none")
        ax_.set_xticks([])

        
    if ("left" in sides_remain) and ("right" in sides_remain):
        ax_.yaxis.set_ticks_position("both")
    elif ("left" in sides_remain):
        ax_.yaxis.set_ticks_position("left")
    elif ("right" in sides_remain):
        ax_.yaxis.set_ticks_position("right")
    else:
        ax_.yaxis.set_ticks_position("none")
        ax_.set_yticks([])
        
#--------------------#
# xy scales & limits #
#--------------------#
def limx(left=None, right=None, expandratio=0.03, ax=None):
    """Sets left and right limits of the x axis. 

    If either of two limits is not given, return currently-set limits instead. 
    
    Args:
        left: left limit. 
        right: right limit. 
        ax: Axes instance to set/get the limit. 
        expandratio: expansion ratio. 0.0 for no expansion. 

    Returns:
        current limits in two-tuple. 
    """
    
    ax_ = _get_proper_axis(ax)
        
    if left==None or right==None:
        return ax_.get_xlim()
    else:
        ax_.set_xlim(left, right)
    
        if expandratio > 0.0:
            expandx(expandratio, ax_)
        if ax_.get_xscale() == "log":
            ticxlog(numticks="every", ax=ax_)
            pass
            
        return ax_.get_xlim()

def limy(lower=None, upper=None, expandratio=0.03, ax=None):
    """Sets lower and upper limits of the x axis. 

    If either of two limits is not given, return currently-set limits instead. 
    
    Args:
        lower: lower limit. 
        upper: upper limit. 
        ax: Axes instance to set/get the limit. 
        expandratio: expansion ratio. 0.0 for no expansion. 

    Returns:
        current limits in two-tuple. 
    """
    ax_ = _get_proper_axis(ax)
    
    if lower==None or upper==None:
        return ax_.get_ylim()
    else:
        ax_.set_ylim(lower, upper)
    
        if expandratio > 0.0:
            expandy(expandratio, ax_)
        if ax_.get_yscale() == "log":
            ticylog(numticks="every", ax=ax_)
            pass
        
        return ax_.get_ylim()

def expandx(ratio=0.03, ax=None):
    """Expands the xlim based on the canvas size. 
    
    Args:
        ratio: Expand ratio. 0.0 for no expansion. 
        ax: Axes object to which the label will be set. 
            If `None`, the current Axes in `pyplot`. 
    """
    ax_ = _get_proper_axis(ax)
    
    xl = ax_.get_xlim()
    if ax_.get_xscale() == "log":
        xw = np.log10(xl[1]/xl[0])
    else:
        xw = xl[1]-xl[0]
    
    # width, height = ax_.get_figure().get_size_inches()
    bbox = ax_.get_window_extent().transformed(ax_.figure.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    asp = height/width
    
    
    if ax_.get_xscale() == "log":
        ax_.set_xlim(xl[0]/np.power(10.0, xw*ratio), xl[1]*np.power(10.0, xw*ratio))
    else:
        ax_.set_xlim(xl[0]-xw*ratio, xl[1]+xw*ratio)

def expandy(ratio=0.03, ax=None):
    """Expands the ylim based on the canvas size. 
    
    Args:
        ratio: Expand ratio. 0.0 for no expansion. 
        ax: Axes object to which the label will be set. 
            If `None`, the current Axes in `pyplot`. 
    """
    ax_ = _get_proper_axis(ax)
        
    yl = ax_.get_ylim()
    if ax_.get_yscale() == "log":
        yw = np.log10(yl[1]/yl[0])
    else:
        yw = yl[1]-yl[0]
    
    # width, height = ax_.get_figure().get_size_inches()
    bbox = ax_.get_window_extent().transformed(ax_.figure.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    asp = height/width
    
    if ax_.get_yscale() == "log":
        ax_.set_ylim(yl[0]/np.power(10.0, yw*ratio/asp), yl[1]*np.power(10.0, yw*ratio/asp))
    else:
        ax_.set_ylim(yl[0]-yw*ratio/asp, yl[1]+yw*ratio/asp)

def log(axis="xy", b=True, ax=None):
    """
    
    """
    ax_ = _get_proper_axis(ax)
    
    if axis in ("xy", "both", "all", "XY", "Both", "All"):
        _logx(b, ax_)
        _logy(b, ax_)
    elif axis in ("x", "X"):
        _logx(b, ax_)
    elif axis in ("y", "Y"):
        _logy(b, ax_)
    else:
        sys.stderr.write("No such axis: {0}".format(axis))

def _logx(b, ax, numticks="every"):
    """Internal function to properly set the log scale. 

    This is needed to draw nice ticks. 

    Args:
        b: If `True`, set x axis to logscale. Otherwise, revert to linear scale. 
        ax: Axes object to which the scale will be set. 
    """
    if b:
        if (ax.get_xlim()[0] < 0.0) or (ax.get_xlim()[1] < 0.0):
            ax.set_xlim(1e-1, 1e1)
        ax.set_xscale("log")
        ticxlog(ax=ax, numticks=numticks)
    else:
        ax.set_xscale("linear")
        ticx(step="auto", ax=ax)

def _logy(b, ax, numticks="every"):
    """Internal function to properly set the log scale. 

    This is needed to draw nice ticks. 

    Args:
        b: If `True`, set y axis to logscale. Otherwise, revert to linear scale. 
        ax: Axes object to which the scale will be set. 
    """
    if b:
        if (ax.get_ylim()[0] < 0.0) or (ax.get_ylim()[1] < 0.0):
            ax.set_ylim(1e-1, 1e1)
        ax.set_yscale("log")
        ticylog(ax=ax, numticks=numticks)
    else:
        ax.set_yscale("linear")
        ticy(step="auto", ax=ax)

def ticx(step="auto", minor=False, ax=None):
    """Sets the ticks nicely on linear-scaled x axis. 

    Args:
        b: If `True`, set x axis to logscale. Otherwise, revert to linear scale. 
        ax: Axes object to which the scale will be set. 
    """
    ax_ = _get_proper_axis(ax)
        
    sc = ax_.get_xscale()
    if sc == "linear":
        if step == "auto":
            loc = ticker.AutoLocator()
        else:
            loc = ticker.MultipleLocator(step)
        if minor:
            ax_.xaxis.set_minor_locator(loc)
        else:
            ax_.xaxis.set_major_locator(loc)
    elif sc == "log":
        sys.stderr.write("ticxlog() when using log scale! \n")

def ticy(step="auto", minor=False, ax=None):
    ''' set y axis tick interval '''
    ax_ = _get_proper_axis(ax)
    
    sc = ax_.get_yscale()
    if sc == "linear":
        if step == "auto":
            loc = ticker.AutoLocator()
        else:
            loc = ticker.MultipleLocator(step)
        if minor:
            ax_.yaxis.set_minor_locator(loc)
        else:
            ax_.yaxis.set_major_locator(loc)
    elif sc == "log":
        sys.stderr.write("Use ticylog() when using log scale! \n")

def _calc_log_tics(base, lim):
    
    logx0 = np.log10(lim[0])/np.log10(base)
    logx1 = np.log10(lim[1])/np.log10(base)
        
    if logx0 <= logx1:
        logx0 = np.floor(logx0)
        logx1 = np.ceil(logx1)
        logw = logx1 - logx0
        logx0 = logx0 - logw
        logx1 = logx1 + logw
    else:
        logx0 = np.ceil(logx0)
        logx1 = np.floor(logx1)
        logw = logx0 - logx1
        logx0 = logx0 + logw
        logx1 = logx1 - logw
        
    logs = np.arange(logx0, logx1).astype(np.float)
    tics = np.power(base, logs)
    
    mtics_basic = np.arange(1, 10).astype(np.float)
    mtics = []
    for tic in tics:
        mtics.extend(tic * mtics_basic)
    return tics, np.array(mtics)

def ticxlog(numticks="every", ax=None):
    ax_ = _get_proper_axis(ax)
    sc = ax_.get_xscale()
    if sc == "log":
        if numticks == "every":
            lower = np.floor(np.log10(ax_.get_xlim()[0]))
            upper = np.ceil(np.log10(ax_.get_xlim()[1]))
            num = int(np.abs(upper - lower))
            major_loc = ticker.LogLocator(base=10.0, numticks=num*2)
            minor_loc = ticker.LogLocator(base=10.0, subs=range(10), numticks=num*2)
        else:
            major_loc = ticker.LogLocator(base=10.0, numticks=numticks)
            minor_loc = ticker.LogLocator(base=10.0, subs=range(10), numticks=numticks)
            
        ax_.xaxis.set_major_locator(major_loc)
        ax_.xaxis.set_minor_locator(minor_loc)
        ax_.xaxis.set_minor_formatter(ticker.NullFormatter())
        # lim = ax_.get_xlim()
        # tics, mtics = _calc_log_tics(10.0, lim)
        # ax_.xaxis.set_ticks(tics)
        # ax_.xaxis.set_ticks(mtics, minor=True)
        # ax_.set_xlim(*lim)
    elif sc == "linear":
        sys.stderr.write("Use ticy() when using linear scale! \n")

def ticylog(numticks="every", ax=None):
    ax_ = _get_proper_axis(ax)
    sc = ax_.get_yscale()
    if sc == "log":
        if numticks == "every":
            lower = np.floor(np.log10(ax_.get_ylim()[0]))
            upper = np.ceil(np.log10(ax_.get_ylim()[1]))
            num = int(np.abs(upper - lower))
            major_loc = ticker.LogLocator(base=10.0, numticks=num*2)
            minor_loc = ticker.LogLocator(base=10.0, subs=range(10), numticks=num*2)
        else:
            major_loc = ticker.LogLocator(base=10.0, numticks=numticks)
            minor_loc = ticker.LogLocator(base=10.0, subs=range(10), numticks=numticks)

        ax_.yaxis.set_major_locator(major_loc)
        ax_.yaxis.set_minor_locator(minor_loc)
        ax_.yaxis.set_minor_formatter(ticker.NullFormatter())
        # lim = ax_.get_ylim()
        # tics, mtics = _calc_log_tics(10.0, lim)
        # ax_.yaxis.set_ticks(tics)
        # ax_.yaxis.set_ticks(mtics, minor=True)
        # ax_.set_ylim(*lim)
    elif sc == "linear":
        sys.stderr.write("Use ticx() when using linear scale! \n")

def autoscale(axis="both", ax=None):
    ax_ = _get_proper_axis(ax)
    
    ax_.autoscale(axis=axis)
    
    if ax_.get_xscale() == "log":
        ticxlog(ax=ax_)
    if ax_.get_yscale() == "log":
        ticylog(ax=ax_)
        
#-------------------#
# Get axes & figure #
#-------------------#
def gca():
    ax = pyplot.gca()
    assert isinstance(ax, axes.Axes)
    return ax

def clf():
    pyplot.clf()

def cla():
    pyplot.cla()

#----------#
# Plotting #
#----------#
def plot(*args, **kwargs):
    return pyplot.plot(*args, **kwargs)

def errorbar(*args, **kwargs):
    return pyplot.errorbar(*args, **kwargs)

#--------#
# Legend #
#--------#
def legend(*axes_objects, **legendprops):
    liness = []
    labelss = []
    if axes_objects:
        for ax in axes_objects:
            lines, labels = ax.get_legend_handles_labels()
            liness.extend(lines)
            labelss.extend(labels)
        # get axis border width 
        borderwidth = max(s.get_linewidth() for s in axes_objects[0].spines.values())
            
    else:
        lines, labels = pyplot.gca().get_legend_handles_labels()
        liness.extend(lines)
        labelss.extend(labels)        
        # get axis border width 
        borderwidth = max(s.get_linewidth() for s in pyplot.gca().spines.values())
    
    try:
        rev = legendprops.pop("reverse")
    except KeyError:
        rev = False
    
    try:
        align = legendprops.pop("align")
    except KeyError:
        align = ""
        
    if rev:
        liness = liness[::-1]
        labelss = labelss[::-1]
    
    leg = pyplot.legend(liness, labelss, **legendprops)
    
    if align:
        vp = leg._legend_box._children[-1]._children[0] 
        if align in ("r", "right"):
            for c in vp._children: 
                c._children.reverse() 
            vp.align="right"
        elif align in ("l", "left"):
            vp.align="left"
        else:
            pass
    
    leg.get_frame().set_linewidth(borderwidth)
    
    return leg

#------------------#
# Other appearance #
#------------------#

#-----------#
# Save/show #
#-----------#
def savefig(*args, **kwargs):    
    return pyplot.savefig(*args, **kwargs)

def show(*args, **kwargs):
    return pyplot.show(*args, **kwargs)

#----------#
# Cleaning #
#----------#
def delete_lines(*axes_objects):
    if axes_objects:
        axes_ = axes_objects
    else:
        axes_ = [pyplot.gca()]
        
    for ax in axes_:
        [l.remove() for l in list(ax.lines)]
        [c.remove() for c in list(ax.collections)]
        [t.remove() for t in list(ax.texts)]
        [p.remove() for p in list(ax.patches)]
                
        if ax.get_legend():
            ax.get_legend().remove()
        
        # ax.lines = []
        # ax.collections = []
        # ax.texts = []
        # ax.containers = []
        # ax.patches = []

#---------#
# utility #
#---------#
def get_colors(num, name="inferno"):
    arr = np.linspace(0, 1, num)
    norm = colors.Normalize(arr[0], arr[-1])
    cmap = cm.ScalarMappable(norm=norm, cmap=name)
    return [cmap.to_rgba(v) for v in arr]

def get_markers(num, symbols="osD^>v<XP"):
    syms = list(symbols)
    return [syms[j%len(syms)] for j in range(num)]

def get_linestyles(num, styles=["-", ":", "--", "-."]):
    return [styles[j%len(styles)] for j in range(num)]

def tint(c, f):
    rgb = colors.to_rgb(c)
    return tuple([v + (1.0 - v)*f for v in rgb])

def _get_proper_axis(ax=None):
    if ax:
        return ax
    else:
        return pyplot.gca()



def isnan_multiple(*arrays):
    truth = np.isnan(arrays[0])
    if len(arrays) > 1:
        for arr in arrays[1:]:
            truth = np.logical_or(truth, np.isnan(arr))
    
    return truth

def isinf_multiple(*arrays):
    truth = np.isinf(arrays[0])
    if len(arrays) > 1:
        for arr in arrays[1:]:
            truth = np.logical_or(truth, np.isinf(arr))
    
    return truth

def iscropped(xarr, rangex):
    return np.logical_and(min(rangex) <= xarr, xarr <= max(rangex) )

def isincluded(xarr, rangex):
    return np.logical_or(min(rangex) > xarr, xarr > max(rangex) )

def removenans(*arrays):
    '''
    Removes nans from the input arrays with the same size. 
    '''
    truth = np.logical_not(isnan_multiple(*arrays))
    inds = np.where(truth)
    return [np.copy(arr[inds]) for arr in arrays]

def removeinfs(*arrays):
    truth = np.logical_not(isinf_multiple(*arrays))
    inds = np.where(truth)
    return [np.copy(arr[inds]) for arr in arrays]

def cropper(xarr, rangex, *arrays, remove_nans=True, remove_infs=True):
    
    truth = iscropped(xarr, rangex)
    if remove_nans:
        truth = np.logical_and(truth, np.logical_not(isnan_multiple( *([xarr] + list(arrays)) )))
        
    if remove_infs:
        truth = np.logical_and(truth, np.logical_not(isinf_multiple( *([xarr] + list(arrays)) )))
    
    return np.where(truth)

def excluder(xarr, rangex, *arrays, remove_nans=True, remove_infs=True):

    truth = isincluded(xarr, rangex)
    if remove_nans:
        truth = np.logical_and(truth, np.logical_not(isnan_multiple( *([xarr] + list(arrays)) )))
        
    if remove_infs:
        truth = np.logical_and(truth, np.logical_not(isinf_multiple( *([xarr] + list(arrays)) )))
    
    return np.where(truth)

def crop(xarr, rangex, *arrays, remove_nans=True, remove_infs=True):
    inds = cropper(xarr, rangex, *arrays, remove_nans=remove_nans, remove_infs = remove_infs)
    
    allarrays = [xarr] + list(arrays)
    
    return [np.copy(arr[inds]) for arr in allarrays]

 
def exclude(xarr, rangex, *arrays, remove_nans=True, remove_infs=True):

    inds = excluder(xarr, rangex, *arrays, remove_nans=remove_nans, remove_infs = remove_infs)
    
    allarrays = [xarr] + list(arrays)
    
    return [np.copy(arr[inds]) for arr in allarrays]
 
def cut(xarr, yarr, rangex):
    assert len(rangex) == 2
        
    xarr_, yarr_ = removenans(xarr, yarr)
        
    inds = cutter(xarr_, rangex)
    
    return np.copy(xarr_[inds]), np.copy(yarr_[inds])


def cutter(xarr, rangex):
    '''
    Not compatible with arrays containing NaNs. 
    First remove NaNs with removenans(). 
    '''
    return np.where(np.logical_and(min(rangex) <= xarr, xarr <= max(rangex) ))


#--------------#
# text/drawing #
#--------------#
def draw_powerlaw(xi=1e0, xf=1e1, yi=1e0, exponent=1.0, ax=None, **lineprops):
    ax_ = _get_proper_axis(ax)
    return ax_.plot([xi, xf], [yi, yi*np.power(xf/xi, exponent)], **lineprops)[0]

def text_corner(s, x=0.03, y=0.90, ax=None):
    if not ax:
        a = pyplot.gca()
    else:
        a = ax
        
    return a.text(x, y, s, transform=a.transAxes)

def arrow_edge(height, side="left", len_rel=0.1, ax=None, text="", **arrowprops):
    a = _get_proper_axis(ax)
    
    if a.get_xscale() == "linear":
        xl, xr = a.get_xlim()
        xw = abs(xr - xl)
        len_abs = xw*len_rel

        if side == "left":
            xy = (xl, height)
            xytext = (xl + len_abs, height)
        elif side == "right":
            xy = (xr, height)
            xytext = (xr - len_abs, height)

    else:
        xl, xr = a.get_xlim()
        log_xl = np.log10(xl)
        log_xr = np.log10(xr)
        log_xw = abs(log_xr - log_xl)
        log_len_abs = log_xw*len_rel

        if side == "left":
            xy = (xl, height)
            xytext = (xl * np.power(10.0, log_len_abs), height)
        elif side == "right":
            xy = (xr, height)
            xytext = (xr / np.power(10.0, log_len_abs), height)

    
    return a.annotate(text, xy=xy, xytext=xytext, xycoords="data", textcoords="data", 
                      arrowprops=arrowprops)

def axvline(*args, **kwargs):
    return pyplot.axvline(*args, **kwargs)

def axhline(*args, **kwargs):
    return pyplot.axhline(*args, **kwargs)

if __name__ == "__main__":
    pass