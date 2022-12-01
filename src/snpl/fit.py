# coding=utf-8
"""Utilities for curve fitting analyses

Note:
    This module is not really useful anymore. 
    It is encouraged to use directly the ``scipy.optimize`` package instead. 
"""

import glob
import sys
import math

import numpy as np
import scipy.optimize
import scipy.special
import scipy.integrate

import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("error", OptimizeWarning)

def Line(x, p):
    return p[0]*np.array(x) + p[1]

def Gaussian(x, p):
    '''
    p[0]: amp
    p[1]: position
    p[2]: fwhm
    '''
    return p[0]*np.exp(-np.log(2.) * np.power(((np.array(x)-p[1])/(p[2]/2.)), 2.))

def Lorentzian(x, p):
    return p[0]/(1 + np.power((np.array(x)-p[1])/p[2], 2.))

def Voigt(x, p):
    amp = p[0]
    pos = p[1]
    fwhm = p[2]
    shape = p[3]
    sigma = fwhm/(2.*np.sqrt(np.log(2.))) - np.sqrt(2.)
    gamma = fwhm * shape / 2.
    # z(x): calculate z for given t
    z = lambda t: (t + 1j*gamma)/(sigma + np.sqrt(2))
    A = amp/(scipy.special.wofz(z(0.))).real
    
    w = scipy.special.wofz(z(np.array(x)-pos))
    return A * w.real

def SplitGaussian(x, p):
    is_iterable = False
    try:
        iter(x)
    except TypeError:
        is_iterable = False
        xa = np.array([x])
    else:
        is_iterable = True
        xa = np.array(x)
    
    y = []
    
    for xe in xa:
        if xe < p[1]:
            y.append(p[0]*np.exp(-np.log(2.) * np.power(((np.array(xe)-p[1])/(p[2]/2.)), 2.)))
        elif xe == p[1]:
            y.append(p[0])
        else:
            y.append(p[0]*np.exp(-np.log(2.) * np.power(((np.array(xe)-p[1])/(p[3]/2.)), 2.)))

    if is_iterable:
        return np.array(y)
    else:
        return y[0]

def EMG(x, p):
    return (
            (p[0]/2*p[3]) 
            * np.exp(
                     np.power(p[2], 2)/(2*np.power(p[3], 2))
                     + (p[1] - x)/p[3]
                     )
            * (
               scipy.special.erf(
                                 (x-p[1])/(np.sqrt(2.)*p[2])
                                 - p[2]/(np.sqrt(2.)*p[3])
                                 )
               + p[3]/np.abs(p[3])
               )
            )

def PseudoVoigt(x, p):
    '''
    Gaussian-Lorentzian Sum (Amplitude), from PeakFit v4.12 Manual 7-11
    '''
    return (
            p[0]
            * (
               p[3]*np.sqrt(np.log(2.))/(p[2]*np.sqrt(np.pi))
               * np.exp(-4*np.log(2.)*np.power((x-p[1])/p[2], 2.))
               + (1-p[3])/(np.pi*p[2]*(1+4*np.power((x-p[1])/p[2], 2.)))
               )
            / (
               p[3]*np.sqrt(np.log(2.))/(p[2]*np.sqrt(np.pi))
               + (1-p[3])/(np.pi*p[2])
               )
            )

def Parabolic(x, p):
    '''
    p[0]: height (>0)
    p[1]: position
    p[2]: full width at half maximum (fwhm)
    '''
    return p[0]*(1. - 2*np.power((x-p[1])/p[2], 2.))

def AreaGaussian(p):
    return (p[0]*p[2]/2.) * math.sqrt(math.pi/math.log(2.))

def AreaLorentzian(p):
    return math.pi*p[0]*p[2]

def AreaPseudoVoigt(p):
    '''
    I think it's ok...
    '''
    return p[0]/2.0/(
                     p[3]*np.sqrt(np.log(2.))/(p[2]*np.sqrt(np.pi))
                     + (1-p[3])/(np.pi*p[2])
                     )

class Function(object):
    """
    """
    
    def __init__(self, expr, params, share=[], lock=[], headers=[], **lineprops):
        '''
        @param expr: callable function with expr(x, p) where p is a list of parameters. 
        @param params: initial list of parameters. 
        @param share: indices of parameters to be shared (self.Share() is also available. ). 
        @param lock: indices of parameters to be locked (self.Lock() is also available. ). 
        @param headers: list of strings describing the role of parameters. 
        @raise AssertionError: 
        '''
        
        self.expr = expr
        self.params = params
        self.errors = [0.0 for p in params] # added 2017-05-07
        self.locked = []
        self.shared = []
        self.bounds = []
        
        if not headers:
            self.headers = [str(n) for n in range(len(self.params))]
        else:
            self.headers = headers
        
        for p in self.params:
            self.bounds.append((-np.inf, np.inf))
        self.lineprops = lineprops
        
        self.Lock(*lock)
        self.Share(*share)
    
    def EvaluateWithParam(self, x, params):
        '''
        Get the value of the function at x with given parameters. 
        NOTE: This method doesn't overwrite the parameters held by this class. 
        @return: float number
        '''
        return self.expr(x, params)
    
    def EvaluateWithPartialParam(self, x, *params):
        '''
        Get the value of the function at x with given parameters. 
        This method doesn't overwrite the parameters held by this class. 
        Unlike EvaluateWithParam(), this method accepts a "partial" parameter 
        excluding locked values. 
        '''
        params_full = self.Decode(params)
        return self.expr(x, params_full)
    
    def Evaluate(self, x):
        '''
        Get the value of the function at x with current parameters. 
        @return: float number
        '''
        return self.expr(x, self.params)
    
    def Update(self, params, errors=None):
        '''
        Overwrite parameters. 
        '''
        self.params = list(params)
        if errors == None:
            self.errors = [0.0 for p in self.params]
        else:
            self.errors = list(errors)
    
    def Lock(self, *args):
        '''
        Lock the parameter(s). Locked parameters don't change during fitting. 
        @param *args: index (or indices) of the parameters to be locked. NOT A LIST OF INDICES. 
        '''
        for num in args:
            assert num < len(self.params)
            self.locked.append(num)
    
    def Share(self, *args):
        '''
        Specify the parameter(s) to be shared. 
        @param *args: index (or indices) of the parameters to be shared. NOT A LIST OF INDICES.
        '''
        for num in args:
            assert num < len(self.params)
            self.shared.append(num)
    
    def Bound(self, ind, bound=(-np.inf, np.inf)):
        '''
        Specify the boundary for the parameter. 
        @param ind: index of the target parameter
        @param bound: two-tuple of the lower and upper boundary. 
        '''
        self.bounds[ind] = bound
    
    def Unlock(self, *args):
        for num in args:
            assert num < len(self.params)
            try:
                self.locked.remove(num)
            except ValueError:
                pass

    def Unshare(self, *args):
        for num in args:
            assert num < len(self.params)
            try:
                self.shared.remove(num)
            except ValueError:
                pass
    
    def Unbound(self, ind):
        try:
            self.bounds[ind] = (-np.inf, np.inf)
        except IndexError:
            pass
    
    def Integrate(self, xi, xf):
        ret = scipy.integrate.quad(self.Evaluate, xi, xf)
        return ret[0]
    
    def GetParameterLength(self):
        '''
        Get the number of parameters needed (including locked/shared parameters). 
        '''
        return len(self.params)
    
    def GetPlotData(self, start, stop, num=1000, xlog=False, baseline=None):
        '''
        Make the arrays for plotting functions. 
        @param start: starting value for x
        @param stop: stopping value for x
        @param num: number of points
        @param xlog: If True, x will be evenly spaced in log scale. 
        @param baseline: another Function() instance
        @return: (array of x, array of y)
        '''
        if xlog:
            xs = np.logspace(np.log10(start), np.log10(stop), num=num)
        else:
            xs = np.linspace(start, stop, num=num)
            
        if baseline:
            assert isinstance(baseline, Function)
            ys = self.Evaluate(xs) + baseline.Evaluate(xs)
        else:
            ys = self.Evaluate(xs)
        return np.array(xs), np.array(ys)
    
    def GetLineProps(self):
        return self.lineprops
    
    def GetParamsHeadersString(self, fmt="{0:>15s}", sep="\t"):
        '''
        Returns a formatted string of parameter headers. 
        '''
        return sep.join([fmt.format(h) for h in self.headers])
    
    def GetParamsString(self, fmt="{0:>15f}", sep="\t"):
        return sep.join([fmt.format(h) for h in self.params])
    
    def Encode(self):
        '''
        Makes the "partial" parameters & bounds excluding locked parameters, 
        from the currently-set parameters & errors.
        @return: two-tuple of lists 
        '''
        params_part = []
        bounds_part = []
        
        for i in range(len(self.params)):
            if i in self.locked:
                pass
            else:
                params_part.append(self.params[i])
                bounds_part.append(self.bounds[i])
        
        return params_part, bounds_part
    
    def Decode(self, params):
        '''
        Makes the "full" parameters containing locked parameters, 
        from the "partial" parameters excluding locked ones. 
        '''
        params_full = list(self.params) # copy
        j = 0 # counter for parameters excluding locked ones
        for i in range(len(params_full)):
            if i in self.locked:
                pass
            else:
                params_full[i] = params[j]
                j = j + 1
        
        return params_full

    def DecodeErrors(self, errors):
        '''
        Makes the "full" parameters containing locked parameters, 
        from the "partial" parameters excluding locked ones. 
        '''
        errors_full = list(self.errors) # copy
        j = 0 # counter for parameters excluding locked ones
        for i in range(len(errors_full)):
            if i in self.locked:
                pass
            else:
                errors_full[i] = errors[j]
                j = j + 1
        
        return errors_full
    
    def UpdateParameter(self, params, errors):
        '''
        Updates the currently-set parameter & error info
        with the given "partial" parameter & error excluding locked values. 
        '''
        self.params = self.Decode(params)
        self.errors = self.DecodeErrors(errors)
            
    def Fit(self, xs, ys, sigma=None, **kwargs):
        '''
        Fit the model to given data set. 
        @param xs: array of x
        @param ys: array of y
        @param sigma: array of std. dev., optional
        '''
        
        p0, b = self.Encode()
        
        bounds = list(zip(*b))
        
        try:
            popt, pcov = scipy.optimize.curve_fit(f=self.EvaluateWithPartialParam, 
                                                   xdata=xs, 
                                                   ydata=ys, 
                                                   p0=p0, 
                                                   bounds=bounds, 
                                                   sigma=sigma, 
                                                   absolute_sigma=False, **kwargs)
        except (RuntimeError, OptimizeWarning, ValueError):
            raise
        else:
            # calculate error
            errors = [0.0 for p in popt]
            for i in range(len(popt)):
                errors[i] = np.sqrt(np.absolute(pcov[i][i]))
            
            self.UpdateParameter(popt, errors)


class Residuals(object):
    '''
    A class for general least-square optimization. 
    It calculates the sum of squared residuals and try to minimize it 
    with respect to the parameters, starting from p0. 
    '''
    
    def __init__(self, residuals_function, p0):
        self.residuals_function = residuals_function
        self.params = p0
        self.errors = [0.0 for p in p0]
        self.bounds = [[-np.inf, np.inf] for p in p0]
        
        self.covmat = np.zeros((len(p0), len(p0)))
    
    def Bound(self, ind, bound=(-np.inf, np.inf)):
        '''
        Specify the boundary for the parameter. 
        @param ind: index of the target parameter
        @param bound: two-tuple of the lower and upper boundary. 
        '''
        self.bounds[ind] = bound
    
    def Fit(self):
        res = scipy.optimize.least_squares(self.residuals_function, self.params, bounds=([b[0] for b in self.bounds], [b[1] for b in self.bounds]))
        
        # calculate standard deviation of each free params
        # using Jacobian
        # this is somewhat empirical although there should be a solid theoretical basis
        JTJ = np.matmul(res.jac.T, res.jac)
        JTJinv = np.linalg.inv(JTJ)
        
        data_length = len(self.residuals_function(res.x))
        
        s_sq = res.cost/(data_length - len(self.params))
            
        pcov = JTJinv * s_sq * 2.0
        e_free = [np.sqrt(np.absolute(pcov[i,i])) for i in range(len(self.params))]
        
        self.params = [v for v in res.x]
        self.errors = e_free
        self.covmat = pcov
            
    

class Baseline(Function):
    '''
    A class for baseline. 
    '''
    
    def __init__(self, xdata, ydata, x_range=[]):
        '''
        @param xdata, ydata: lists
        @param x_range: list 
        '''
        if len(x_range) == 2:
            i0, i1 = [np.argmin(np.abs(np.array(xdata)-target)) for target in x_range]
            x0 = xdata[i0]
            x1 = xdata[i1]
            y0 = ydata[i0]
            y1 = ydata[i1]
        else:
            x0 = xdata[0]
            x1 = xdata[-1]
            y0 = ydata[0]
            y1 = ydata[-1]
    
        a = (y1-y0)/(x1-x0)
        b = (x1*y0 - x0*y1)/(x1-x0)
        
        Function.__init__(self, Line, [a, b], lock=[0, 1])

class BaselineStable(Function):
    
    def __init__(self, xdata, ydata, left, right, hwidth):
        
        xarr = np.array(xdata)
        yarr = np.array(ydata)
                
        i_left = np.argmin(np.abs(xarr-left))
        i_right = np.argmin(np.abs(xarr-right))
        
        hwidth_lim = max([i_left, len(xdata) - i_right])
        if hwidth > hwidth_lim:
            raise ValueError("hwidth too large")

        x0 = left
        x1 = right
        y0 = np.average(yarr[i_left -hwidth:i_left +hwidth+1])
        y1 = np.average(yarr[i_right-hwidth:i_right+hwidth+1])
                
        a = (y1-y0)/(x1-x0)
        b = (x1*y0 - x0*y1)/(x1-x0)
        
        Function.__init__(self, Line, [a, b], lock=[0, 1])

class Model(object):
    '''
    A class for fitting model function composed of multiple functions. 
    '''
    
    def __init__(self, funcs=[]):
        '''
        @param funcs: a list of Function() objects. 
        @param axes: Reference to matplotlib axes. 
        '''
        self.funcs = []
        self.map = []
        self.parameter_length = 0
        for f in funcs:
            self.AddFunction(f)
                    
    def AddFunction(self, func):
        ''' 
        add a function to the model 
        @param func: Function object. 
        '''
        self.funcs.append(func)
        self.UpdateEncoderMap()
        
    def UpdateEncoderMap(self):
        '''
        Make a map for encoding parameters. 
        '''
        shared = {} # buffer for shared parameters
        counter = 0 # increments when new parameter is added 
        m_f = [None]*len(self.funcs) # the whole map
        
        # for each function (i: function number)
        for i, f in enumerate(self.funcs):
            m_p = [None]*f.GetParameterLength() # a part of the map corresponding to the current function
            # for each parameter (j: parameter number)
            for j in range(f.GetParameterLength()):
                # if jth parameter is locked, None is inserted to the map. 
                if j in f.locked:
                    m_p[j] = None
                    continue
                elif j in f.shared:
                    # if jth parameter is shared and already in the buffer, 
                    # the index from the buffer is inserted to the map. 
                    if j in shared.keys():
                        m_p[j] = shared[j]
                        continue
                    # if jth parameter is shared for the first time, 
                    # the current counter is inserted to the map and registered to the buffer. 
                    else:
                        m_p[j] = counter
                        shared[j] = counter
                        counter += 1
                        continue
                # if jth parameter is neither locked nor shared, 
                # the current counter is inserted to the map. 
                else:
                    m_p[j] = counter
                    counter += 1
                    continue
            m_f[i] = m_p
        
        # update 
        self.map = m_f
        self.parameter_length = counter
    
    def Encode(self):
        '''
        Return a list of current parameters (considering locked and shared parameters)
        @return: a list of parameters and a list of two-tuples of bounds
        
        20160930 added bounds tuples
        '''
        v = [None]*self.parameter_length # a list of parameters to be returned 
        b = [None]*self.parameter_length # a list of bounds tuples
        # for each function
        for i in range(len(self.funcs)):
            f = self.funcs[i]
            # for each parameter
            for j in range(f.GetParameterLength()):
                # get an appropriate position from self.map
                pos = self.map[i][j]
                # if it's None, i.e. it's locked, skip it.  
                if pos == None:
                    continue
                # if it's not None and v[pos] has no value yet, insert the current parameter. 
                elif v[pos] == None:
                    v[pos] = f.params[j]
                    b[pos] = f.bounds[j]
        
        return v, b
    
    def Decode(self, params):
        '''
        Chop the given parameters into parameter lists for each function. 
        @param params: a list of parameters
        @return: a list of lists of parameters (ret[i][j]: jth parameter of ith function)
        '''
        v_f = [None]*len(self.funcs) # a list of lists
        # for each function
        for i in range(len(self.funcs)):
            f = self.funcs[i]
            v_p = [None]*f.GetParameterLength() # a list
            # for each parameter
            for j in range(f.GetParameterLength()):
                # get an appropriate position from self.map
                pos = self.map[i][j]
                # if it's locked, use the value from original function. 
                if pos == None:
                    v_p[j] = f.params[j]
                # if it's not locked, use the value from "params". 
                else:
                    v_p[j] = params[pos]
            
            v_f[i] = v_p
        
        return v_f # a list of lists
    
    def UpdateParameters(self, params, errors=None):
        '''
        Update the parameters with given parameter list. 
        @param params: a list of parameters
        '''
        # "decode" the parameter list
        decoded_params = self.Decode(params)
        if errors != None:
            decoded_errors = self.Decode(errors)
        # for each function
        for i in range(len(self.funcs)):
            # "decoded" parameter can be passed directly to Function object. 
            if errors != None:
                self.funcs[i].Update(decoded_params[i], decoded_errors[i])
            else:
                self.funcs[i].Update(decoded_params[i], errors=None)
    
    def EvaluateWithParams(self, x, *params):
        '''
        Evaluate the model using x and given parameters. 
        Note that the parameters given here do not affect 
        the parameters of this instance. 
        To update the parameters of the instance, 
        call self.UpdateParameters. 
        @param x: 
        @param *params: parameters
        '''
        # "decode" the parameter list
        decoded = self.Decode(params)
        val = 0. # a value to be returned
        # for each function
        for i, f in enumerate(self.funcs):
            # the function will be evaluated without updating parameters. 
            val = val + f.EvaluateWithParam(x, decoded[i])
            
        return val
    
    def Evaluate(self, x):
        '''
        Evalute the model using currently set parameters. 
        @param x: x
        '''
        val = 0.
        # for each function
        for i, f in enumerate(self.funcs):
            # evaluate the function with current set of parameters. 
            val = val + f.Evaluate(x)
        return val
    
    def GetPlotData(self, start, stop, num=1000, xlog=False, ind=None, baseline=None):
        '''
        Make the arrays for plotting functions. 
        @param start: starting value for x
        @param stop: stopping value for x
        @param num: number of points
        @param xlog: use x evenly spaced in log space if True. 
        @param ind: index of a function to be plotted. If None, the data for whole model will be evaluated.
        @param baseline: another Function() object
        @return: (array of x, array of y)
        '''
        if xlog:
            xs = np.logspace(np.log10(start), np.log10(stop), num=num)
        else:
            xs = np.linspace(start, stop, num=num)
            
        if ind == None:
            ys = self.Evaluate(xs)
        else:
            ys = self.funcs[ind].Evaluate(xs)
        
        if baseline:
            assert isinstance(baseline, Function)
            ys = ys + baseline.Evaluate(xs)
        
        return xs, ys
        
    def Fit(self, xs, ys, sigma=None):
        '''
        Fit the model to given data set. 
        @param xs: array of x
        @param ys: array of y
        @param sigma: array of std. dev., optional
        '''
        
        p0, b = self.Encode()
        
        bounds = list(zip(*b))
        
        popt, pcov = scipy.optimize.curve_fit(f=self.EvaluateWithParams, 
                                                xdata=xs, 
                                                ydata=ys, 
                                                p0=p0, 
                                                bounds=bounds, 
                                                sigma=sigma, 
                                                absolute_sigma=False)

        # calculate error
        errors = [0.0 for p in popt]
        for i in range(len(popt)):
            errors[i] = np.sqrt(np.absolute(pcov[i][i]))
        self.UpdateParameters(popt, errors)
    
if __name__ == '__main__':
    
    import snpl
    
    x = np.linspace(0.0, 100.0, 20)
    y = Gaussian(x, [5.0, 50.0, 30.0]) + np.random.rand(len(x)) - 0.5
    
    p0 = [1.0, 50.0, 10.0]
    
    # f = Function(Gaussian, [1.0, 50.0, 10.0])
    # f.Fit(x, y)
    
    g = lambda x, amp, pos, fwhm: Gaussian(x, [amp, pos, fwhm])
    
    popt, pcov = scipy.optimize.curve_fit(g, x, y, p0)
    
    residuals = lambda p: Gaussian(x, p) - y
    square = lambda p: np.sum(np.power(residuals(p), 2))
    
    res = scipy.optimize.least_squares(residuals, x0=p0)
    
    print("popt", popt)
    print("pcov", pcov)
    print("res.x", res.x)
        
    JTJ = np.matmul(res.jac.T, res.jac)
    JTJinv = np.linalg.inv(JTJ)
    
    s_sq = res.cost/(len(x) - len(p0))
        
    print(JTJinv * s_sq * 2.0)