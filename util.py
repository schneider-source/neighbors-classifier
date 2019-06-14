"""
Created on Wed Feb 6 08:15:23 2019
@author: Daniel Schneider

Utility functions
"""

import os
import sys
import shutil


def mkdir(dirname, replace=False):
    """
    Create directory.
    :param dirname: str, "name of the directory to be created"
    :param replace: bool, optional, "if already exists, replace old directory"
    :return: str, "unchanged param dirname"
    """

    assert(isinstance(dirname, str))
    assert(isinstance(replace, bool))

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        if replace:
            assert(os.path.isdir(dirname))
            shutil.rmtree(dirname)
            os.makedirs(dirname)

    return dirname


def aprint(*args, func=print, fmt='d', fname=None, **kwargs):
    """
    Advanced printing with certain colors and font styles. Printing to files
    is also possible.
    :param args: "args to pass to param func"
    :param func: functional, optional, "function that prints stuff"
    :param fmt: str, optional, one out of fmts, "format indicator"
    :param fname: str, optional, "filename to print to", if set, then param
        fmt is ignored
    :param kwargs: "kwargs to pass to param func"
    """

    fmts = {'h': '\x1b[42m\x1b[90m',  # heading
            'bh': '\x1b[1m\x1b[42m\x1b[90m',  # bold heading
            'i': '\x1b[32m',  # info
            'bi': '\x1b[1m\x1b[32m',  # bold info
            'w': '\x1b[93m',  # warning
            'bw': '\x1b[1m\x1b[93m',  # bold warning
            'e': '\x1b[31m',  # error
            'be': '\x1b[1m\x1b[31m',  # bold error
            'b': '\x1b[1m',  # bold
            'd': '\x1b[0m'}  # default

    assert(callable(func))
    assert(fmt in fmts.keys())
    assert(isinstance(fname, str) or fname is None)

    if fname is None:
        # print to console in specified format
        sys.stdout.write(fmts[fmt])
        func(*args, **kwargs)
        sys.stdout.write(fmts['d'])
    else:
        # print to file
        with open(fname, 'a+') as sys.stdout:
            func(*args, **kwargs)
        sys.stdout = sys.__stdout__


