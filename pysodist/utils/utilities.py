# -*- coding: utf-8 -*-
"""
@author: Joey Davis <jhdavis@mit.edu> jhdavislab.org
@version: 0.0.4
"""

from datetime import datetime as dt
import sys
import os


def log(msg, outfile=None):
    msg = '{} --> {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(msg)
    sys.stdout.flush()
    if outfile is not None:
        try:
            with open(outfile, 'a') as f:
                f.write(msg + '\n')
        except Exception as e:
            log(e)

def check_dir(dirname, make = False):
    if not dirname.endswith('/'):
        dirname = dirname + '/'
    if make:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    return dirname