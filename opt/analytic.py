import os
import sys 
sys.path.append(".")

import argparse
import math
import numpy as np
from copy import deepcopy

from analytic_funcs import *

NARGS = 134
def parseArgs(): 
    parser = argparse.ArgumentParser()
    for idx in range(NARGS): 
        parser.add_argument("--x" + str(idx), required=True, action="store")
    return parser.parse_args()

def genConfig(filename): 
    info = ""
    for idx in range(NARGS): 
        info += "x" + str(idx) + " float 0 1.0\n"
    with open(filename, "w") as fout: 
        fout.write(info)

if __name__ == "__main__": 
    genConfig("patue/analytic.txt")
    exit(1)
    args = parseArgs()
    x = []
    for idx in range(NARGS): 
        name = "x" + str(idx)
        value = args.__dict__[name]
        x.append(float(value))
    print(str(Currin(x, NARGS) / 1e1), str(branin(x, NARGS) / 1e2), str(RASTRIGIN(x, NARGS) / 1e3))

# Initialization only: 14m27.865s
# Opt: 486m
