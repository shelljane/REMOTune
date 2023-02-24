import os
import sys
import argparse

from utils import *

def parseArgs(filename): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=False, 
                        action="store", \
                        default=".")
    parser.add_argument("--timeout", required=False, 
                        action="store", \
                        default=None)
    parser.add_argument("--lib", required=False, 
                        action="append", \
                        default=[]) # ./lib/tcbn65lptc.lib
    parser.add_argument("--typical", required=False, 
                        action="append", \
                        default=[])
    parser.add_argument("--best", required=False, 
                        action="append", \
                        default=[])
    parser.add_argument("--worst", required=False, 
                        action="append", \
                        default=[])
    parser.add_argument("--lef", required=False, 
                        action="append", \
                        default=[]) # lib/tcbn65lp_6lmT1.lef
    parser.add_argument("--cap", required=False, 
                        action="append", \
                        default=[]) # cln65lp_1p06m+alrdl_top1_typical.captable
    parser.add_argument("--mmmc", required=False, 
                        action="append", \
                        default=[])
    parser.add_argument("--hdl", required=True, 
                        action="append", \
                        default=[]) # top.v
    parser.add_argument("--sdc", required=True, 
                        action="append", \
                        default=[]) # top.sdc
    parser.add_argument("--top", required=True, 
                        action="store", \
                        default="") 
    parser.add_argument("--timing", required=False, 
                        action="store", \
                        default="")
    parser.add_argument("--area", required=False, 
                        action="store", \
                        default="")
    parser.add_argument("--power", required=False, 
                        action="store", \
                        default="")
    parser.add_argument("--drc", required=False, 
                        action="store", \
                        default="")
    parser.add_argument("--synthed_hdl", required=False, 
                        action="store", \
                        default="")
    parser.add_argument("--synthed_sdc", required=False, 
                        action="store", \
                        default="")
    parser.add_argument("--synthed_sdf", required=False, 
                        action="store", \
                        default="")
    names, types, ranges = readConfig(filename)
    for idx, name in enumerate(names): 
        if types[idx] == "enum": 
            parser.add_argument("--" + name, required=False, 
                                action="store", default=ranges[idx][0])
        elif types[idx] == "int": 
            parser.add_argument("--" + name, required=False, 
                                action="store", default=int((int(ranges[idx][0]) + int(ranges[idx][1])) / 2))
        elif types[idx] == "float": 
            parser.add_argument("--" + name, required=False, 
                                action="store", default=((float(ranges[idx][0]) + float(ranges[idx][1])) / 2))
        else: 
            assert 0
    return parser.parse_args()


import genus_sv
import innovus
def main(args): 
    genus_sv.main(args)
    innovus.main(args)
    # os.system("rm -rf " + args.output)


if __name__ == "__main__": 
    main(parseArgs("script/params.txt"))
