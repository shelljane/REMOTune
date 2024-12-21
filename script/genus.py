import os
import sys
import argparse

from utils import *

def parseArgs(filename="script/genus.txt"): 
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
    parser.add_argument("--lef", required=False, 
                        action="append", \
                        default=[]) # lib/tcbn65lp_6lmT1.lef
    parser.add_argument("--cap", required=False, 
                        action="append", \
                        default=[]) # cln65lp_1p06m+alrdl_top1_typical.captable
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


def run(basedir, timeout=None): 
    prefix = "timeout %d " % (int(timeout), ) if not timeout is None else ""
    os.system(prefix + "/opt2/cadence/GENUS171/bin/genus -no_gui -log %s -batch -files %s > %s 2>&1" % \
              (basedir + "/log/genus", basedir + "/script/genus.tcl", "/dev/null"))


def result(timingfile, powerfile, areafile): 
    slack = None
    with open(timingfile, "r") as fin: 
        for line in fin.readlines(): 
            splited = line.strip().split()
            if len(splited) > 1 and splited[0] == "Slack:=": 
                slack = float(splited[1]) / 1000.0
        
    power = None # NOTE: get the total power
    with open(powerfile, "r") as fin: 
        lines = fin.readlines()
        for idx, line in enumerate(lines): 
            if line[0:4] == "----" and idx + 1 < len(lines): 
                splited = lines[idx+1].strip().split()
                if len(splited) > 4: 
                    power = float(splited[4]) / 1000000.0
        
    area = None
    with open(areafile, "r") as fin: 
        lines = fin.readlines()
        for idx, line in enumerate(lines): 
            if line[0:4] == "----" and idx + 1 < len(lines): 
                splited = lines[idx+1].strip().split()
                if len(splited) > 4: 
                    area = float(splited[4])
        
    return slack, power, area

def main(args, filename="script/genus.txt"): 

    basedir = None
    libfile = None
    leffile = None
    capfile = None
    hdlfile = None
    sdcfile = None
    topname = None
    timinglog = None
    arealog   = None
    powerlog  = None
    synthed_hdl = None
    synthed_sdc = None
    synthed_sdf = None
    for key, value in args.__dict__.items(): 
        if key == "output": 
            basedir = value
        elif key == "lib": 
            libfile = value
        elif key == "lef": 
            leffile = value
        elif key == "cap": 
            capfile = value
        elif key == "hdl": 
            hdlfile = value
        elif key == "sdc": 
            sdcfile = value
        elif key == "top": 
            topname = value
        elif key == "timing": 
            timinglog = value
        elif key == "area": 
            arealog = value
        elif key == "power": 
            powerlog = value
        elif key == "synthed_hdl": 
            synthed_hdl = value
        elif key == "synthed_sdc": 
            synthed_sdc = value
        elif key == "synthed_sdf": 
            synthed_sdf = value
    assert not basedir is None
    assert not libfile is None
    assert not leffile is None
    assert not capfile is None
    assert not hdlfile is None
    assert not sdcfile is None
    assert not topname is None
    assert not timinglog is None
    assert not arealog is None
    assert not powerlog is None
    assert not synthed_hdl is None
    assert not synthed_sdc is None
    assert not synthed_sdf is None
    if len(libfile) == 0: 
        libfile = ["lib/tcbn65lptc.lib"]
    if len(leffile) == 0: 
        leffile = ["lib/tcbn65lp_6lmT1.lef"]
    if len(capfile) == 0: 
        capfile = ["lib/cln65lp_1p06m+alrdl_top1_typical.captable"]
    if len(timinglog) == 0: 
        timinglog = basedir + "/log/timing_genus.log"
    if len(arealog) == 0: 
        arealog = basedir + "/log/area_genus.log"
    if len(powerlog) == 0: 
        powerlog = basedir + "/log/power_genus.log"
    if len(synthed_hdl) == 0: 
        synthed_hdl = basedir + "/script/synthed.v"
    if len(synthed_sdc) == 0: 
        synthed_sdc = basedir + "/script/synthed.sdc"
    if len(synthed_sdf) == 0: 
        synthed_sdf = basedir + "/script/synthed.sdf"
        
    if not os.path.exists(basedir): 
        os.mkdir(basedir)
    if not os.path.exists(basedir + "/script"): 
        os.mkdir(basedir + "/script")
    if not os.path.exists(basedir + "/log"): 
        os.mkdir(basedir + "/log")

    names, types, ranges = readConfig(filename)
    info = ""
    for key, value in args.__dict__.items(): 
        if not key in names: 
            continue
        info += "set_db " + key + " " + str(value) + "\n"

    info += "\n"

    info += "set_db library [list " + " ".join(libfile) + "]\n"
    info += "set_db lef_library [list " + " ".join(leffile) + "]\n"
    info += "set_db cap_table_file [list " + " ".join(capfile) + "]\n"
    info += "read_hdl [list " + " ".join(hdlfile) + "]\n"
    info += "elaborate\n"
    info += "current_design " + topname + "\n"
    info += "read_sdc [list " + " ".join(sdcfile) + "]\n"
    info += "uniquify " + topname + "\n"
    info += "syn_generic\n"
    info += "syn_map\n"
    info += "syn_opt\n"
    info += "report_timing > " + timinglog + "\n"
    info += "report_area > " + arealog + "\n"
    info += "report_power > " + powerlog + "\n"
    info += "write_hdl > " + synthed_hdl + "\n"
    info += "write_sdc > " + synthed_sdc + "\n"
    # info += "write_sdf > " + synthed_sdf + "\n"

    outfile = basedir + "/script/genus.tcl"
    with open(outfile, "w") as fout: 
        fout.write(info)

    run(basedir, args.timeout)
    res = result(timinglog, powerlog, arealog)
    if None in res: 
        res = ["ERR", "ERR", "ERR"]
    output = str(res[0]) + " " + str(res[1]) + " " + str(res[2])
    with open(basedir + "/log/genus_result.log", "w") as fout: 
        fout.write(output)
    print(output)

    # os.system("rm -rf " + basedir + "/log/genus.*")


if __name__ == "__main__": 
    main(parseArgs("script/genus.txt"))
