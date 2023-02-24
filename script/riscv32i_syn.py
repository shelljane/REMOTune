import sys
import glob

from utils import *

if __name__ == "__main__": 
    # HDL and SDC
    hdl = glob.glob("./riscv32i/*.v")
    sdc = "./riscv32i/constraint.sdc"
    # Mode
    # mode = "cadence"
    # configfile = "script/params.txt"
    mode = "genus"
    configfile = "script/genus.txt"
    names, types, ranges = readConfig(configfile)
    # Reference clock period
    clkref = 7.6
    # Baseline
    script = None
    precfg = None
    baseline = None
    if mode == "genus": 
        script = "script/genus.py"
        precfg = {"top": "riscv_top", "hdl": hdl, "sdc": sdc, }
        baseline = [7.441999999999999, 3.5443431409999997, 27722.88]
    elif mode == "innovus": 
        script = "script/innovus.py"
        precfg = {"synthed_hdl": hdl, "synthed_sdc": sdc, }
        baseline = [7.5889999999999995, 4.3468392, 28549.8]
    else: 
        script = "script/cadence.py"
        precfg = {"top": "riscv_top", "hdl": hdl, "sdc": sdc, }
        baseline = [7.5889999999999995, 4.3468392, 28549.8]
    
    if len(sys.argv) == 1: 
        basecfg = precfg.copy()
        basecfg["output"] = "baseline"
        runPythonCommand(script, basecfg, timeout=None, outfile="baseline.txt")
        baseline = parseResult("baseline.txt", clkref)
        print("[BASELINE]:", baseline)
    else: 
        nameVars = []
        variables = []
        for idx in range(1, len(sys.argv), 2): 
            assert sys.argv[idx][0:2] == "--"
            name = sys.argv[idx][2:]
            if not name in names: 
                continue
            nameVars.append(name)
            variables.append(sys.argv[idx+1])
        results = getQoR(nameVars, variables, precfg, script, baseline, clkref, basedir="run")
        print(results[0], results[1], results[2])




