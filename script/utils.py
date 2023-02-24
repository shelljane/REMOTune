import os
import sys 
sys.path.append(".")

import time
import numpy as np 
import subprocess as sp


def getQoR(nameVars, variables, precfg, script, baseline, clkref, basedir="run"): 
    configs = precfg.copy()
    for idx, name in enumerate(nameVars): 
        configs[name] = variables[idx]
    basedir = basedir + "/time" + str(time.time()).replace(".", "")
    configs["output"] = basedir
    outfile = basedir + "/result.txt"
    try: 
        runPythonCommand(script, configs, timeout=None, outfile=outfile)
        results = parseResult(outfile, clkref)
    except Exception as e: 
        print("[EVALUATION]: FAILED or TIMEOUT")
        print(e)
        results = ["ERR", "ERR", "ERR"]
    portion = ["ERR", "ERR", "ERR"]
    if not "ERR" in results: 
        for idx in range(len(portion)): 
            portion[idx] = 100 + 100 * ((results[idx] - baseline[idx]) / baseline[idx])
    os.system("rm -rf " + basedir + " > /dev/null 2> /dev/null")
    return portion


def parseResult(filename, clkref=2.5): 
    slack, power, area = "ERR", "ERR", "ERR"
    with open(filename, "r") as fin: 
        lines = fin.readlines()
        for line in lines: 
            splited = line.strip().split()
            if len(splited) >= 3: 
                try: 
                    slack = clkref - float(splited[0])
                    power = float(splited[1])
                    area  = float(splited[2])
                except ValueError: 
                    slack = "ERR"
                    power = "ERR"
                    area  = "ERR"
    return [slack, power, area]


def evalQoR(nameVars, variables, baseline, iter, script, clkref, precfg, visited): 
    print("[ITERATION]:", iter, ";", variables)
    configs = precfg.copy()
    for idx, name in enumerate(nameVars): 
        configs[name] = variables[idx]
    basedir = "run/iter_" + str(iter)
    configs["output"] = basedir
    outfile = basedir + "/result.txt"
    outfile = "tmp.txt"
    try: 
        runPythonCommand(script, configs, timeout=None, outfile=outfile)
        results = parseResult(outfile, clkref)
    except Exception: 
        print("[EVALUATION]: TIMEOUT")
        results = ["ERR", "ERR", "ERR"]
    portion = ["ERR", "ERR", "ERR"]
    if not "ERR" in results: 
        for idx in range(len(portion)): 
            portion[idx] = 100 + 100 * ((results[idx] - baseline[idx]) / baseline[idx])
    try: 
        with open(visited, "a+") as fout: 
            fout.write("Visited\n")
            fout.write(str(nameVars) + "\n")
            fout.write(str(variables) + "\n")
            fout.write(str(results) + "\n")
    except Exception: 
        pass
    return portion


def evaluate(nameVars, variables, baseline, iter): 
    print("[ITERATION]:", iter, ";", variables)
    configs = {}
    for idx, name in enumerate(nameVars): 
        configs[name] = variables[idx]
    basedir = "run/iter_" + str(iter)
    configs["output"] = basedir
    outfile = basedir + "/result.txt"
    outfile = "tmp.txt"
    try: 
        runPythonCommand("cmd/evaluate.py", configs, timeout=600, outfile=outfile)
        results = parseResult(outfile)
    except Exception: 
        print("[EVALUATION]: TIMEOUT")
        results = ["ERR", "ERR", "ERR"]
    portion = ["ERR", "ERR", "ERR"]
    if not "ERR" in results: 
        for idx in range(len(portion)): 
            portion[idx] = 100 + 100 * ((results[idx] - baseline[idx]) / baseline[idx])
    try: 
        with open("visited.txt", "a+") as fout: 
            fout.write("Visited\n")
            fout.write(str(nameVars) + "\n")
            fout.write(str(variables) + "\n")
            fout.write(str(results) + "\n")
    except Exception: 
        pass
    return portion


def cadenceParse(filename, clkref=2.5): 
    slack, power, area = "ERR", "ERR", "ERR"
    with open(filename, "r") as fin: 
        lines = fin.readlines()
        for line in lines: 
            splited = line.strip().split()
            if len(splited) >= 3: 
                try: 
                    slack = clkref-float(splited[0])
                    power = float(splited[1])
                    area  = float(splited[2])
                except ValueError: 
                    slack = "ERR"
                    power = "ERR"
                    area  = "ERR"
    return [slack, power, area]


def cadenceEval(nameVars, variables, baseline, iter): 
    print("[ITERATION]:", iter, ";", variables)
    configs = {}
    for idx, name in enumerate(nameVars): 
        configs[name] = variables[idx]
    basedir = "run/iter_" + str(iter)
    configs["output"] = basedir
    outfile = basedir + "/result.txt"
    outfile = "tmp.txt"
    try: 
        configs["hdl"] = "src/gcd.v"
        configs["sdc"] = "src/gcd.sdc"
        runPythonCommand("cmd/cadence.py", configs, timeout=600, outfile=outfile)
        results = cadenceParse(outfile)
    except Exception: 
        print("[EVALUATION]: TIMEOUT")
        results = ["ERR", "ERR", "ERR"]
    portion = ["ERR", "ERR", "ERR"]
    if not "ERR" in results: 
        for idx in range(len(portion)): 
            portion[idx] = 100 + 100 * ((results[idx] - baseline[idx]) / baseline[idx])
    try: 
        with open("visited.txt", "a+") as fout: 
            fout.write("Visited\n")
            fout.write(str(nameVars) + "\n")
            fout.write(str(variables) + "\n")
            fout.write(str(results) + "\n")
    except Exception: 
        pass
    return portion


def readTrials(filename): 
    with open(filename, "r") as fin: 
        lines = fin.readlines()
    
    configs = []
    scores = []
    idx = 0
    while idx < len(lines): 
        while idx < len(lines) and lines[idx].strip() != "Visited": 
            idx += 1
        idx += 1
        if idx + 2 >= len(lines): 
            break
        nameVars  = eval(lines[idx].strip())
        variables = eval(lines[idx + 1].strip())
        results   = eval(lines[idx + 2].strip())
        idx += 2
        
        assert len(nameVars) == len(variables)
        config = {}
        for jdx, name in enumerate(nameVars): 
            config[name] = variables[jdx]
        configs.append(config)
        scores.append(results)
        
    return configs, scores


def readConfig(filename): 
    names = []
    types = []
    ranges = []
    with open(filename, "r") as fin: 
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            splited = line.split()
            if len(splited) < 3: 
                continue
            name = splited[0]
            typename = splited[1]
            values = splited[2:]
            for idx in range(len(values)): 
                if typename == "int": 
                    values[idx] = int(values[idx])
                elif typename == "float": 
                    values[idx] = float(values[idx])
            names.append(name)
            types.append(typename)
            ranges.append(values)
    return names, types, ranges
    

def runPythonCommand(filename, configs, timeout=None, outfile=None): 
    command = ["python3", filename, ]
    for key, value in configs.items(): 
        if isinstance(value, list) or isinstance(value, tuple): 
            for elem in value: 
                command.append("--" + key)
                command.append(str(elem))
        else: 
            command.append("--" + key)
            command.append(str(value))
    fout = sp.DEVNULL
    if not outfile is None: 
        fout = sp.PIPE
    ret = sp.run(command, timeout=timeout, shell=False, stdout=fout, stderr=fout)
    if not outfile is None: 
        with open(outfile, "w") as fout: 
            if ret.stdout.strip() != "": 
                fout.write(ret.stdout.decode("UTF-8"))
            if ret.stderr.strip() != "": 
                fout.write(ret.stderr.decode("UTF-8"))
    return ret.returncode
    

def dominate(a, b): 
    assert len(a) == len(b)
    domin1 = True
    domin2 = False
    for idx in range(len(a)): 
        if a[idx] > b[idx]: 
            domin1 = False
        elif a[idx] < b[idx]: 
            domin2 = True
    return domin1 and domin2
    

def newParetoSet(paretoParams, paretoValues, newParams, newValue): 
    assert len(paretoParams) == len(paretoValues)
    dupli = False
    removed = set()
    indices = []
    for idx, elem in enumerate(paretoValues): 
        if str(paretoParams[idx]) == str(newParams): 
            dupli = True
            break
        if dominate(newValue, elem): 
            removed.add(idx)
    if dupli: 
        return paretoParams, paretoValues
    for idx, elem in enumerate(paretoValues): 
        if not idx in removed: 
            indices.append(idx)
    newParetoParams = []
    newParetoValues = []
    for index in indices: 
        newParetoParams.append(paretoParams[index])
        newParetoValues.append(paretoValues[index])
    bedominated = False
    for idx, elem in enumerate(newParetoValues): 
        if dominate(elem, newValue): 
            bedominated = True
    if len(removed) > 0:
        assert not bedominated
    if len(removed) > 0 or len(paretoParams) == 0 or not bedominated: 
        newParetoParams.append(newParams)
        newParetoValues.append(newValue)
    return newParetoParams, newParetoValues
    

def pareto(params, values): 
    paretoParams = []
    paretoValues = []

    for var, objs in zip(params, values): 
        paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, var, objs)

    return paretoParams, paretoValues
    

if __name__ == "__main__": 
    paretoParams = [0, 1, 2, 3, 4, 5, 6, 7]
    paretoValues = [[70.76775036084953, 176.14678899082568, 72.04301075268818], [75.61802643938873, 125.6880733944954, 81.72043010752688], [70.04834238321075, 379.8165137614679, 66.66666666666667], [78.62625151785919, 141.28440366972475, 62.365591397849464], [81.19000160377574, 306.42201834862385, 50.53763440860215], [95.55754118267005, 106.42201834862385, 77.41935483870968], [82.08582491351066, 107.33944954128441, 60.215053763440864], [78.01681673425435, 117.43119266055047, 63.44086021505376]]

    newParam = 8
    newValue = [89.75416408916993, 97.24770642201834, 26.88172043010752]

    paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, newParam, newValue)
    print(paretoParams, paretoValues)
    




