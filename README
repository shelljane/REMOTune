# REMOTune: Random Embedding and Multi-objective Trust-region Bayesian Optimization for VLSI Flow Parameter Tuning

## Dependency: 

### botorch (https://botorch.org/)

### optuna (https://optuna.org/)

## Run REMOTune

'''
python3 opt/remotune.py -p script/params.txt -c script/riscv32i_syn.py -d script/riscv32i_cad.py -r 150.0 -e 16 -n 3 -b 8 -i 64 -s 256 -t 1800 -o tmp -j 16 -m 4 --scale 0.25
'''

## Run MOTPE/BO

'''
python3 opt/motpe.py -p script/params.txt -c script/riscv32i_cad.py -r 150.0 -n 3 -i 16 -s 64 -t 1800 -o tmp
python3 opt/bo.py -p script/params.txt -c script/riscv32i_cad.py -r 150.0 -n 3 -i 16 -s 64 -t 1800 -o tmp
'''

## Note

The parameters for Genus and Innovus is in script/params.txt

The code in script/genus.py and script/innovus.py are used to generate the scripts for Genus and Innovus, respectively. 

Note that we need the .lib, .lef, .captable files, Genus, and Innovus to run the code. 

MOTPE is the method used in AutoTuner. 

## TODO

We are going to support OpenROAD soon. We will release the full code that can be evaluated on OpenROAD.  
