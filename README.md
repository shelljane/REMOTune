# REMOTune: Random Embedding and Multi-objective Trust-region Bayesian Optimization for VLSI Flow Parameter Tuning

This is the official repo for the paper: 

Su Zheng, Hao Geng, Chen Bai, Bei Yu, Martin Wong, “Boosting VLSI Design Flow Parameter Tuning with Random Embedding and Multi-objective Trust-region Bayesian Optimization”, ACM Transactions on Design Automation of Electronic Systems (TODAES), vol. 28, no. 05, pp. 1–23, 2023. 

## Dependency: 

### botorch (https://botorch.org/) for REMOTune

### optuna (https://optuna.org/) for MOTPE

### platypus for (https://platypus.readthedocs.io/en/latest/) for PTPT

## Run REMOTune

'''
python3 opt/remotune.py -p script/params.txt -c script/riscv32i_syn.py -d script/riscv32i_cad.py -r 150.0 -e 16 -n 3 -b 8 -i 64 -s 256 -t 1800 -o tmp -j 16 -m 4 --scale 0.25
'''

## Run MOTPE

'''
python3 opt/motpe.py -p script/params.txt -c script/riscv32i_cad.py -r 150.0 -n 3 -i 16 -s 64 -t 1800 -o tmp
'''

## Run BO

'''
python3 opt/bo.py -p script/params.txt -c script/riscv32i_cad.py -r 150.0 -n 3 -i 16 -s 64 -t 1800 -o tmp
'''

## Run PTPT

'''
python3 opt/ptpt.py -p script/params.txt -c script/riscv32i_cad.py -r 150.0 -n 3 -i 16 -s 64 -t 1800 -o tmp
'''

## Note

The parameters for Genus and Innovus is in script/params.txt. If you want to explore Genus (logic synthesis) or Innovus (physical design) individually, you may use script/genus.txt for Genus and script/innvous.txt for Innovus.  

The code in script/genus.py and script/innovus.py are used to generate the scripts for Genus and Innovus, respectively. 

Note that we need the .lib, .lef, .captable files, Genus, and Innovus to run the code. 

MOTPE is the method used in AutoTuner. 

## OpenROAD Support

We have supported OpenROAD, see opt/remoroad.py and opt/tperoad.py. The "--flowdir" option specifies the path of OpenROAD-flow-script. 

## Citation

```
@article{zheng2023boosting,
  title={Boosting VLSI Design Flow Parameter Tuning with Random Embedding and Multi-objective Trust-region Bayesian Optimization},
  author={Zheng, Su and Geng, Hao and Bai, Chen and Yu, Bei and Wong, Martin DF},
  journal={ACM Transactions on Design Automation of Electronic Systems},
  volume={28},
  number={5},
  pages={1--23},
  year={2023},
  publisher={ACM New York, NY}
}
```
