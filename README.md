# Optimizing Neural Networks for PYNQ
Neil Kim Nielsen (neni@itu.dk), Robert Bayer (roba@itu.dk)

## Requirements
### Training and pruning
- Windows, Linux or macOS.
- Nvidia GPU (optional)
- Brevitas ^0.4.0
- FINN 0.5b

### Synthesis and deployment
- Linux
- Vivado 2019.1 or 2020.1
- FINN (from https://github.com/rbcarlos/finn) - part of the contributions of our thesis
- 32 GB of RAM
- XILINX ZYNQ-Z1 or Ultra96 (for deployment only)

## Training
To train the neural network, run the trainer.py script. Example:
```
BREVITAS_JIT=1 python3 trainer.py --lr 0.02 -b 128 -j 1 --epochs 2 --bit-width 8 --experiments model/ --dist-url 'tcp://127.0.0.1:23456' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```
This script supports multi-gpu training. 
To see the options for the script, run ```python3 trainer.py -h```

## Automatic tuning of folding factors
The `set_folding_exhaustive` notebook contains all the code necessary to generate the folding factors for a CNV model, composed of only convolutional and fully-connected layers. 

The high-level steps needed to generate the folding factors are as follows:
- Run an arbitrary `DataFlowBuild` on a network with fold 1, i.e. all folding set to 1, where possible. Make sure at least one of the steps creates an intermediate representation of the network where the layers have been turned into `ConvolutionInputGenerator` and `StreamingFCLayer` nodes.
- In the **main body of algorithm** cell, update pruning-ratio, target device LUTs, LUT downscaling ratio.
- Add path to the intermediate representation of the network with the two types of nodes in the main body cell.
- Run through the next couple of cells, primitive logging is done during optimization for debugging.
- Lastly, `plot_cycles_remainder()` allows for visual inspection of network layout, `get_res()` returns the folding factorsr.

Tip: to look at internals of folding configurations being considered for a node in conjuction with the logging, one can easily do: `node_folding_attrs = new_all_attrs.get("StreamingFCLayer_Batch_1").get("possible_attrs")` and then `sorted(node_folding_attrs.items(), key = lambda k_v: (k_v[1].get('Cycles'), k_v[0][0]), reverse = True)` to show the possible folding factors being run through.

## Pruning
To prune a network run one of these scripts (examples):
- pruning_2bit.py for pruning of 2-bit network, example:
```
BREVITAS_JIT=1 python3 pruning_l1.py --max-sparsity 0.9 --simd-list "9, 16, 16, 12, 8, 6"
```
- pruning_4bit.py for pruning of 4-bit network, example:
```
BREVITAS_JIT=1 python3 pruning_4bit.py --max-sparsity 0.9 --simd-list "9, 8, 6, 9, 3, 4" --model model/cnv_q4_20210325_112610/checkpoints/model_best.pth.tar
``` 
- pruning_8bit.py for pruning of 8-bit network, example:
```
BREVITAS_JIT=1 python3 pruning_8bit.py --max-sparsity 0.9 --simd-list "9, 8, 6, 9, 3, 4" --model model/cnv_q8_20210325_112253/checkpoints/model_best.pth.tar
```
___
To see the specific options, run ```python3 pruning_2bit.py -h```
