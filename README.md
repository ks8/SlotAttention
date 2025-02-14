# Slot Attention

This repository implements Slot Attention, as described in the paper [Object-Centric Learning with Slot Attention
](https://arxiv.org/abs/2006.15055). 

## Table of Contents
- [Installation](#installation)
- [Training Data](#training-data)
- [Training](#training)

## Installation
SlotAttention can be installed from source:
1. ```git clone https://github.com/ks8/SlotAttention.git```
2. ```cd SlotAttention```
3. ```conda env create -f environment.yml```
4. ```conda activate SlotAttention```

## Training Data
We use a portion of the Tetrominoes dataset for training, which we borrowed 
from https://github.com/adityabingi/Slot-Attention. As noted in that repository, the Tetrominoes dataset is part 
of the Google Multi-Object Datasets and is available as TFRecords [here](https://github.com/deepmind/multi_object_datasets).
We only use h5py versions of this dataset that @pemami4911 created by modifying the TFRecords. We copied a version 
of this and made it available for download [here](https://drive.google.com/file/d/1RNEdk4UI2pnGr_B3ZWqsBwBRolGRT2VJ/view?usp=sharing). 

## Training 
Download the training data file, tetrominoes.h5, into the SlotAttention directory. Then, simply run:  

```
python main.py
```

Check out the ```main.py``` file for various other CLI options, many of which are enabled by Pytorch Lightning.  

