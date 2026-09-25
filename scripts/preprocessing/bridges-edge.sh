#!/bin/bash
torchrun --standalone --nproc_per_node=4 preprocess.py --root ./-edge --transforms edge lpe --task bridges --max_eigvals 96