#!/bin/bash
python preprocess.py --root ./ --transforms token_index rwse lpe --task pcq --max_rw_steps 32 --max_eigvals 96