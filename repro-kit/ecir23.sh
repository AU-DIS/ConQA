#!/bin/bash

python experiment.py -z ecir23 -s clip -m ViT-L/14 -e ecir23 -r hit_rate@1,hit_rate@5,hit_rate@10 --header
python experiment.py -z ecir23 -s stclip -m ViT-L/14 -e ecir23 -r hit_rate@1,hit_rate@5,hit_rate@10 
