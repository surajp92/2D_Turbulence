#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:30:01 2021

@author: suraj
"""
import yaml

nx = 1024
re = 8000

filename = f'input_{nx}_{re}.yaml'

with open(filename) as file:
    input_data = yaml.load(file, Loader=yaml.FullLoader)
file.close() 

for i in range(1,11):
    seedn = int(i*10)
    input_data['seedn'] = seedn
    
    filename = f'input_{nx}_{re}_{i:03d}.yaml'
    with open(filename, 'w') as outfile:
        yaml.dump(input_data, outfile, default_flow_style=False)