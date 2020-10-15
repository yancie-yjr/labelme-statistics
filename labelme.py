# coding: UTF-8
"""
Created on Thu Jun  4 15:58:21 2020

@author: PAN Fei
"""

import collections
import json
import os
import os.path as osp
import pprint


# 标注文件的路径
annotated_dir = r"G:\carton_dataset_v3\吴胜凯v3"
counter = collections.Counter()
for dirpath, dirnames, filenames in os.walk(annotated_dir):
    for filename in filenames:
        if osp.splitext(filename)[-1] != '.json':
            continue
        filename = osp.join(dirpath, filename)
        print(filename)

        with open(filename) as f:
            data = json.load(f)
        for shape in data['shapes']:
            counter[shape['label']] += 1

print('---')

# 统计label的总数
total = 0

print('# of Labels')
for label, count in counter.items():
    total = total + count
    print('{:>10}: {}'.format(label, count))

print('{}: {}'.format("总共", total))
