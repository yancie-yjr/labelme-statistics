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
import matplotlib.pyplot as plt
import pylab
import pandas as pd



# # 标注文件的路径
###统计所以框的个数
# annotated_dir = r"G:\carton_dataset_v3\杨金荣v3\优选数据"
# counter = collections.Counter()
# for dirpath, dirnames, filenames in os.walk(annotated_dir):
#     for filename in filenames:
#         if osp.splitext(filename)[-1] != '.json':
#             continue
#         filename = osp.join(dirpath, filename)
#         print(filename)
#
#         with open(filename) as f:
#             data = json.load(f)
#         for shape in data['shapes']:
#             counter[shape['label']] += 1
#
# print('---')
#
# # 统计label的总数
# total = 0
#
# print('# of Labels')
# for label, count in counter.items():
#     total = total + count
#     print('{:>10}: {}'.format(label, count))
#
# print('{}: {}'.format("总共", total))


# 标注文件的路径
# 统计每张照片的instance个数
annotated_dir = r"G:\carton_dataset_v3\杨金荣v3\优选数据"
counter = collections.Counter()
instance_list=[]   ###
for dirpath, dirnames, filenames in os.walk(annotated_dir):
    for filename in filenames:
        if osp.splitext(filename)[-1] != '.json':
            continue
        filename = osp.join(dirpath, filename)
        # print(filename)

        with open(filename) as f:
            data = json.load(f)
        instance_list.append(len(data['shapes']))
        print(len(instance_list),filename,'is',instance_list[-1])

print('---------')
print('total image is: ', len(instance_list))
# print(instance_list)

plt.hist(instance_list,bins=100)
plt.ylabel('instances')

plt.show()


