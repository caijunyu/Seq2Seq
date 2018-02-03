# -*- coding: utf-8 -*-
import os
import random


conv_path = ''
if not os.path.exists(conv_path):
    exit()

convs = [] #用于存储对话集合
with open(conv_path) as f:
    one_conv = []  #存储一次完整对话
    for line in f:
        line = line.strip('\n').replace('/', '')   #去除换行符，并在字符间添加空格符，原因是用于区分123 与1 2 3.
        if line == '':
            continue
        elif line[0] == 'E':
            if one_conv:
                convs.append(one_conv)
            one_conv = []
        elif line[0] == 'M':
            one_conv.append(line.split(' ')[1])    #将一次完整的对话存储下来