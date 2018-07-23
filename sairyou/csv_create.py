# -*- coding: utf-8 -*-

import numpy as np

#中身は後で変えるので適当な値が入ったものを作る
data=np.empty((20,20), dtype=int)

count=0
for i in range(len(data)):
    for j in range(len(data[0])):
        random=np.random.rand()
        if random<=0.3:
            data[i][j]=-1
        elif random>0.3 and random<0.95:
            data[i][j]=0
        else:
            data[i][j]=3

#print(data)

#csvで保存
np.savetxt('resources/sample.csv', data, delimiter=',')
