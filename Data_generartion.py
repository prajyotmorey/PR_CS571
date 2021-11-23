# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 23:44:52 2021

@author: prajy
"""
import numpy as np


Z=np.zeros((10000,4,4))
Z[0:199]=np.array([[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1]])   ##A
Z[200:399]=np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[0,1,1,1]]) ##C
Z[400:599]=np.array([[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0]]) ##D
Z[600:799]=np.array([[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,0]]) ##F
Z[800:999]=np.array([[0,1,1,0],[1,0,0,0],[1,0,1,1],[0,1,1,0]]) ##G
Z[1000:1199]=np.array([[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1]]) ##H
Z[1200:1399]=np.array([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]]) ##I
Z[1400:1599]=np.array([[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,1,1,1]]) ##J
Z[1600:1799]=np.array([[1,0,0,1],[1,0,1,0],[1,1,1,0],[1,0,0,1]]) ##K
Z[1800:1999]=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]) ##L
Z[2000:2199]=np.array([[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1]]) ##N
Z[2200:2399]=np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]]) ##O
Z[2400:2599]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,0]]) ##P
Z[2600:2799]=np.array([[1,1,1,0],[1,0,1,0],[1,1,1,0],[0,0,1,1]]) ##Q
Z[2800:2999]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,0],[1,0,0,1]]) ##R
Z[3000:3199]=np.array([[1,1,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]) ##T
Z[3200:3399]=np.array([[0,1,1,0],[0,1,0,0],[0,0,1,0],[0,1,1,0]]) ##S
Z[3400:3599]=np.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]]) ##U
Z[3600:3799]=np.array([[0,1,0,1],[0,1,0,1],[0,1,0,1],[0,0,1,0]]) ##V
Z[3800:3999]=np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]]) ##X
Z[4000:4199]=np.array([[0,1,0,1],[0,1,0,1],[0,0,1,0],[0,0,1,0]]) ##Y
Z[4200:4399]=np.array([[1,1,1,0],[0,0,1,0],[0,1,0,0],[0,1,1,1]]) ##Z
############Repeating Alphabets################
Z[4400:4599]=np.array([[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1]]) ##A
Z[4600:4799]=np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]]) ##X
Z[4800:4999]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,0]]) ##P
Z[5000:5199]=np.array([[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1]]) ##H
Z[5200:5399]=np.array([[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1]]) ##N
Z[5400:5599]=np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]]) ##O
Z[5600:5799]=np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[0,1,1,1]]) ##C
Z[5800:5999]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,0],[1,0,0,1]]) ##R
Z[6000:6199]=np.array([[1,1,1,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]) ##T
Z[6200:6399]=np.array([[1,1,1,0],[1,0,1,0],[1,1,1,0],[0,0,1,1]]) ##Q
Z[6400:6599]=np.array([[0,1,1,0],[1,0,0,0],[1,0,1,1],[0,1,1,0]]) ##G
Z[6600:6799]=np.array([[0,1,1,0],[0,1,0,0],[0,0,1,0],[0,1,1,0]]) ##S
Z[6800:6999]=np.array([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]]) ##I
Z[7000:7199]=np.array([[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0]]) ##D
Z[7200:7399]=np.array([[1,0,0,1],[1,0,1,0],[1,1,1,0],[1,0,0,1]]) ##K
Z[7400:7599]=np.array([[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,1,1,1]]) ##J
Z[7600:7799]=np.array([[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]]) ##U
Z[7800:7999]=np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]]) ##O
Z[8000:8199]=np.array([[1,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,0]]) ##F
Z[8200:8399]=np.array([[1,1,1,0],[0,0,1,0],[0,1,0,0],[0,1,1,1]]) ##Z
Z[8400:8599]=np.array([[0,1,0,1],[0,1,0,1],[0,0,1,0],[0,0,1,0]]) ##Y
Z[8600:8799]=np.array([[0,1,1,0],[1,0,0,0],[1,0,1,1],[0,1,1,0]]) ##G
Z[8800:8999]=np.array([[0,1,0,1],[0,1,0,1],[0,1,0,1],[0,0,1,0]]) ##V
Z[9000:9199]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1],[1,0,0,0]]) ##P
Z[9200:9399]=np.array([[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,1,1,1]]) ##J
Z[9400:9599]=np.array([[1,1,1,0],[0,0,1,0],[0,1,0,0],[0,1,1,1]]) ##Z
Z[9600:9799]=np.array([[1,1,1,1],[1,0,0,1],[1,1,1,0],[1,0,0,1]]) ##R
Z[9800:9999]=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]) ##L

Z=Z.reshape(10000,16)

np.savetxt("data.csv", Z, delimiter=",")
