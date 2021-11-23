# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 13:42:59 2021

@author: prajy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from scipy.ndimage import  rotate
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import DictionaryLearning


df = pd.read_csv('data.csv') #Give Input Here
Z= df.values             #Converting dataset to Matrix for easy and better manipulation
Z=Z.reshape(10000,4,4)
Z= shuffle(Z[:10000])    #Shuffle Data Randomly
OZ=np.zeros((10000,4,4)) #Storage for Original array
OZ=Z #Original Dataset Stored future

#Ploting Some Original Data Images Sample
fig, ax = plt.subplots(5,5, figsize = (10,10),sharex=True,sharey=True)
fig.suptitle('Original Images', fontsize=16)
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(Z[i],cmap="Greys")



Siz=Z[0:199] #For extracting size size track
N_Z=np.zeros((10000,4,4)) #For storing noissy image

#10 Different Noise Models are Generated and added randomly.

#Noise_1
Mean=0;
var=0.1
noise1=np.random.normal(Mean,var,Siz.shape)
Z[0:199]=Z[0:199]+noise1
Z[2000:2199]=Z[2000:2199]+noise1
Z[4200:4399]=Z[4200:4399]+noise1
Z[4800:4999]=Z[4800:4999]+noise1
Z[7800:7999]=Z[7800:7999]+noise1

#Noise_2
Mean=0;
var=0.2
noise2=np.random.normal(Mean,var,Siz.shape)
Z[200:399]=Z[200:399]+noise2
Z[2200:2399]=Z[2200:2399]+noise2
Z[5800:5999]=Z[5800:5999]+noise2
Z[6400:6599]=Z[6400:6599]+noise2
Z[8000:8199]=Z[8000:8199]+noise2

#Noise_3
Mean=0;
var=0.08
noise3=np.random.normal(Mean,var,Siz.shape)
Z[400:599]=Z[400:599]+noise3
Z[2400:2599]=Z[2400:2599]+noise3
Z[4600:4799]=Z[4600:4799]+noise3
Z[8200:8399]=Z[8200:8399]+noise3
Z[9200:9399]=Z[9200:9399]+noise3

#Noise_4
Mean=0;
var=0.3
noise4=np.random.normal(Mean,var,Siz.shape)
Z[600:799]=Z[600:799]+noise4
Z[2600:2799]=Z[2600:2799]+noise4
Z[6000:6199]=Z[6000:6199]+noise4
Z[8400:8599]=Z[8400:8599]+noise4
Z[9800:9999]=Z[9800:9999]+noise4

#Noise_5
Mean=0;
var=0.06
noise5=np.random.normal(Mean,var,Siz.shape)
Z[800:999]=Z[800:999]+noise5
Z[2800:2999]=Z[2800:2999]+noise5
Z[5400:5599]=Z[5400:5599]+noise5
Z[6600:6799]=Z[6600:6799]+noise5
Z[7600:7799]=Z[7600:7799]+noise5
Z[9400:9599]=Z[9400:9599]+noise5

#Noise_6
Mean=1
var=0.5
noise6=np.random.normal(Mean,var,Siz.shape)
Z[1000:1199]=Z[1000:1199]+noise6
Z[3000:3199]=Z[3000:3199]+noise6
Z[6200:6399]=Z[6200:6399]+noise6
Z[7400:7599]=Z[7400:7599]+noise6
Z[9600:9799]=Z[9600:9799]+noise6

#Noise_7
Mean=2
var=0.4
noise7=np.random.normal(Mean,var,Siz.shape)
Z[1200:1399]=Z[1200:1399]+noise7
Z[3200:3399]=Z[3200:3399]+noise7
Z[5600:5799]=Z[5600:5799]+noise7
Z[7000:7199]=Z[7000:7199]+noise7
Z[9000:9199]=Z[9000:9199]+noise7


#Noise_8
Mean=2
var=0.05
noise8=np.random.normal(Mean,var,Siz.shape)
Z[1400:1599]=Z[1400:1599]+noise8
Z[3400:3599]=Z[3400:3599]+noise8
Z[5200:5399]=Z[5200:5399]+noise8
Z[6800:6999]=Z[6800:6999]+noise8
Z[800:999]=Z[800:999]+noise8

#Noise_9
Mean=2
var=0.18
noise9=np.random.normal(Mean,var,Siz.shape)
Z[1600:1799]=Z[1600:1799]+noise9
Z[3600:3799]=Z[3600:3799]+noise9
Z[4000:4199]=Z[4000:4199]+noise9
Z[5000:5199]=Z[5000:5199]+noise9
Z[8600:8799]=Z[8600:8799]+noise9

#Noise_10
Mean=2
var=0.04
noise10=np.random.normal(Mean,var,Siz.shape)
Z[1800:1999]=Z[1800:1999]+noise10
Z[3800:3999]=Z[3800:3999]+noise10
Z[4400:4599]=Z[4400:4599]+noise10
Z[7200:7399]=Z[7200:7399]+noise10
Z[8800:8999]=Z[8800:8999]+noise10



#Plotting The Noisy Images
N_Z=Z
fig, ax = plt.subplots(5,5, figsize = (10,10),sharex=True,sharey=True)
fig.suptitle('Noisy Images', fontsize=16)
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(N_Z[i],cmap="Greys")

# Rotation
angle = 90
for i in range(0,199):
    N_Z[i]= rotate(N_Z[i], angle)
    
angle = 180
for i in range(200,399):
    N_Z[i]= rotate(N_Z[i], angle)

angle = 270
for i in range(400,599):
    N_Z[i]= rotate(N_Z[i], angle)

angle = 360
for i in range(600,799):
    N_Z[i]= rotate(N_Z[i], angle)

angle = 270
for i in range(800,999):
    N_Z[i]= rotate(N_Z[i], angle)

#Plotting Noisy with Rotation samples
fig, ax = plt.subplots(5,5, figsize = (10,10),sharex=True,sharey=True)
fig.suptitle('Noisy Images with Rotation', fontsize=16)
plt.tight_layout()
axes = ax.flatten()
for i in range(25):
    axes[i].imshow(N_Z[i],cmap="Greys")


A=N_Z.reshape(10000,16)


#####################################Applying PCA##############################
pca=PCA(16)
Fit_try=pca.fit(A)
g=pca.components_

#Plotting PCA Component
Final=g
Final=Final.reshape(16,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10),sharex=True,sharey=True)
fig.suptitle('PCA Output', fontsize=16)
plt.tight_layout()
axes = ax.flatten()
for i in range(16):
    axes[i].imshow(Final[i],cmap="Greys")
    
    
#####################################Applying NMF##############################    
nmf = NMF(16)
nmf.fit(np.abs(A))
Comp=nmf.components_

#Plotting NMF Component
B=Comp.reshape(16,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10),sharex=True,sharey=True)
fig.suptitle('NMF Output', fontsize=16)
plt.tight_layout()
axes = ax.flatten()
for i in range(16):
    axes[i].imshow(B[i],cmap="Greys")



#################################Applying Dictionary Learning##################    
dict_learner = DictionaryLearning(
     n_components=16, transform_algorithm='lasso_lars', random_state=42,
 )
dict_learner.fit(A)
D=dict_learner.components_

#Plotting Dictionary Learning Component
DL=D.reshape(16,4,4)
fig, ax = plt.subplots(5,5, figsize = (10,10),sharex=True,sharey=True)
fig.suptitle('Dictionary Output', fontsize=16)
plt.tight_layout()
axes = ax.flatten()
for i in range(16):
    axes[i].imshow(DL[i],cmap="Greys")


