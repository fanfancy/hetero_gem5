import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum

# Parameter

# architecture parameter
Chiplets = 16
PEs = 16
C0 = 4
K0 = 2

# network parameter
P = 100
Q = 200
C = 64
K = 128
R = 3
S = 3
C_new = math.ceil(C/C0)
K_new = math.ceil(K/K0)
size = [P,Q,C_new,K_new]

# parallel type
# 0: C-P ; 1: P-P ; 2: PQ-P ; 3: KC-P ; 4: all-type
Par_type = 4

# partition list
Pset = [] #[P1,P2,P3]
Qset = [] #[Q1,Q2,Q3]
Cset = [] #[C1,C2,C3,Cp1,Cp2]
Kset = [] #[K1,K2,K3,Cp1,Cp2]
parallel_dim_set = []
parallel_num_set = []

# encode parameter
dim_list = ['P','Q','C','K','R','S']
type_list = ["for","parallel-for-1","parallel-for-2"]
parallel_select = [0,1,3]

# fitness parameter
O_correlation = [1,1,0,1,0,0]
A_correlation = [1,1,1,0,0,0]
W_correlation = [0,0,1,1,1,1]

# PE Distribution
set_1_e_16 = {}
set_2_e_8 = {}
set_8_e_2 = {}
set_16_e_1 = {}
set_4_e_4 = {}
set_1_e_16[0] = {0:[1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19]}
set_2_e_8[0] = {0:[1,2,3,4,11,12,13,14],1:[6,7,8,9,16,17,18,19]}
set_2_e_8[1] = {0:[1,2,3,4,6,7,8,9],1:[11,12,13,14,16,17,18,19]}
set_8_e_2[0] = {0:[1,6],1:[2,7],2:[3,8],3:[4,9],4:[11,16],5:[12,17],6:[13,18],7:[14,19]}
set_8_e_2[1] = {0:[1,11],1:[2,12],2:[3,13],3:[4,14],4:[6,16],5:[7,17],6:[8,18],7:[9,19]}
set_16_e_1[0] = {0:[1],1:[2],2:[3],3:[4],4:[6],5:[7],6:[8],7:[9],8:[11],9:[12],10:[13],11:[14],12:[16],13:[17],14:[18],15:[19]}
set_4_e_4[0] = {0:[1,2,3,4],1:[6,7,8,9],2:[11,12,13,14],3:[16,17,18,19]}
set_4_e_4[1] = {0:[1,6,11,16],1:[2,7,12,17],2:[3,8,13,18],3:[4,9,14,19]}
