import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from matplotlib import pyplot as plt
from gaTest_noc_nop import *
from randomTest_noc_nop import *

times_compare = 20
fitness_random = []
fitness_Ga = []
for i in range(times_compare):
	print("### times iter = ", i)
	fitness_min = gaIter()
	fitness_random.append(fitness_min[len(fitness_min)-1])
	compuation_cycles_1,degrade_ratio_1, fitness_min_list = randomTest()
	fitness_Ga.append(fitness_min_list[len(fitness_min_list)-1])

index = range(times_compare)

plt.figure(1)
plt.plot(index,fitness_random, marker='x', mec='r', mfc='w',label=u'random fitness')
plt.plot(index,fitness_Ga, marker='.', mec='r', mfc='w',label=u'GA fitness')
plt.legend()
plt.xlabel(u"iter times") 
plt.ylabel("fitness")
plt.savefig("compare_Ga_ran.png")
