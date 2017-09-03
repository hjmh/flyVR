""" Script for analysing FlyOver trials of a single animal (assumes use of FlyOver version 9.2 or higher) """

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
from os.path import sep
from sys import path
from glob import glob

import matplotlib.pyplot as plt

from Tkinter import Tk
from tkFileDialog import askdirectory, askopenfilename

from flyVRoptogenetics_oneTrialBasicAnalysis import singleVROptogenTrialAnalysis

# ---------------------------------------------------------------------------------------------
# Choose files to analyse
# ---------------------------------------------------------------------------------------------

baseDir = '/Volumes/jayaramanlab/Hannah/Projects/FlyVR/1_Experiments/'

root = Tk()
# prevents root window from appearing
root.withdraw()
# choose experiment folder
expDir = askdirectory(initialdir=baseDir, title='Select experiment directory of a single fly') + sep
dataDir = sep.join(expDir.split(sep)[0:-2]) + sep
flyID = expDir.split(sep)[-2]

FODataFiles = [filepath.split(sep)[-1] for filepath in glob(expDir + '*.txt')]
FODataFiles = sorted(FODataFiles)

print('\n Analysing the following log files:\n')
print(FODataFiles)

for fileToAnalyse in FODataFiles:
    # ------------------------------------------------------------------------------------------
    # Run for each file in folder
    # ------------------------------------------------------------------------------------------
    returnVal = singleVROptogenTrialAnalysis(expDir + fileToAnalyse)

    FODatLoad = np.load(expDir + fileToAnalyse[:-3] + 'npy')[()]


