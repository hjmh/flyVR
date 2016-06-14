""" Script for analysing all VR trials in one experiment folder (assumes use of FlyOver version 9.2 or higher) """

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
from os import walk
from os.path import sep
from sys import path, exit
from glob import glob

from Tkinter import Tk
from tkFileDialog import askdirectory, askopenfilename

from flyVRstripetracking_oneTrialBasicAnalysis import singleVRStripeTrialAnalysis

# Choose files to analyse ..............................................................................................

baseDir = '/Volumes/jayaramanlab/Hannah/Projects/FlyVR/1_Experiments/singleObject/'

root = Tk()
# prevents root window from appearing
root.withdraw()
# choose experiment folder
dataDir = askdirectory(initialdir=baseDir,
                       title='Select experiment directory (containing directories for multiple flies') + sep
expDirs = sorted(walk(dataDir).next()[1])

try:
    expDirs.remove('virtualWorld')
except:
    print('You selected an invalid data directory.\n' +
          'Expected folder structure of the selected path is some/path/to/experimentName/flyGender/rawData/')
    exit(1)

print('\n Analysing the following folders:\n')
print(expDirs)

# generate fly color map
numFlies = len(expDirs)

# Go through directories of experiments with different flies ...........................................................
for currExpDir in expDirs:

    expDir = dataDir + currExpDir + sep
    print('\n Analysing the following folder:\n')
    print(expDir)

    FODataFiles = [filepath.split(sep)[-1] for filepath in glob(expDir + '*.txt')]
    FODataFiles = sorted(FODataFiles)

    print('\n Analysing the following log files:\n')
    print(FODataFiles)

    # Run single trial analysis on each file in folder .....................................................
    for fileToAnalyse in FODataFiles:

        flyID = fileToAnalyse.split('_')[0]
        trial = FODataFiles.index(fileToAnalyse) + 1

        returnVal = singleVRStripeTrialAnalysis(expDir + fileToAnalyse)
        print('Ran analysis, now loading saved *.npy file.')
        FODatLoad = np.load(expDir + fileToAnalyse[:-3] + 'npy')[()]

print('Analysis ran successfully')