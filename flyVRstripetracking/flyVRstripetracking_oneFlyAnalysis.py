import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import matplotlib.colors as colors
from os import listdir, mkdir
from os.path import isfile, sep
from glob import glob
from scipy.interpolate import interp1d

import seaborn as sns
sns.set_style('ticks')

from sys import path

from Tkinter import Tk
from tkFileDialog import askdirectory, askopenfilename

from flyVRstripetracking_oneTrialBasicAnalysis import singleVRStripeTrialAnalysis
# ----------------------------------------------------------------------------------------------------------------------
# Choose files to analyse
# ----------------------------------------------------------------------------------------------------------------------

baseDir = '/Volumes/jayaramanlab/Hannah/Projects/FlyVR/1_Experiments/singleObject/'

root = Tk()
# prevents root window from appearing
root.withdraw()
# choose experiment folder
expDir = askdirectory(initialdir=baseDir, title='Select experiment directory of a single fly') + sep
dataDir = sep.join(expDir.split(sep)[0:-2]) + sep
flyID = expDir.split(sep)[-2]

# create analysis dir
analysisDir = sep.join(dataDir.split(sep)[:-2]) + sep + 'analysis' + sep
try:
    mkdir(analysisDir)
except OSError:
    print('Analysis directory already exists.')

FODataFiles = [filepath.split(sep)[-1] for filepath in glob(expDir + '*.txt')]
FODataFiles = sorted(FODataFiles)

print('\n Analysing the following log files:\n')
print(FODataFiles)

for FODataFile in FODataFiles:
    # ------------------------------------------------------------------------------------------------------------------
    # Run for each file in folder
    # ------------------------------------------------------------------------------------------------------------------
    returnVal = singleVRStripeTrialAnalysis(expDir+FODataFile)

    # test that npy file got saved
    FODatLoad = np.load(expDir + FODataFile[:-3] + 'npy')[()]

    print('Analysis ran successfully!')

