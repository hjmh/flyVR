""" Script for analysing all VR trials in one experiment folder (assumes use of FlyOver version 9.2 or higher) """

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
import pandas as pd

from os import listdir, mkdir, walk
from os.path import isfile, sep
from sys import path, exit
from glob import glob

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
import seaborn as sns

from Tkinter import Tk
from tkFileDialog import askdirectory, askopenfilename

# Set path to analysis code directory
codeDir = sep.join(os.getcwd().split(sep)[:-2])
path.insert(1, codeDir)

from flyVR.flyVRAnalysis.flyVR_oneTrialBasicAnalysis import singleVRTrialAnalysis

from plottingUtilities.basicPlotting import makeNestedPlotDirectory, myAxisTheme
from plottingUtilities.objectInteractionPlots import residencyWithHistograms_splitOnWalking, curvatureVsHeading_DistanceBoxplot, plotResidencyInMiniarena

# Choose files to analyse ..............................................................................................

baseDir = '/Volumes/jayaramanlab/Hannah/Projects/FlyVR/1_Experiments/'

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

rerunbasicanalysis = raw_input('\nWould you like to (re-)run the basic analysis on each file instead of'
                               'loading previously saved data? (y, n)\n')

# generate fly color map
numFlies = len(expDirs)
cNorm = colors.Normalize(vmin=0, vmax=numFlies)
flyCMap = plt.cm.ScalarMappable(norm=cNorm,cmap='Accent')


# Collect data for residency heatmap
nTrialsVI = 0
nTrialsIN = 0

# Set moving threshold
vTransTH = 3


# collect data from all flies in list of dicitonaries
keylist = ['curvature', 'gamma', 'gammaFull', 'headingAngle', 'objectDistance', 'rotVelo', 'time',
           'transVelo', 'xPos', 'xPosInMiniarena', 'yPos', 'yPosInMiniarena']

keylistLong = ['flyID', 'trial', 'trialtype', 'moving'] + keylist
FOAllFlies_df = pd.DataFrame(data=np.zeros((0, len(keylistLong))), columns=keylistLong)


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

        # Load single trial data from file
        if rerunbasicanalysis == 'y':
            returnVal = singleVRTrialAnalysis(expDir + fileToAnalyse)
            print('Ran analysis, now loading saved *.npy file.')
            FODatLoad = np.load(expDir + fileToAnalyse[:-3] + 'npy')[()]
        else:
            print('Loading saved *.npy file.')
            FODatLoad = np.load(expDir + fileToAnalyse[:-3] + 'npy')[()]

        # Split into visible object trials and invisible object trials
        if 'Invisible' in fileToAnalyse or 'invisible' in fileToAnalyse:
            trialtype = 'invisible'
            nTrialsIN += 1
        else:
            trialtype = 'visible'
            nTrialsVI += 1

        vTrans = np.copy(FODatLoad['transVelo'])
        moving = vTrans >= vTransTH

        FODatLoad_df = pd.DataFrame.from_dict(FODatLoad,orient='columns')
        FODatLoad_df.insert(0, 'flyID', flyID)
        FODatLoad_df.insert(1, 'trial', trial)
        FODatLoad_df.insert(2, 'trialtype', trialtype)
        FODatLoad_df.insert(3, 'moving', moving)

        FOAllFlies_df = FOAllFlies_df.append(FODatLoad_df)

        # TODO: Add selection criteria, e.g. to only include trials from flies that were
        # walking >40% of time and did not have a strong turn bias.

FOAllFlies_df =FOAllFlies_df[keylistLong]

FOAllFlies_df['flyID'] = FOAllFlies_df['flyID'].astype('category')
FOAllFlies_df['trial'] = FOAllFlies_df['trial'].astype('category')
FOAllFlies_df['trialtype'] = FOAllFlies_df['trialtype'].astype('category')

# make fly summary plot of residency:
arenaRad = 60
numBins = 60

keyind_xPos = keylistLong.index('xPosInMiniarena')
keyind_yPos = keylistLong.index('yPosInMiniarena')
keyind_mov = keylistLong.index('moving')

nTrials = (nTrialsVI,nTrialsIN)

for indcond, condition in enumerate(['visible', 'invisible']):
    querystring = '(trialtype == "' + condition + '")'
    xPosMAall = np.asarray(FOAllFlies_df.query(querystring).iloc[:,keyind_xPos:keyind_xPos+1]).squeeze()
    yPosMAall = np.asarray(FOAllFlies_df.query(querystring).iloc[:,keyind_yPos:keyind_yPos+1]).squeeze()
    movingall = np.asarray(FOAllFlies_df.query(querystring).iloc[:,keyind_mov:keyind_mov+1]).squeeze().astype('bool')

    # make plot
    noNans = ~np.isnan(xPosMAall)
    hexplotfig = residencyWithHistograms_splitOnWalking(
        xPosMAall[noNans], yPosMAall[noNans], movingall[noNans], arenaRad, numBins, vTransTH,
        'Trials with ' + condition + ' objects (n=' + str(nTrials[indcond]) + ')')

    makeNestedPlotDirectory(sep.join(expDir.split(sep)[0:-3]) + sep + 'analysis/', 'heatmap/', condition+sep)
    hexplotfig.savefig(sep.join(expDir.split(sep)[0:-3])+sep+'analysis'+sep+'heatmap'+sep+condition+sep+
                       'allFlies_residencyInMiniarena.pdf', format='pdf')

