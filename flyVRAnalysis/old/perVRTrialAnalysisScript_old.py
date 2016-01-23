__author__ = 'hannah'
""" Script for analysing a single FlyOver trial (assumes use of FlyOver version 9.2) """

import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import matplotlib.colors as colors
import seaborn as sns
sns.set_style('ticks')

from os import listdir, mkdir
from os.path import isfile
from sys import path

# Import custom plotting functions
path.insert(0, '/Users/hannah/Dropbox/code/plottingUtilities/')
from plottingUtilities import *
from velocityDistributionPlots import *

# ----------------------------------------------------------------------------------------------------------------------
# Choose files to analyse
# ----------------------------------------------------------------------------------------------------------------------

dataDir = '/Volumes/jayaramanlab/Hannah/Projects/FlyVR/1_Experiments/WTB_FlyOver092/males/'
flyID = 'WTBm03'
expDir = dataDir + 'rawData/' + flyID + '/'
FODataFiles = listdir(expDir)
FODataFiles = sorted(FODataFiles)


try:
    mkdir(dataDir + 'analysis/')
except OSError:
    print('Analysis directory already exists.')

# choose experiment
trial = 4

FODataFile = FODataFiles[trial-1]
dataFileParts = FODataFile.split('_')
titleString = 'fly ' + flyID + ' in ' + dataFileParts[2] + ' of ' + dataFileParts[3] + ', trial' + str(trial)
print('Analysing exeriment...\n')
print(titleString + '\n')

if 'Invisible' in FODataFile or 'invisible' in FODataFile:
    objecttype = 'invisible'
elif 'Empty' in FODataFile or 'empty' in FODataFile:
    objecttype = 'no'
else:
    objecttype = 'visible'


# ----------------------------------------------------------------------------------------------------------------------
# Load or read in logfile data
# ----------------------------------------------------------------------------------------------------------------------

# Parse header .........................................................................................................

with open(expDir + FODataFile) as FOf:
    lines = FOf.read().splitlines()

# Find beginning of data
firstDataLine = 0
while lines[firstDataLine][0] == '#':
    firstDataLine += 1
header = lines[firstDataLine-1][:].split()[1:]


# Get calibration parameter from header ................................................................................
currHeaderLine = 0
while lines[currHeaderLine][0] == '#':
    if (lines[currHeaderLine].split(' ')[1] == 'Treadmill') & (lines[currHeaderLine].split(' ')[2] == 'ball'):
        ballRadius_mm = lines[currHeaderLine].split(' ')[-2]
        xRotCalib_ticksPerSemicirc = lines[currHeaderLine+1].split(' ')[-2]
        yRotCalib_ticksPerSemicirc = lines[currHeaderLine+2].split(' ')[-2]
        zRotCalib_ticksPerSemicirc = lines[currHeaderLine+3].split(' ')[-2]
        # print('ball radius = ' + str(ballRadius_mm))
        break
    currHeaderLine += 1


# Parse data ...........................................................................................................

# Move text data into a np array
numFrames = len(lines[firstDataLine:-1])

# Check if data has been loaded before
if isfile(expDir + FODataFile[:-3] + 'npy'):
    np.load(expDir + FODataFile[:-3] + 'npy')
else:
    frameRange = range(0, numFrames)

    FOData = np.zeros((numFrames, 12))
    tmp = np.asarray(lines[firstDataLine:len(lines)+1])

    for line in frameRange:
        FOData[line, :] = np.asarray(tmp[line].split(',')).astype('float')


# Read in object coordinates ...........................................................................................

# Find name of coord file
currHeaderLine = 0
while lines[currHeaderLine][0] == '#':

    if lines[currHeaderLine].split(' ')[1] == 'Scene':
        coordFile = lines[currHeaderLine].split(' ')[-1].split('/')[-1]
        coordFile = coordFile[0:-3] + 'coords'
        break
    currHeaderLine += 1

# Read coord file and extract coordinates of visible objects
with open(dataDir + 'rawData/' + coordFile) as Coof:
    lines = Coof.read().splitlines()

invisibleObjectCoords = []
visibleObjectCoords = []

for line in range(len(lines)):
    currObject = lines[line].split(' ')

    if len(currObject) < 3:
        continue

    if currObject[0][0] == '_':
        invisibleObjectCoords.append(np.asarray((float(currObject[2]), float(currObject[3]))))
    elif currObject[0] == 'Origin':
        origin = [float(currObject[2]), float(currObject[3])]
    else:
        visibleObjectCoords.append(np.asarray((float(currObject[2]), float(currObject[3]))))

invisibleObjectCoords = np.asarray(invisibleObjectCoords)
visibleObjectCoords = np.asarray(visibleObjectCoords)


# Compute movement velocities ..........................................................................................

# Convert heading angle (FlyOver) to rad
angle = np.zeros(numFrames)
angle[:] = np.pi/180*FOData[:, 5]
# angle[np.pi/180*FOData[:,5]>np.pi] = angle[np.pi/180*FOData[:,5] > np.pi] - np.pi
# angle[np.pi/180*FOData[:,5]<np.pi] = angle[np.pi/180*FOData[:,5] < np.pi] + np.pi

# Compute translational and rotational velocity
logTime = np.copy(FOData[:, 0])
time = np.linspace(0, logTime[-1], numFrames)

vTrans = np.zeros(numFrames)
vTrans[0:-1] = np.sqrt(np.square(np.diff(FOData[:, 1])), np.square(np.diff(FOData[:, 2]))) / np.diff(time)
vTrans[np.where(np.isnan(vTrans))[0]] = 0

vRot = np.zeros(numFrames)
vRot[0:-1] = np.diff(angle)
vRot[vRot > np.pi] -= 2*np.pi
vRot[vRot <= -np.pi] += 2*np.pi
vRot[0:-1] = vRot[0:-1] / np.diff(time)
vRot[np.where(np.isnan(vRot))[0]] = 0

# Filter translational and rotational velocities
N = 60
vTransFilt = np.convolve(vTrans, np.ones((N,))/N, mode='same')
vRotFilt = np.convolve(vRot, np.ones((N,))/N, mode='same')


# Down sample data to 20 Hz ............................................................................................
samplingRate = 20
time_ds = np.linspace(0, logTime[-1], 20*round(logTime[-1]))

numFrames_ds = len(time_ds)

f_xPos = interp1d(time, FOData[:, 1], kind='linear')
f_yPos = interp1d(time, FOData[:, 2], kind='linear')

f_angle = interp1d(time, angle, kind='linear')

xPos_ds = f_xPos(time_ds)
yPos_ds = f_yPos(time_ds)

angle_ds = f_angle(time_ds)

# Compute down sampled translational and rotational velocity ...........................................................
vTrans_ds = np.zeros(numFrames_ds)
vRot_ds = np.zeros(numFrames_ds)

vTrans_ds[0:-1] = np.hypot(np.diff(xPos_ds), np.diff(yPos_ds)) / np.diff(time_ds)

vRot_ds[0:-1] = np.diff(angle_ds)
vRot_ds[vRot_ds > np.pi] -= 2*np.pi
vRot_ds[vRot_ds < -np.pi] += 2*np.pi
vRot_ds[0:-1] = vRot_ds[0:-1] / np.diff(time_ds)

vTrans_ds[np.where(np.isnan(vTrans_ds))[0]] = 0
vRot_ds[np.where(np.isnan(vRot_ds))[0]] = 0

# Filter down sampled translational and rotational velocities
N = 5
vTransFilt_ds = np.convolve(vTrans_ds, np.ones((N,))/N, mode='same')
vRotFilt_ds = np.convolve(vRot_ds, np.ones((N,))/N, mode='same')


# ----------------------------------------------------------------------------------------------------------------------
# Generate basic analysis plots
# ----------------------------------------------------------------------------------------------------------------------

# Time step plot .......................................................................................................

tstpfig = plt.figure(figsize=(10, 3))
gs = gridspec.GridSpec(1, 2, width_ratios=np.hstack((2, 1)))
tstpfig.suptitle(titleString, fontsize=14)
histRange = (0, 0.011)

ax = tstpfig.add_subplot(gs[0])
ax.plot(FOData[0:-2, 0], (FOData[1:-1, 0]-FOData[0:-2, 0]).astype('float'), '.', alpha=0.1)
ax.set_ylim(histRange)
ax.set_xlim((0, time[-1]))
ax.set_xlabel('time [s]')
ax.set_ylabel('time step [1/s]')
myAxisTheme(ax)

ax = tstpfig.add_subplot(gs[1])
ax.hist(FOData[1:-1, 0]-FOData[0:-2, 0], 50, histRange)
ax.set_xlabel('time step [1/s]')
ax.set_ylabel('count')
ax.set_title('mean time step = ' + str(round(np.mean((FOData[1:-1, 0]-FOData[0:-2, 0])*1000.0), 2)) + 'ms')
myAxisTheme(ax)

tstpfig.tight_layout()

try:
    mkdir(analysisDir + 'timeStepPlot/')
except OSError:
    print('Analysis directory already exists.')
try:
    mkdir(analysisDir + 'timeStepPlot/' + objecttype + '/')
except OSError:
    print('Plot directory already exists.')

tstpfig.savefig(analysisDir + 'timeStepPlot/' + objecttype + '/'
                + FODataFile[0:-4] + '_timeStepPlot_trial' + str(trial) + '.pdf', format='pdf')


# Plot of walking trace (+ colorbar for time) with object locations ....................................................
tStart = 0
tEnd = len(FOData[:, 1])
tStep = 72
frameRange = range(tStart, tEnd, tStep)
colMap = 'Accent'
arrowLength = 5

trajfig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=np.hstack((10, 1)))

axTraj = trajfig.add_subplot(gs[0])
axTime = trajfig.add_subplot(gs[1])

plotPosInRange(axTraj, axTime, frameRange, FOData[:, 0], FOData[:, 1], FOData[:, 2], np.pi/180*FOData[:, 5],
               colMap, arrowLength, 0.5, 5)
axTraj.scatter(visibleObjectCoords[:, 0], visibleObjectCoords[:, 1], 50, alpha=0.5,
               facecolor='black', edgecolors='none')
axTraj.scatter(invisibleObjectCoords[:, 0], invisibleObjectCoords[:, 1], 50, alpha=0.5,
               facecolors='none', edgecolors='black')
axTraj.set_xlabel(header[1], fontsize=12)
axTraj.set_ylabel(header[2], fontsize=12)
axTraj.set_title('Walking trace of ' + titleString, fontsize=14)
axTraj.set_xlim([max(-400, min(FOData[:, 1]) - 20), min(400, max(FOData[:, 1]) + 20)])
axTraj.set_ylim([max(-400, min(FOData[:, 2]) - 20), min(400, max(FOData[:, 2]) + 20)])
myAxisTheme(axTraj)

axTime.set_xlabel(header[0], fontsize=12)
plt.xlim((0, FOData[-1, 0]))
timeAxisTheme(axTime)

try:
    mkdir(analysisDir + 'tracePlot/')
except OSError:
    print('Analysis directory already exists.')
try:
    mkdir(analysisDir + 'tracePlot/' + objecttype + '/')
except OSError:
    print('Plot directory already exists.')
trajfig.savefig(analysisDir + 'tracePlot/' + objecttype + '/'
                + FODataFile[0:-4] + '_traceObjectPlot_trial' + str(trial) + '.pdf', format='pdf')


# Velocity distributions ...............................................................................................
# rotLim = (-15, 15)
# transLim = (0, 50)
# angleLim = (-np.pi, np.pi)
# summaryVeloFig = velocitySummaryPlot(time, vTrans, vTransFilt, vRot, vRotFilt, angle, rotLim, transLim, angleLim,
#                                      'Raw and filtered velocity traces, ' + titleString)
# try:
#     mkdir(analysisDir + 'velocityTraces/')
# except OSError:
#     print('Plot directory already exists.')
# summaryVeloFig.savefig(analysisDir + '/velocityTraces/' + FODataFile[0:-4] + '_veloTraces.pdf', format='pdf')


# Plot velocity distributions of downs sampled data ....................................................................
rotLim = (-5, 5)
transLim = (0, 30)
angleLim = (-np.pi, np.pi)
summaryVeloFig_ds = velocitySummaryPlot(time_ds, vTrans_ds, vTransFilt_ds, vRot_ds, vRotFilt_ds, angle_ds, rotLim,
                                        transLim, angleLim, 'Down sampled velocity traces, ' + titleString)
try:
    mkdir(analysisDir + 'velocityTraces/')
except OSError:
    print('Analysis directory already exists.')
try:
    mkdir(analysisDir + 'velocityTraces/' + objecttype + '/')
except OSError:
    print('Plot directory already exists.')

summaryVeloFig_ds.savefig(analysisDir + 'velocityTraces/' + objecttype + '/'
                          + FODataFile[0:-4] + '_veloTraces_ds_trial' + str(trial) + '.pdf', format='pdf')


# ----------------------------------------------------------------------------------------------------------------------
#  Collapse traces to single object cell and plot resulting trace
# ----------------------------------------------------------------------------------------------------------------------

path.insert(0, '/Users/hannah/Dropbox/code/trajectoryAnalysis/')
from periodicWorldAnalysis import *

# Collapse to 'mini-arena' while preserving the global heading
arenaRad = 60
if objecttype == 'visible':
    objectCoords = np.copy(visibleObjectCoords[0:-3, 0:2])
else:  # use of non-physics, invisible objects allows us to mark virtual object positions in empty arena
    objectCoords = np.copy(invisibleObjectCoords[:, 0:2])

xPosMA, yPosMA = collapseToMiniArena(FOData[:, 1], FOData[:, 2], arenaRad, objectCoords)

# Compute donw sampled collapsed traces
f_xPosMA = interp1d(time, xPosMA, kind='linear')
f_yPosMA = interp1d(time, yPosMA, kind='linear')

xPosMA_ds = f_xPosMA(time_ds)
yPosMA_ds = f_yPosMA(time_ds)

# Plot collapsed trace together with original
MATrace = plt.figure(figsize=(5, 5))

projPlot = MATrace.add_subplot(111)
projPlot.plot(xPosMA, yPosMA, '.', alpha=0.02)
projPlot.plot(FOData[:, 1], FOData[:, 2], 'k.', alpha=0.01)
projPlot.scatter(visibleObjectCoords[:, 0], visibleObjectCoords[:, 1], 50, alpha=0.5,
                 facecolor='black', edgecolors='none')
projPlot.scatter(invisibleObjectCoords[:, 0], invisibleObjectCoords[:, 1], 50, alpha=0.5,
                 facecolors='none', edgecolors='black')
projPlot.set_xlim([min(min(FOData[:, 1])-10, -80), max(max(FOData[:, 1])+10, 65)])
projPlot.set_ylim([min(min(FOData[:, 2])-10, -80), max(max(FOData[:, 2])+10, 65)])
projPlot.set_aspect('equal')
myAxisTheme(projPlot)


# Plot collapsed, down sampled trace ...................................................................................
tStart = 0
tEnd = numFrames_ds
tStep = 4
frameRange = range(tStart, tEnd, tStep)
colMap = 'Accent'
arrowLength = 5

colTrFig = plt.figure(figsize=(9, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=np.hstack((10, 1)))

colTrFig.suptitle('Collapsed walking trace ("mini arena" with central object)\n' + titleString, fontsize=14)

axTraj = colTrFig.add_subplot(gs[0])
axTime = colTrFig.add_subplot(gs[1])
plotPosInRange(axTraj, axTime, frameRange, time_ds, xPosMA_ds, yPosMA_ds, angle_ds, colMap, 4, 0.5, 7)
axTraj.plot(0, 0, marker='o', markersize=20, linestyle='none', alpha=0.5, color='black')
axTraj.set_xlabel(header[1], fontsize=12)
axTraj.set_ylabel(header[2], fontsize=12)
axTraj.set_ylim([-(arenaRad+5), arenaRad + 5])
axTraj.set_xlim([-(arenaRad+5), arenaRad + 5])
myAxisTheme(axTraj)
axTime.set_xlabel(header[0], fontsize=12)
plt.xlim((0, time_ds[-1]))
timeAxisTheme(axTime)

try:
    mkdir(analysisDir + 'collapsedTracePlot/')
except OSError:
    print('Analysis directory already exists.')
try:
    mkdir(analysisDir + 'collapsedTracePlot/' + objecttype + '/')
except OSError:
    print('Plot directory already exists.')

colTrFig.savefig(analysisDir + 'collapsedTracePlot/' + objecttype + '/'
                 + FODataFile[0:-4] + '_traceObjectPlot_ds_trial' + str(trial) + '.pdf', format='pdf')

# ----------------------------------------------------------------------------------------------------------------------
# Compute heading angle relative to closest object (use collapsed coordinates)
# ----------------------------------------------------------------------------------------------------------------------

def dotproduct2d(a, b):
    # 2D dot product
    return a[0, :]*b[0, :] + a[1, :]*b[1, :]

def veclength2d(vec):
    return np.sqrt(vec[0, :]**2 + vec[1, :]**2)

# Vector to object location
objectDirection = np.vstack((-xPosMA_ds, -yPosMA_ds))

objDist = veclength2d(objectDirection)

# Fly orientation vector
flyDirection = np.vstack((np.cos(angle_ds), np.sin(angle_ds)))

# Angle to object relative from fly's orientation
dotProd = dotproduct2d(flyDirection, objectDirection)
lenFlyVec = np.hypot(flyDirection[0, :], flyDirection[1, :])
lenObjVec = np.hypot(objectDirection[0, :], objectDirection[1, :])

gamma = np.arccos(dotProd / (lenFlyVec * lenObjVec))

gammaFull = np.arctan2(flyDirection[1, :], flyDirection[0, :])-np.arctan2(objectDirection[1, :], objectDirection[0, :])
gammaFull[gammaFull < 0] += 2 * np.pi
gammaFull[gammaFull > np.pi] -= 2 * np.pi

# Change in heading rel. to object
gammaV = np.hstack((np.diff(gamma)/np.diff(time_ds), 0))

sf = 0
ef = len(xPosMA_ds)
near = 6
far = arenaRad
vTransTH = 2

turnTH = 3*np.std(abs(vRotFilt_ds))
turnMask = (abs(vRotFilt_ds) > turnTH)

selectedRangeDist = np.logical_and(np.logical_and(objDist > near, objDist < far), vTrans_ds > vTransTH)
selectedRangeDistTurn = np.logical_and(selectedRangeDist, turnMask)

headTurnFig = plt.figure(figsize=(17, 5))

headTurnFig.suptitle('Relationship between turns and relativel heading angle\n' + flyID + ', trial' + str(trial) +
                     ', percentage turns: ' + str(round(100.0*sum(turnMask)/len(vRotFilt_ds), 2)) + ' (within range '
                     + str(near) + '-' + str(far) + 'mm, trans. vel. > ' + str(vTransTH) + ')\n', fontsize=13)

distRange = (near, far)
angleRange = (0, np.pi)
vHeadRange = (-5, 5)

ax0 = headTurnFig.add_subplot(131)
ax0.plot(xPosMA_ds[sf:ef], yPosMA_ds[sf:ef], '.', color='grey', alpha=0.4)
ax0.plot(xPosMA_ds[turnMask[sf:ef]], yPosMA_ds[turnMask[sf:ef]], '.', color='red', alpha=0.7)
ax0.plot(0, 0, marker='o', markersize=15, linestyle='none', alpha=0.5, color='black')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
ax0.set_aspect('equal')
myAxisTheme(ax0)

ax1 = headTurnFig.add_subplot(132)
ax1 = niceScatterPlot(ax1, objDist[selectedRangeDist], gamma[selectedRangeDist], distRange, angleRange, 'grey', 0.3)
ax1 = niceScatterPlot(ax1, objDist[selectedRangeDistTurn], gamma[selectedRangeDistTurn], distRange, angleRange, 'red', 0.7)
plt.xlabel('distance from object')
plt.ylabel('heading angle')

ax2 = headTurnFig.add_subplot(133)
ax2 = niceScatterPlot(ax2, gammaV[selectedRangeDist], gamma[selectedRangeDist], vHeadRange, angleRange, 'grey', 0.3)
ax2 = niceScatterPlot(ax2, gammaV[selectedRangeDistTurn], gamma[selectedRangeDistTurn], vHeadRange, angleRange, 'red', 0.7)
plt.xlabel('change in heading angle (pos - increase, neg - decrease)')
plt.ylabel('heading angle')

try:
    mkdir(analysisDir + 'relativeHeading/')
except OSError:
    print('Analysis directory already exists.')
try:
    mkdir(analysisDir + 'relativeHeading/' + objecttype + '/')
except OSError:
    print('Plot directory already exists.')

headTurnFig.savefig(analysisDir + 'relativeHeading/' + objecttype + '/'
                    + FODataFile[0:-4] + '_headingAndTurns_ds_trial' + str(trial) + '.pdf', format='pdf')

# Visualise effect of turns ............................................................................................
turnEffectFig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(3, 3, height_ratios=np.hstack((1, 1.5, 1.5)), width_ratios=np.hstack((1.5, 1, 1)))

ax0 = turnEffectFig.add_subplot(gs[:, 0])
ax0.plot(xPosMA_ds[vTrans_ds[sf:ef] > vTransTH], yPosMA_ds[vTrans_ds[sf:ef] > vTransTH], '.', color='grey', alpha=0.4)
ax0.plot(xPosMA_ds[turnMask[vTrans_ds[sf:ef] > vTransTH]], yPosMA_ds[turnMask[vTrans_ds[sf:ef] > vTransTH]], '.', color='red', alpha=0.7)
ax0.plot(0, 0, marker='o', markersize=15, linestyle='none', alpha=0.5, color='black')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
ax0.set_aspect('equal')
myAxisTheme(ax0)
ax0.set_title('Effect of turns in ' + flyID + ', trial' + str(trial) +
              '\n percentage turns: ' + str(round(100.0*sum(turnMask)/len(vRotFilt_ds), 2)) + '\n', fontsize=13)

ax = turnEffectFig.add_subplot(gs[0, 1:3])
plt.hist(gammaFull[~np.isnan(gammaFull[sf:ef])], bins=50, color='lightblue', alpha=0.8)
plt.hist(gamma[~np.isnan(gamma[sf:ef])], bins=50, color='grey', alpha=0.5)
plt.xlabel('relative heading angle [rad], not filtered for vTrans >' + str(vTransTH))
plt.ylabel('count')
myAxisTheme(ax)

headingHist = plotVeloHistogram_fancy(gamma[selectedRangeDist], gs[1, 1], (0, np.pi), 'grey', 0.5)
headingHist.set_ylabel('walking filtered\n count')

headingHistTurn = plotVeloHistogram_fancy(gamma[selectedRangeDistTurn], gs[2, 1], (0, np.pi), 'red', 0.5)
headingHistTurn.set_ylabel('walking & turn filtered\n count')
headingHistTurn.set_xlabel('heading angle\n [rad]')

rotVHist = plotVeloHistogram_fancy(gammaV[selectedRangeDist], gs[1, 2], (-10, 10), 'grey', 0.5)
rotVHist.set_xlabel('change in heading while walking\n [rad/s]')

rotVHistFilt = plotVeloHistogram_fancy(gammaV[selectedRangeDistTurn], gs[2, 2], (-10, 10), 'red', 0.5)
rotVHistFilt.set_xlabel('change in heading during turn\n [rad/s]')

turnEffectFig.tight_layout()

try:
    mkdir(analysisDir + 'effectOfTurn/')
except OSError:
    print('Analysis directory already exists.')
try:
    mkdir(analysisDir + 'effectOfTurn/' + objecttype + '/')
except OSError:
    print('Plot directory already exists.')

turnEffectFig.savefig(analysisDir + 'effectOfTurn/' + objecttype + '/'
                      + FODataFile[0:-4] + '_turnHeadingChange_trial' + str(trial) + '.pdf', format='pdf')


# Directional modulation of runs (Gomez-Marin and Louis, 2014) .........................................................

turnModfig = plt.figure(figsize=(12, 7))
turnMod = turnModfig.add_subplot(111)
turnModsc = turnMod.scatter(gammaFull[selectedRangeDist], vRotFilt_ds[selectedRangeDist], marker='o', s=15, linewidths=0,
                c=objDist[selectedRangeDist], cmap=plt.cm.Spectral, alpha=0.5)
turnMod.scatter(gammaFull[selectedRangeDistTurn], vRotFilt_ds[selectedRangeDistTurn], marker='o', s=15,
                linewidths=0.5, c=objDist[selectedRangeDistTurn], cmap=plt.cm.Spectral, alpha=0.5)
turnMod.set_xlim(-np.pi, np.pi)
turnMod.set_ylim(-5, 5)
turnMod.set_xlabel('relative heading [rad]\n(or local bearing)')
turnMod.set_ylabel('instantaneous rot. velocity [rad/s]\n(filtered)')

turnModcb = plt.colorbar(turnModsc)
turnModcb.set_label('distance from object [mm]')
myAxisTheme(turnMod)

turnMod.set_title(flyID + ' in VR (within range ' + str(near) + '-' + str(far) + 'mm)',  fontsize=13)

try:
    mkdir(analysisDir + 'headingVsRotation/')
except OSError:
    print('Analysis directory already exists.')
try:
    mkdir(analysisDir + 'headingVsRotation/' + objecttype + '/')
except OSError:
    print('Plot directory already exists.')

turnModfig.savefig(analysisDir + 'headingVsRotation/' + objecttype + '/'
                   + FODataFile[0:-4] + '_headingVsRotation_trial' + str(trial) + '.pdf', format='pdf')

# ----------------------------------------------------------------------------------------------------------------------
# Save position and velocities for future analysis
# ----------------------------------------------------------------------------------------------------------------------

toSave = {'time': time_ds,
          'xPos': xPos_ds,
          'yPos': yPos_ds,
          'xPosInMiniarena': xPosMA_ds,
          'yPosInMiniarena': yPosMA_ds,
          'headingAngle': angle_ds,
          'rotVelo': vRot_ds,
          'transVelo': vTrans_ds,
          'objectDistance': objDist,
          'gamma': gamma}
# Alternatively use numpy array:
# toSave = np.zeros((10, len(time_ds)))
# toSave[:,:] = np.vstack((time_ds,xPos_ds, yPos_ds, xPosMA_ds, yPosMA_ds, angle_ds, vRot_ds,vTrans_ds,objDist,gamma))

# Save data in this format as *.npy for easy loading..
np.save(expDir + FODataFile[:-4], toSave)

# ----------------------------------------------------------------------------------------------------------------------
# TODO Comparison of flyVR vs. treadmill trajectory
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
print('\n \n Analysis ran successfully. \n \n')
# ----------------------------------------------------------------------------------------------------------------------
