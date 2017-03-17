""" Function for analysing a single stripe fixation trial """

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import matplotlib.colors as colors
from os import mkdir
from os.path import sep
from scipy.interpolate import interp1d

import seaborn as sns
sns.set_style('ticks')

from sys import path

# import basic data processing function
path.insert(1, '/Users/hannah/Dropbox/code/flyVR/utilities/')
from loadSingleTrial import loadSingleVRLogfile
from loadObjectCoords import loadObjectCoordIdentities

path.insert(1, '/Users/hannah/Dropbox/code/trajectoryAnalysis/')
from downsample import donwsampleFOData
from trajectoryDerivedParams import convertRawHeadingAngle, velocityFromTrajectory, relationToObject
from periodicWorldAnalysis import collapseToMiniArena

# import custom plotting functions
path.insert(1, '/Users/hannah/Dropbox/code/plottingUtilities/')
from plottingUtilities import myAxisTheme
from velocityDistributionPlots import velocitySummaryPlot

# ----------------------------------------------------------------------------------------------------------------------
# Define basic analysis script function
# ----------------------------------------------------------------------------------------------------------------------

def processStripetrackingTrial(experimentDir, logFile, dataDir, titleString):

    # Load FlyOver log file, extract calibration paramater and name of cood file

    header, FOData, numFrames, frameRange, calibParams, coordFile = loadSingleVRLogfile(experimentDir, logFile)

    # Read in object coordinates
    # visObjCoords, visObjName, invisObjCoords, origin = loadObjectCoordIdentities(dataDir, coordFile)

    # Compute derived values
    # Compute movement velocities
    logTime = np.copy(FOData[:, 0])
    time = np.linspace(0, logTime[-1], numFrames)
    angle = convertRawHeadingAngle(FOData[:, 5])

    # Compute fly's walking velocieties from TM raw values
    dx1 = FOData[:, 6]
    dy1 = FOData[:, 7]
    dx2 = FOData[:, 8]
    dy2 = FOData[:, 9]

    dtime = np.diff(time)
    dtime = np.hstack((dtime[0], dtime))

    # parameter definitions
    gammaRad = 45*np.pi/180
    rBall = float(calibParams[0])
    pixel2mm = 0.013514

    conversionFactor_pitch = (1.0/pixel2mm)*float(calibParams[0])*2.0*(np.pi/2.0)*(1.0/float(calibParams[1]))
    conversionFactor_yaw = (1.0/pixel2mm)*float(calibParams[0])*2.0*(np.pi/2.0)*(1.0/float(calibParams[2]))

    # compute virtual rotation of fly
    #   add components along longitudinal axis
    vFwdBall = - (dy1 + dy2) * np.cos(gammaRad)
    #   add components along transversal axis
    vSideBall = - (dy1 - dy2) * np.sin(gammaRad)
    #   average measured displacement along aximuth
    vRotBall = - (dx1 + dx2)/2

    # convert A.U. --> pixel --> mm
    vFwdBall = pixel2mm * vFwdBall * conversionFactor_pitch
    vRotBall = pixel2mm * vRotBall * conversionFactor_yaw
    # use mean of pitch and yaw
    vSideBall = pixel2mm * vSideBall * ((conversionFactor_yaw + conversionFactor_pitch)/2)

    # convert to mm/s
    vFwdBall = vFwdBall / dtime
    vSideBall = vSideBall / dtime
    vRotBall = vRotBall / dtime

    # ...mm/s to deg/s
    vRot = - vRotBall / rBall

    # Assume initial position (0 0 0) = (x-coord, y-coord, angle): fly in origin, aligned with x axis (head forward)
    # During measurement coordinate system is fly-centered, moves with fly. Compute all changes along those axes by
    # updating angle and projecting the position changes onto the fixed coordinate system.

    angle = np.cumsum(vRot * dtime)
    angle = np.mod((angle + np.pi), 2*np.pi) - np.pi

    # movement in x and y direction
    #   compute increments
    yTM_i = vSideBall * np.cos(-angle) - vFwdBall * np.sin(-angle)
    xTM_i = vSideBall * np.sin(-angle) + vFwdBall * np.cos(-angle)
    #   integrate increments
    yPos = np.cumsum(yTM_i* dtime)
    xPos = np.cumsum(xTM_i * dtime)

    # vTrans = np.hypot(xTM_i, yTM_i)

    # Downsample data to 20 Hz
    samplingRate = 20
    time_ds, xPos_ds, yPos_ds, angle_ds, numFrames_ds = donwsampleFOData(samplingRate, logTime, time, xPos, yPos, angle)

    # and compute downsampled velocities
    N = 5
    vTrans_ds, vRot_ds, vTransFilt_ds, vRotFilt_ds\
        = velocityFromTrajectory(time_ds, angle_ds, xPos_ds, yPos_ds, N, numFrames_ds)

    # Classify time point as 'moving' or 'non-moving' based on transl. velocity
    vTransTH = 2.0
    moving = vTrans_ds > vTransTH

    ## Compute relative heading
    gamma = abs(angle_ds)
    gammaFull = angle_ds

    # Generate basic processing plots
    # Time step plot
    tstpfig = plt.figure(figsize=(10, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=np.hstack((2, 1)))
    tstpfig.suptitle(titleString, fontsize=13)
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
        mkdir(dataDir + 'analysis/timeStepPlot/')
    except OSError:
        print('Plot directory already exists.')

    tstpfig.savefig(dataDir + 'analysis/timeStepPlot/' + logFile[0:-4] + '_timeStepPlot.pdf', format='pdf')

    # Plot velocity distributions of downsampled data
    rotLim = (-5, 5)
    transLim = (0, 30)
    angleLim = (-np.pi, np.pi)
    summaryVeloFig_ds = velocitySummaryPlot(time_ds, vTrans_ds, vTransFilt_ds, vRot_ds, vRotFilt_ds, angle_ds,
                                            rotLim, transLim, angleLim,
                                            'Downsampled and filtered downsampled velocity traces, ' + titleString)

    try:
        mkdir(dataDir + 'analysis/velocityTraces/')
    except OSError:
        print('Plot directory already exists.')

    summaryVeloFig_ds.savefig(dataDir + 'analysis/velocityTraces/' + logFile[0:-4] + '_veloTraces_ds.pdf',
                              format='pdf')

    # Heading angle distribution
    headingfig = plt.figure(figsize=(10, 8))

    gammaPlt = headingfig.add_subplot(221)
    histRange = (0, np.pi)
    nhead, edges = np.histogram(gamma[moving > 0], density=True, range=histRange, bins=18)
    normFactor = nhead.sum()
    gammaPlt.plot(edges[:-1]+np.diff(edges)/2, nhead/normFactor)
    gammaPlt.set_xlim(histRange)
    gammaPlt.set_xlabel('rel. heading')
    gammaPlt.set_ylabel('frequency (when moving)')
    myAxisTheme(gammaPlt)

    gammaFullPlt = headingfig.add_subplot(222)
    histRange = (-np.pi, np.pi)
    nhead, edges = np.histogram(gammaFull[moving > 0], density=True, range=histRange, bins=36)
    normFactor = nhead.sum()
    gammaFullPlt.plot(edges[:-1]+np.diff(edges)/2, nhead/normFactor)
    gammaFullPlt.set_xlim(histRange)
    gammaFullPlt.set_xlabel('rel. heading (full)')
    myAxisTheme(gammaFullPlt)

    gammaPlt = headingfig.add_subplot(223)
    histRange = (0, np.pi)
    nhead, edges = np.histogram(gamma, density=True, range=histRange, bins=18)
    normFactor = nhead.sum()
    gammaPlt.plot(edges[:-1]+np.diff(edges)/2, nhead/normFactor, color='grey')
    gammaPlt.set_xlim(histRange)
    gammaPlt.set_xlabel('rel. heading')
    gammaPlt.set_ylabel('frequency')
    myAxisTheme(gammaPlt)

    gammaFullPlt = headingfig.add_subplot(224)
    histRange = (-np.pi, np.pi)
    nhead, edges = np.histogram(gammaFull, density=True, range=histRange, bins=36)
    normFactor = nhead.sum()
    gammaFullPlt.plot(edges[:-1]+np.diff(edges)/2, nhead/normFactor, color='grey')
    gammaFullPlt.set_xlim(histRange)
    gammaFullPlt.set_xlabel('rel. heading (full)')
    myAxisTheme(gammaFullPlt)

    headingfig.suptitle(titleString, fontsize=13)
    headingfig.tight_layout()

    try:
        mkdir(dataDir + 'analysis/heading/')
    except OSError:
        print('Plot directory already exists.')

    headingfig.savefig(dataDir + 'analysis/heading/' + logFile[0:-4] + '_headingDistribution.pdf', format='pdf')

    # Save values for future analysis

    xPosMA_ds = np.nan*np.ones(np.size(time_ds))
    yPosMA_ds = np.nan*np.ones(np.size(time_ds))

    toSave = {'time': time_ds,
              'xPos': xPos_ds,
              'yPos': yPos_ds,
              'xPosInMiniarena': xPosMA_ds,
              'yPosInMiniarena': yPosMA_ds,
              'headingAngle': angle_ds,
              'rotVelo': vRot_ds,
              'transVelo': vTrans_ds,
              'gammaFull': gammaFull,
              'gamma': gamma,
              'moving': moving}

    np.save(experimentDir + logFile[:-4], toSave)

    return 0
