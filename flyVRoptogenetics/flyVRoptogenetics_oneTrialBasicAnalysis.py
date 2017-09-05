""" Funtion for analysing a single FlyOver trial (assumes use of FlyOver version 9.2 or 9.3) """

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

from os import mkdir, getcwd
from os.path import isfile, sep
from sys import path, exit
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
import seaborn as sns

# Set path to analysis code directory
codeDir = sep.join(getcwd().split(sep)[:-2])
path.insert(1, codeDir)

# Import custom plotting functions
from plottingUtilities.basicPlotting import myAxisTheme, timeAxisTheme, niceScatterPlot, makeNestedPlotDirectory
from plottingUtilities.flyTracePlots import plotPosInRange, plotFlyVRtimeStp, plotPolarTrace
from plottingUtilities.velocityDistributionPlots import plotVeloHistogram_fancy, velocitySummaryPlot

# Import custom data processing functions
from flyVR.utilities.loadSingleTrial import loadSingleVRLogfile, rZoneParamsFromLogFile
from flyVR.utilities.loadObjectCoords import loadObjectCoords, loadObjectCoordIdentities
from flyVR.utilities.objectInteractionPlots import modulationOfRuns, residencyWithHistograms_splitOnWalking,\
    curvatureVsHeading_DistanceBoxplot, plotResidencyInMiniarena

from trajectoryAnalysis.downsample import donwsampleFOData
from trajectoryAnalysis.trajectoryDerivedParams import convertRawHeadingAngle, velocityFromTrajectory, relationToObject, cartesian2polar,\
    polarCurvature
from trajectoryAnalysis.periodicWorldAnalysis import collapseToMiniArena, collapseTwoObjGrid

sns.set_style('ticks')

def singleVROptogenTrialAnalysis(fileToAnalyse):
    # fileToAnalyse should be to complete path to the logfile of the FlyOver trial to be analysed.

    # TODO: add savePlots,recomputeFlyData as function arguments

    # ------------------------------------------------------------------------------------------
    # Extract folder and file name info
    # ------------------------------------------------------------------------------------------

    print('Data file: \n' + fileToAnalyse + '\n')

    dataDir = sep.join(fileToAnalyse.split(sep)[0:-3]) + sep
    flyID = fileToAnalyse.split(sep)[-2]
    expDir = sep.join(fileToAnalyse.split(sep)[0:-1]) + sep

    if not ('males' in expDir.split(sep)[-4] or 'females'in expDir.split(sep)[-4] or 'FlyOverDebugging'):
        print('You selected an invalid data directory.\n' +
              'Expected folder structure of the selected path is some/path/to/experimentName/flyGender/rawData/')
        exit(1)

    genotype = expDir.split(sep)[-5]

    # create analysis dir
    analysisDir = sep.join(dataDir.split(sep)[:-1]) + sep + 'analysis' + sep
    try:
        mkdir(analysisDir)
    except OSError:
        print('Analysis directory already exists.')

    FODataFile = fileToAnalyse.split(sep)[-1]
    FODataFiles = [filepath.split(sep)[-1] for filepath in glob(expDir + '*.txt')]
    FODataFiles = sorted(FODataFiles)

    trial = FODataFiles.index(FODataFile) + 1

    # ------------------------------------------------------------------------------------------
    # Load or read in logfile data
    # ------------------------------------------------------------------------------------------

    # Read in logfile data, parse header ...............................................................................
    header, FOData, numFrames, frameRange, calibParams, coordFile = loadSingleVRLogfile(expDir, FODataFile)

    if 'rZones' in coordFile:
        rZones = 'on'
    else:
        rZones = 'off'

    if 'invisible' in coordFile or 'Invisible' in coordFile:
        invisible = 'on'
        objecttype = 'invisible'

    else:
        invisible = 'off'
        objecttype = 'visible'

    dataFileParts = FODataFile.split('_')

    titleString = genotype + ' fly "' + flyID + '" in ' + dataFileParts[0] + ' of ' + dataFileParts[1] + 's\n' + \
        'with reinforcement zones ' + rZones + ', trial' + str(trial)
    print(titleString)

    # Extract reinforcement zone parameter ......................................................
    rZone_rInner, rZone_rOuter, rZone_max, rZone_bl, rZone_gExp = rZoneParamsFromLogFile(expDir, FODataFile)

    # Read in object coordinates ................................................................
    visibleObjectCoords, invisibleObjectCoords, origin = loadObjectCoords(dataDir, coordFile)

    # Compute movement velocities ...............................................................
    logTime = np.copy(FOData[:, 0])
    time = np.linspace(0, logTime[-1], numFrames)

    angle = convertRawHeadingAngle(FOData[:, 5])

    # N = 60
    # vTrans, vRot, vTransFilt, vRotFilt = velocityFromTrajectory(time, angle, FOData[:, 1], FOData[:, 2], N, numFrames)

    # Down sample data to 20 Hz .................................................................
    samplingRate = 20
    time_ds, xPos_ds, yPos_ds, angle_ds, numFrames_ds \
        = donwsampleFOData(samplingRate, logTime, time, FOData[:, 1], FOData[:, 2], angle)

    # and compute downsampled velocities
    N = 5
    vTrans_ds, vRot_ds, vTransFilt_ds, vRotFilt_ds \
        = velocityFromTrajectory(time_ds, angle_ds, xPos_ds, yPos_ds, N, numFrames_ds)

    # -------------------------------------------------------------------------------------------
    # Generate basic analysis plots
    # -------------------------------------------------------------------------------------------
    # Time step plot ............................................................................

    plotStp = 5
    tstpfig = plotFlyVRtimeStp(plotStp, FOData[:, 0], titleString)

    makeNestedPlotDirectory(analysisDir, 'timeStepPlot/', 'rZones_' + rZones + sep)

    tstpfig.savefig(analysisDir + 'timeStepPlot/' + 'rZones_' + rZones + sep
                    + FODataFile[0:-4] + '_timeStepPlot_trial' + str(trial) + '.pdf', format='pdf')

    # Plot of walking trace (+ colorbar for time) with object locations .........................
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
    axTraj.scatter(invisibleObjectCoords[4:, 0], invisibleObjectCoords[4:, 1], 50, alpha=0.5,
                   facecolors='none', edgecolors='black')
    axTraj.set_xlabel(header[1], fontsize=12)
    axTraj.set_ylabel(header[2], fontsize=12)
    axTraj.set_title('Walking trace of ' + titleString, fontsize=14)
    axTraj.set_xlim([max(-650, min(FOData[:, 1]) - 20), min(650, max(FOData[:, 1]) + 20)])
    axTraj.set_ylim([max(-650, min(FOData[:, 2]) - 20), min(650, max(FOData[:, 2]) + 20)])
    myAxisTheme(axTraj)

    axTime.set_xlabel(header[0], fontsize=12)
    plt.xlim((0, FOData[-1, 0]))
    timeAxisTheme(axTime)

    makeNestedPlotDirectory(analysisDir, 'tracePlot/', 'rZones_' + rZones + sep)

    trajfig.savefig(analysisDir + 'tracePlot/' + 'rZones_' + rZones + sep
                    + FODataFile[0:-4] + '_traceObjectPlot_trial' + str(trial) + '.pdf', format='pdf')

    # Visualise strength of optogenetic stimulation .............................................

    rEvents = FOData[:, 11]
    # downsample rEvents
    f_rEvents = interp1d(time, rEvents, kind='linear')
    rEvents_ds = f_rEvents(time_ds)

    tStep = 72
    frameRange = range(tStart, tEnd, tStep)

    trajRZfig = plt.figure(figsize=(10, 10))
    axTraj = trajRZfig.add_subplot(111)

    axTraj.scatter(visibleObjectCoords[:, 0], visibleObjectCoords[:, 1], 50, alpha=0.5,
                   facecolor='black', edgecolors='none')
    axTraj.scatter(invisibleObjectCoords[4:, 0], invisibleObjectCoords[4:, 1], 50, alpha=0.5,
                   facecolors='none', edgecolors='black')

    plt.plot(FOData[frameRange, 1], FOData[frameRange, 2], marker='.', markerfacecolor='grey',
             markeredgecolor='none', alpha=0.25)

    # overlay reinforcement value
    axTraj.scatter(FOData[frameRange, 1], FOData[frameRange, 2], s=rEvents[frameRange]*10,
                   c=rEvents[frameRange]*10, alpha=0.7, cmap=plt.get_cmap('Reds'))

    axTraj.set_xlabel(header[1], fontsize=12)
    axTraj.set_ylabel(header[2], fontsize=12)
    axTraj.set_title('Walking trace of ' + titleString, fontsize=14)
    axTraj.set_xlim([max(-650, min(FOData[:, 1]) - 20), min(650, max(FOData[:, 1]) + 20)])
    axTraj.set_ylim([max(-650, min(FOData[:, 2]) - 20), min(650, max(FOData[:, 2]) + 20)])
    axTraj.set_aspect('equal')
    myAxisTheme(axTraj)

    makeNestedPlotDirectory(analysisDir, 'tracePlotRZ/', 'rZones_' + rZones + sep)

    trajRZfig.savefig(analysisDir + 'tracePlotRZ/' + 'rZones_' + rZones + sep
                      + FODataFile[0:-4] + '_traceObjectPlot_trial' + str(trial) + '.pdf', format='pdf')

    # Plot velocity distributions of downs sampled data .........................................
    rotLim = (-5, 5)
    transLim = (0, 30)
    angleLim = (-np.pi, np.pi)
    summaryVeloFig_ds = velocitySummaryPlot(time_ds, vTrans_ds, vTransFilt_ds, vRot_ds, vRotFilt_ds, angle_ds, rotLim,
                                            transLim, angleLim, 'Down sampled velocity traces, ' + titleString)

    makeNestedPlotDirectory(analysisDir, 'velocityTraces/', 'rZones_' + rZones + sep)

    summaryVeloFig_ds.savefig(analysisDir + 'velocityTraces/' + 'rZones_' + rZones + sep
                              + FODataFile[0:-4] + '_veloTraces_ds_trial' + str(trial) + '.pdf', format='pdf')

    # -------------------------------------------------------------------------------------------
    #  Collapse traces to single object cell and plot resulting trace
    # -------------------------------------------------------------------------------------------

    # Collapse to 'mini-arena' while preserving the global heading .............................
    arenaRad = 60 # 1/2 distance between cones
    if invisible == 'off':
        objectCoords = np.copy(visibleObjectCoords[0:-3, 0:2])
    else:  # use of non-physics, invisible objects allows us to mark virtual object positions in empty arena
        objectCoords = np.copy(invisibleObjectCoords[4:, 0:2])

    xPosMA, yPosMA = collapseToMiniArena(FOData[:, 1], FOData[:, 2], arenaRad, objectCoords)

    # Compute donw sampled collapsed traces
    f_xPosMA = interp1d(time, xPosMA, kind='linear')
    f_yPosMA = interp1d(time, yPosMA, kind='linear')

    xPosMA_ds = f_xPosMA(time_ds)
    yPosMA_ds = f_yPosMA(time_ds)

    # Plot collapsed, down sampled trace ........................................................
    tStart = 0
    tEnd = numFrames_ds
    tStep = 4
    frameRange = range(tStart, tEnd, tStep)
    colMap = 'Accent'

    colTrFig = plt.figure(figsize=(9, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=np.hstack((10, 1)))

    colTrFig.suptitle('Collapsed walking trace ("mini arena" with central object)\n' + titleString, fontsize=14)

    axTraj = colTrFig.add_subplot(gs[0])
    axTime = colTrFig.add_subplot(gs[1])
    plotPosInRange(axTraj, axTime, frameRange, time_ds, xPosMA_ds, yPosMA_ds, angle_ds, colMap, 4, 0.5, 7)

    if invisible == 'off':
        axTraj.plot(0, 0, marker='o', markersize=20, linestyle='none', alpha=0.75, color='black')
    else:
        axTraj.plot(0, 0, marker='o', markersize=20, alpha=0.75, markeredgewidth=0.5,
                    markerfacecolor='None', markeredgecolor='black')

    if rZones == 'on':
        rZoneRange = float(rZone_rOuter - rZone_rInner)
        for zRad in range(rZone_rInner, rZone_rOuter):
            circle1 = plt.Circle((0, 0), zRad, color='r', alpha=1.0/rZoneRange)
            axTraj.add_artist(circle1)

    axTraj.set_xlabel(header[1], fontsize=12)
    axTraj.set_ylabel(header[2], fontsize=12)
    axTraj.set_ylim([-(arenaRad+5), arenaRad + 5])
    axTraj.set_xlim([-(arenaRad+5), arenaRad + 5])
    myAxisTheme(axTraj)

    axTime.set_xlabel(header[0], fontsize=12)
    plt.xlim((0, time_ds[-1]))
    timeAxisTheme(axTime)

    makeNestedPlotDirectory(analysisDir, 'tracePlotMA/', 'rZones_' + rZones + sep)

    colTrFig.savefig(analysisDir + 'tracePlotMA/' + 'rZones_' + rZones + sep
                     + FODataFile[0:-4] + '_traceObjectPlot_ds_trial' + str(trial) + '.pdf', format='pdf')

    # -------------------------------------------------------------------------------------------
    # Compute heading angle relative to closest object (use collapsed coordinates)
    # -------------------------------------------------------------------------------------------

    # Compute parameters characterising fly's relationship to object ...................................................
    objLocation = [0, 0]
    objDirection, objDistance, gammaFull, gamma, gammaV \
        = relationToObject(time_ds, xPosMA_ds, yPosMA_ds, angle_ds, objLocation)

    # Change in heading rel. to object ..........................................................
    near = 6
    far = arenaRad
    vTransTH = 2

    turnTH = 2.5*np.std(abs(vRotFilt_ds))
    turnMask = (abs(vRotFilt_ds) > turnTH)

    preTurnMask = np.hstack((turnMask[samplingRate/20:], np.zeros(samplingRate/20)))

    selectedRangeDistAll = np.logical_and(objDistance > near, objDistance < far)

    selectedRangeDist = np.logical_and(np.logical_and(objDistance > near, objDistance < far), vTrans_ds > vTransTH)
    selectedRangeDistPreTurnWalk = np.logical_and(selectedRangeDist, preTurnMask)
    selectedRangeDistPreTurn = np.logical_and(np.logical_and(objDistance > near, objDistance < far), preTurnMask)
    selectedRangeDistTurn = np.logical_and(np.logical_and(objDistance > near, objDistance < far), turnMask)

    # Visualise effect of turns .................................................................
    turnEffectFig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(3, 3, height_ratios=np.hstack((1, 1.5, 1.5)), width_ratios=np.hstack((1.5, 1, 1)))

    ax0 = turnEffectFig.add_subplot(gs[:, 0])
    ax0.plot(xPosMA_ds[selectedRangeDist], yPosMA_ds[selectedRangeDist], '.', color='grey', alpha=0.4)
    ax0.plot(xPosMA_ds[selectedRangeDistTurn], yPosMA_ds[selectedRangeDistTurn], '.', color='lightblue', alpha=0.7)
    ax0.plot(xPosMA_ds[selectedRangeDistPreTurn], yPosMA_ds[selectedRangeDistPreTurn], '.', color='red', alpha=0.4)
    ax0.plot(0, 0, marker='o', markersize=15, linestyle='none', alpha=0.5, color='black')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    ax0.set_xlim((-arenaRad, arenaRad))
    ax0.set_ylim((-arenaRad, arenaRad))
    ax0.set_aspect('equal')
    myAxisTheme(ax0)
    ax0.set_title('Effect of turns in ' + flyID + ', trial' + str(trial) +
                  '\n percentage turns: ' + str(round(100.0*sum(turnMask)/len(vRotFilt_ds), 2)) + '\n', fontsize=13)

    ax = turnEffectFig.add_subplot(gs[0, 1:3])
    gammaFullSelect = gammaFull[selectedRangeDistAll]
    gammaSelect = gamma[selectedRangeDistAll]
    try:
        plt.hist(gammaFullSelect[~np.isnan(gammaFullSelect)], bins=50, color='lightblue', alpha=0.8)
        plt.hist(gammaSelect[~np.isnan(gammaSelect)], bins=50, color='grey', alpha=0.5)
    except ValueError:
        print('Not enough values for histogram.')
    plt.xlabel('relative heading angle [rad], not filtered for vTrans > ' + str(vTransTH))
    plt.ylabel('count')
    ax.set_xlim((-np.pi, np.pi))
    myAxisTheme(ax)

    headingHist = plotVeloHistogram_fancy(gamma[selectedRangeDist], gs[1, 1], (0, np.pi), 'grey', 0.5)
    headingHist.set_ylabel('walking filtered\n count (vTrans > ' + str(vTransTH) + ')')

    headingHistTurn = plotVeloHistogram_fancy(gamma[selectedRangeDistPreTurnWalk], gs[2, 1], (0, np.pi), 'red', 0.5)
    headingHistTurn.set_ylabel('walking & turn filtered\n count')
    headingHistTurn.set_xlabel('heading angle\n [rad]')

    rotVHist = plotVeloHistogram_fancy(gammaV[selectedRangeDist], gs[1, 2], (-10, 10), 'grey', 0.5)
    rotVHist.set_xlabel('change in heading while walking\n [rad/s]')

    rotVHistFilt = plotVeloHistogram_fancy(gammaV[selectedRangeDistPreTurnWalk], gs[2, 2], (-10, 10), 'red', 0.5)
    rotVHistFilt.set_xlabel('change in heading during turn\n [rad/s]')

    turnEffectFig.tight_layout()

    makeNestedPlotDirectory(analysisDir, 'effectOfTurn/', 'rZones_' + rZones + sep)

    turnEffectFig.savefig(analysisDir + 'effectOfTurn/' + 'rZones_' + rZones + sep
                          + FODataFile[0:-4] + '_turnHeadingChange_trial' + str(trial) + '.pdf', format='pdf')

    # -------------------------------------------------------------------------------------------
    # Convert trajectory to polar coordinates and visualise trace and effect of turns
    # -------------------------------------------------------------------------------------------

    # transform trajectory to polar coordinates
    objDist, theta = cartesian2polar(xPosMA_ds, yPosMA_ds)

    # compute curvature
    polarCurv, d_theta, dtheta_objDist = polarCurvature(theta, objDist)

    d_objDist = np.hstack((0, np.diff(objDist)))

    # Compute sign of turn relative to object
    turnSign = np.sign(polarCurv)
    turnSign[d_theta > 0] = np.sign(polarCurv[d_theta > 0])
    turnSign[d_theta < 0] = -np.sign(polarCurv[d_theta < 0])

    # Compute curvature-based criterion for turns
    # q2, q98 = np.percentile(polarCurv[~np.isnan(polarCurv)], [2, 98])
    # curvSelect = abs(polarCurv) < (q98 - q2)/2
    # curvTurnMask_L = polarCurv > q98
    # curvTurnMask_R = polarCurv < q2

    # Generate filtered curvature for plots and curvature magnitude
    # polarCurvPlt = polarCurv[curvSelect]
    # curvMag = abs(polarCurv)
    # correctedPolarCurv = abs(polarCurv)*turnSign

    selectPts_apr = d_objDist < 0
    selectPts_dep = d_objDist > 0

    selectPts_apr_turnR = np.logical_and(selectPts_apr, vRotFilt_ds < -1*turnTH)
    selectPts_apr_turnL = np.logical_and(selectPts_apr, vRotFilt_ds > 1*turnTH)
    selectPts_dep_turnR = np.logical_and(selectPts_dep, vRotFilt_ds < -1*turnTH)
    selectPts_dep_turnL = np.logical_and(selectPts_dep, vRotFilt_ds > 1*turnTH)

    fig = plt.figure(figsize=(15, 10))

    xlimRange = (5, 60)

    ax = fig.add_subplot(211)
    ax = plotPolarTrace(ax, titleString + '\nApproaches to ' + objecttype + ' object',
                        selectPts_apr, selectPts_apr_turnR, selectPts_apr_turnL, objDist, gammaFull, vRot_ds, xlimRange)

    ax = fig.add_subplot(212)
    ax = plotPolarTrace(ax, 'Departures from ' + objecttype + ' object',
                        selectPts_dep, selectPts_dep_turnR, selectPts_dep_turnL, objDist, gammaFull, vRot_ds, xlimRange)

    makeNestedPlotDirectory(analysisDir, 'polarTrace/', 'rZones_' + rZones + sep)

    fig.savefig(analysisDir + 'polarTrace/' + 'rZones_' + rZones + sep
                + FODataFile[0:-4] + '_polarTrace_trial' + str(trial) + '.pdf', format='pdf')

    # -------------------------------------------------------------------------------------------
    # Save position and velocities for future analysis
    # -------------------------------------------------------------------------------------------

    toSave = {'time': time_ds,
              'xPos': xPos_ds,
              'yPos': yPos_ds,
              'xPosInMiniarena': xPosMA_ds,
              'yPosInMiniarena': yPosMA_ds,
              'headingAngle': angle_ds,
              'rotVelo': vRot_ds,
              'transVelo': vTrans_ds,
              'objectDistance': objDistance,
              'gammaFull': gammaFull,
              'gamma': gamma,
              'rEvents': rEvents_ds}

    # Alternatively use numpy array:
    # toSave=np.vstack((time_ds,xPos_ds, yPos_ds, xPosMA_ds, yPosMA_ds, angle_ds, vRot_ds,vTrans_ds,objDistance,gamma))

    # Save data in this format as *.npy for easy loading..
    np.save(expDir + FODataFile[:-4], toSave)

    # -------------------------------------------------------------------------------------------
    print('\n \n Analysis ran successfully. \n \n')
    # -------------------------------------------------------------------------------------------

    plt.close('all')

    return 0


def singleTwoObjVROptogenTrialAnalysis(fileToAnalyse):
    # fileToAnalyse should be to complete path to the logfile of the FlyOver trial to be analysed.

    # TODO: add savePlots,recomputeFlyData as function arguments

    # -------------------------------------------------------------------------------------------
    # Extract folder and file name info
    # -------------------------------------------------------------------------------------------

    print('Data file: \n' + fileToAnalyse + '\n')

    dataDir = sep.join(fileToAnalyse.split(sep)[0:-3]) + sep
    flyID = fileToAnalyse.split(sep)[-2]
    expDir = sep.join(fileToAnalyse.split(sep)[0:-1]) + sep

    if not ('males' in expDir.split(sep)[-4] or 'females'in expDir.split(sep)[-4] or 'FlyOverDebugging'):
        print('You selected an invalid data directory.\n' +
              'Expected folder structure of the selected path is some/path/to/experimentName/flyGender/rawData/')
        exit(1)

    genotype = expDir.split(sep)[-5]

    # create analysis dir
    analysisDir = sep.join(dataDir.split(sep)[:-1]) + sep + 'analysis' + sep
    try:
        mkdir(analysisDir)
    except OSError:
        print('Analysis directory already exists.')

    FODataFile = fileToAnalyse.split(sep)[-1]
    FODataFiles = [filepath.split(sep)[-1] for filepath in glob(expDir + '*.txt')]
    FODataFiles = sorted(FODataFiles)

    trial = FODataFiles.index(FODataFile) + 1
    trialType = 'pre'
    rZones = 'off'

    if 'train' in FODataFile:
        trialType = 'train'
        rZones = 'on'
    elif 'post' in FODataFile:
        trialType = 'post'
        rZones = 'off'

    dataFileParts = FODataFile.split('_')

    invisible = 'off'
    objecttype = 'visible'

    titleString = genotype + ' fly "' + flyID + '" in ' + dataFileParts[0] + ' of ' + dataFileParts[1] + ' and ' + \
        dataFileParts[2] + '\n' + trialType + ' trial'

    print(titleString)

    # -------------------------------------------------------------------------------------------
    # Load or read in logfile data
    # -------------------------------------------------------------------------------------------

    # Read in logfile data, parse header ...............................................................................
    header, FOData, numFrames, frameRange, calibParams, coordFile = loadSingleVRLogfile(expDir, FODataFile)

    # Extract reinforcement zone parameter ......................................................
    #rZone_rInner, rZone_rOuter, rZone_max, rZone_bl, rZone_gExp = rZoneParamsFromLogFile(expDir, FODataFile)

    # Read in object coordinates ................................................................
    visibleObjectCoords, visibleObjectName, invisibleObjectCoords, origin = loadObjectCoordIdentities(dataDir, coordFile)

    # Compute movement velocities ...............................................................
    logTime = np.copy(FOData[:, 0])
    time = np.linspace(0, logTime[-1], numFrames)

    angle = convertRawHeadingAngle(FOData[:, 5])

    # N = 60
    # vTrans, vRot, vTransFilt, vRotFilt = velocityFromTrajectory(time, angle, FOData[:, 1], FOData[:, 2], N, numFrames)

    # Down sample data to 20 Hz .................................................................
    samplingRate = 20
    time_ds, xPos_ds, yPos_ds, angle_ds, numFrames_ds \
        = donwsampleFOData(samplingRate, logTime, time, FOData[:, 1], FOData[:, 2], angle)

    # and compute downsampled velocities
    N = 5
    vTrans_ds, vRot_ds, vTransFilt_ds, vRotFilt_ds \
        = velocityFromTrajectory(time_ds, angle_ds, xPos_ds, yPos_ds, N, numFrames_ds)

    # -------------------------------------------------------------------------------------------
    # Generate basic analysis plots
    # -------------------------------------------------------------------------------------------
    # Time step plot ............................................................................

    plotStp = 5
    tstpfig = plotFlyVRtimeStp(plotStp, FOData[:, 0], titleString)

    makeNestedPlotDirectory(analysisDir, 'timeStepPlot/', trialType + 'Trial' + sep)

    tstpfig.savefig(analysisDir + 'timeStepPlot/' + trialType + 'Trial' + sep + FODataFile[0:-4] + '_timeStepPlot.pdf',
                    format='pdf')

    # Plot of walking trace (+ colorbar for time) with object locations .........................

    coneShape = np.asarray([bool('Cone' in objName) for objName in visibleObjectName])
    cyliShape = np.asarray([bool('Cyli' in objName) for objName in visibleObjectName])

    tStart = 0
    tEnd = len(FOData[:, 1])
    tStep = 72
    frameRange = range(tStart, tEnd, tStep)
    colMap = 'Accent'
    arrowLength = 4

    trajfig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=np.hstack((10, 1)))

    axTraj = trajfig.add_subplot(gs[0])
    axTime = trajfig.add_subplot(gs[1])

    plotPosInRange(axTraj, axTime, frameRange, FOData[:, 0], FOData[:, 1], FOData[:, 2], np.pi/180*FOData[:, 5],
                   colMap, arrowLength, 0.5, 5)
    axTraj.scatter(visibleObjectCoords[coneShape, 0], visibleObjectCoords[coneShape, 1], 50, alpha=0.5,
                   facecolors='black', edgecolors='none', marker='^')
    axTraj.scatter(visibleObjectCoords[cyliShape, 0], visibleObjectCoords[cyliShape, 1], 50, alpha=0.5,
                   facecolors='black', edgecolors='none', marker='s')

    axTraj.set_xlabel(header[1], fontsize=12)
    axTraj.set_ylabel(header[2], fontsize=12)
    axTraj.set_title('Walking trace of ' + titleString, fontsize=14)
    axTraj.set_xlim([max(-650, min(FOData[:, 1]) - 20), min(650, max(FOData[:, 1]) + 20)])
    axTraj.set_ylim([max(-650, min(FOData[:, 2]) - 20), min(650, max(FOData[:, 2]) + 20)])
    myAxisTheme(axTraj)

    axTime.set_xlabel(header[0], fontsize=12)
    plt.xlim((0, FOData[-1, 0]))
    timeAxisTheme(axTime)

    makeNestedPlotDirectory(analysisDir, 'tracePlot/', trialType + 'Trial' + sep)

    trajfig.savefig(analysisDir + 'tracePlot/' + trialType + 'Trial' + sep
                    + FODataFile[0:-4] + '_traceObjectPlot.pdf', format='pdf')

    # Visualise strength of optogenetic stimulation .............................................

    rEvents = FOData[:, 11]
    # downsample rEvents
    f_rEvents = interp1d(time, rEvents, kind='linear')
    rEvents_ds = f_rEvents(time_ds)

    tStep = 72
    frameRange = range(tStart, tEnd, tStep)

    trajRZfig = plt.figure(figsize=(10, 10))
    axTraj = trajRZfig.add_subplot(111)

    axTraj.scatter(visibleObjectCoords[coneShape, 0], visibleObjectCoords[coneShape, 1], 50, alpha=0.5,
                   facecolors='black', edgecolors='none', marker='^')
    axTraj.scatter(visibleObjectCoords[cyliShape, 0], visibleObjectCoords[cyliShape, 1], 50, alpha=0.5,
                   facecolors='black', edgecolors='none', marker='s')

    plt.plot(FOData[frameRange, 1], FOData[frameRange, 2], marker='.', markerfacecolor='grey',
             markeredgecolor='none', linestyle='none', alpha=0.1)

    #overlay reinforcement val
    axTraj.scatter(FOData[frameRange, 1], FOData[frameRange, 2], s=rEvents[frameRange]*3,
                   c=rEvents[frameRange]*3, alpha=0.7, cmap=plt.get_cmap('Reds'))

    axTraj.set_xlabel(header[1], fontsize=12)
    axTraj.set_ylabel(header[2], fontsize=12)
    axTraj.set_title('Walking trace of ' + titleString, fontsize=14)
    axTraj.set_xlim([max(-650, min(FOData[:, 1]) - 20), min(650, max(FOData[:, 1]) + 20)])
    axTraj.set_ylim([max(-650, min(FOData[:, 2]) - 20), min(650, max(FOData[:, 2]) + 20)])
    axTraj.set_aspect('equal')
    myAxisTheme(axTraj)

    makeNestedPlotDirectory(analysisDir, 'tracePlotRZ/', trialType + 'Trial' + sep)

    trajRZfig.savefig(analysisDir + 'tracePlotRZ/' + trialType + 'Trial' + sep
                    + FODataFile[0:-4] + '_traceObjectPlot.pdf', format='pdf')

    # Plot velocity distributions of downs sampled data .........................................
    rotLim = (-5, 5)
    transLim = (0, 30)
    angleLim = (-np.pi, np.pi)
    summaryVeloFig_ds = velocitySummaryPlot(time_ds, vTrans_ds, vTransFilt_ds, vRot_ds, vRotFilt_ds, angle_ds, rotLim,
                                            transLim, angleLim, 'Down sampled velocity traces, ' + titleString)

    makeNestedPlotDirectory(analysisDir, 'velocityTraces/', trialType + 'Trial' + sep)

    summaryVeloFig_ds.savefig(analysisDir + 'velocityTraces/' + trialType + 'Trial' + sep
                              + FODataFile[0:-4] + '_veloTraces_ds.pdf', format='pdf')

    # -------------------------------------------------------------------------------------------
    #  Collapse traces to single object cell and plot resulting trace
    # -------------------------------------------------------------------------------------------

    # Collapse to 'mini-arena' while preserving the global heading ..............................
    gridSize = 60.0 # closest distance between landmarks
    gridRepeat = (6, 5) # grid height in repeats of gridSize in x and y
    xPosMA, yPosMA = collapseTwoObjGrid(FOData[:, 1], FOData[:, 2], gridSize, gridRepeat)

    # Compute donw sampled collapsed traces
    f_xPosMA = interp1d(time, xPosMA, kind='linear')
    f_yPosMA = interp1d(time, yPosMA, kind='linear')

    xPosMA_ds = f_xPosMA(time_ds)
    yPosMA_ds = f_yPosMA(time_ds)

    # Plot collapsed, down sampled trace ........................................................
    tStart = 0
    tEnd = numFrames_ds
    tStep = 4
    frameRange = range(tStart, tEnd, tStep)
    colMap = 'Accent'

    colTrFig = plt.figure(figsize=(9, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=np.hstack((10, 1)))

    colTrFig.suptitle('Collapsed walking trace ("mini arena" with central object)\n' + titleString, fontsize=14)

    axTraj = colTrFig.add_subplot(gs[0])
    axTime = colTrFig.add_subplot(gs[1])
    plotPosInRange(axTraj, axTime, frameRange, time_ds, xPosMA_ds, yPosMA_ds, angle_ds, colMap, 4, 0.5, 7)

    axTraj.plot(gridSize/2, -gridSize/2, 'ks', markersize=8)
    axTraj.plot(gridSize/2, gridSize/2, 'k^', markersize=10)
    axTraj.plot(3*gridSize/2, -gridSize/2, 'k^', markersize=10)
    axTraj.plot(3*gridSize/2, gridSize/2, 'ks', markersize=8)

    axTraj.set_xlabel(header[1], fontsize=12)
    axTraj.set_ylabel(header[2], fontsize=12)
    axTraj.set_ylim([-(5+gridSize), gridSize+5])
    axTraj.set_xlim([-5, 2*gridSize+5])
    myAxisTheme(axTraj)

    axTime.set_xlabel(header[0], fontsize=12)
    plt.xlim((0, time_ds[-1]))
    timeAxisTheme(axTime)

    makeNestedPlotDirectory(analysisDir, 'tracePlotMA/', trialType + 'Trial' + sep)

    colTrFig.savefig(analysisDir + 'tracePlotMA/' + trialType + 'Trial' + sep
                     + FODataFile[0:-4] + '_traceObjectPlot_ds.pdf', format='pdf')

    # -------------------------------------------------------------------------------------------
    # Save position and velocities for future analysis
    # -------------------------------------------------------------------------------------------

    toSave = {'time': time_ds,
              'xPos': xPos_ds,
              'yPos': yPos_ds,
              'xPosInMiniarena': xPosMA_ds,
              'yPosInMiniarena': yPosMA_ds,
              'headingAngle': angle_ds,
              'rotVelo': vRot_ds,
              'transVelo': vTrans_ds,
              'rEvents': rEvents_ds}

    # Save data in this format as *.npy for easy loading..
    np.save(expDir + FODataFile[:-4], toSave)

    # -------------------------------------------------------------------------------------------
    print('\n \n Analysis ran successfully. \n \n')
    # -------------------------------------------------------------------------------------------

    plt.close('all')

    return 0
