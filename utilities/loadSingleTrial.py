"""
Functions for reading in raw FlyOver data.
Assumes logfile format with header. Tested with FlyOver 093 and later versions.
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np


def loadSingleVRLogfile(expDir, FODataFile):
    # Load or read in logfile data
    # Parse header .....................................................................................................

    with open(expDir + FODataFile) as FOf:
        lines = FOf.read().splitlines()

    # Find beginning of data
    firstDataLine = 0
    while lines[firstDataLine][0] == '#':
        firstDataLine += 1
    header = lines[firstDataLine-1][:].split()[1:]

    # Get calibration parameter from header ............................................................................
    currHeaderLine = 0
    calibParams = []
    while lines[currHeaderLine][0] == '#':
        if (lines[currHeaderLine].split(' ')[1] == 'Treadmill') & (lines[currHeaderLine].split(' ')[2] == 'ball'):
            ballRadius_mm = lines[currHeaderLine].split(' ')[-2]
            xRotCalib_ticksPerSemicirc = lines[currHeaderLine+1].split(' ')[-2]
            yRotCalib_ticksPerSemicirc = lines[currHeaderLine+2].split(' ')[-2]
            zRotCalib_ticksPerSemicirc = lines[currHeaderLine+3].split(' ')[-2]
            # print('ball radius = ' + str(ballRadius_mm))

            calibParams = [ballRadius_mm, xRotCalib_ticksPerSemicirc,
                           yRotCalib_ticksPerSemicirc, zRotCalib_ticksPerSemicirc]
            break
        currHeaderLine += 1

    # Parse data .......................................................................................................
    # Move text data into a np array
    numFrames = len(lines[firstDataLine:-1])

    # Check if data has been loaded before
    frameRange = range(0, numFrames)

    FOData = np.zeros((numFrames, 12))
    tmp = np.asarray(lines[firstDataLine:len(lines)+1])

    for line in frameRange:
        FOData[line, :] = np.asarray(tmp[line].split(',')).astype('float')

    # Find name of coord file ..........................................................................................
    currHeaderLine = 0
    coordFile = ''
    while lines[currHeaderLine][0] == '#':

        if lines[currHeaderLine].split(' ')[1] == 'Scene':
            coordFile = lines[currHeaderLine].split(' ')[-1].split('/')[-1]
            coordFile = coordFile[0:-3] + 'coords'
            break
        currHeaderLine += 1

    return header, FOData, numFrames, frameRange, calibParams, coordFile


def rZoneParamsFromLogFile(expDir, FODataFile):
    # Load or read in logfile data

    with open(expDir + FODataFile) as FOf:
        lines = FOf.read().splitlines()

    # Get rZone parameter from header
    currHeaderLine = 0

    while lines[currHeaderLine][0] == '#':
        if ('Reinforcement' in lines[currHeaderLine]):
            rZone_rInner = int(lines[currHeaderLine+1].split(' ')[-2])
            rZone_rOuter = int(lines[currHeaderLine+2].split(' ')[-2])
            rZone_max = int(lines[currHeaderLine+3].split(' ')[-3])
            rZone_gExp = int(lines[currHeaderLine+4].split(' ')[-2])

            return rZone_rInner, rZone_rOuter, rZone_max, rZone_gExp

        currHeaderLine += 1
