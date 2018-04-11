"""
Functions for reading in object coordinates
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np


def loadObjectCoords(dataDir, coordFile):
    # Read coord file and extract coordinates of visible objects
    with open(dataDir + 'rawData/' + coordFile) as Coof:
        lines = Coof.read().splitlines()

    invisibleObjectCoords = []
    visibleObjectCoords = []
    origin = []

    for line in range(len(lines)):
        currObject = lines[line].split(' ')

        if len(currObject) < 3:
            continue

        if currObject[0][0] == '_':
            if currObject[0] not in ['_start_', '_camera_block_pm_', '_cylinder_pic_', '_Plane_p_']:
                invisibleObjectCoords.append(np.asarray((float(currObject[2]), float(currObject[3]))))

        elif currObject[0] == 'Origin':
            origin = [float(currObject[2]), float(currObject[3])]
        else:
            visibleObjectCoords.append(np.asarray((float(currObject[2]), float(currObject[3]))))

    invisibleObjectCoords = np.asarray(invisibleObjectCoords)
    visibleObjectCoords = np.asarray(visibleObjectCoords)

    return visibleObjectCoords, invisibleObjectCoords, origin


def loadObjectCoordIdentities(dataDir, coordFile):
    # Read coord file and extract coordinates of visible objects
    with open(dataDir + 'rawData/' + coordFile) as Coof:
        lines = Coof.read().splitlines()

    invisibleObjectCoords = []
    invisibleObjectName = []
    visibleObjectCoords = []
    visibleObjectName = []
    origin = []

    for line in range(len(lines)):
        currObject = lines[line].split(' ')

        if len(currObject) < 3:
            continue

        if currObject[0][0] == '_':
            if currObject[0] not in ['_start_', '_camera_block_pm_', '_cylinder_pic_', '_Plane_p_']:
                invisibleObjectCoords.append(np.asarray((float(currObject[2]), float(currObject[3]))))
                invisibleObjectName.append(currObject[0])

        elif currObject[0] == 'Origin':
            origin = [float(currObject[2]), float(currObject[3])]
        else:
            visibleObjectCoords.append(np.asarray((float(currObject[2]), float(currObject[3]))))
            visibleObjectName.append(currObject[0])

    invisibleObjectCoords = np.asarray(invisibleObjectCoords)
    visibleObjectCoords = np.asarray(visibleObjectCoords)

    return visibleObjectCoords, visibleObjectName, invisibleObjectCoords, origin

def loadObjectCoordAllIdentities(dataDir, coordFile):
    # Read coord file and extract coordinates of visible objects
    with open(dataDir + 'rawData/' + coordFile) as Coof:
        lines = Coof.read().splitlines()
    
    invisibleObjectCoords = []
    invisibleObjectName = []
    visibleObjectCoords = []
    visibleObjectName = []
    origin = []

    for line in range(len(lines)):
        currObject = lines[line].split(' ')
        
        if len(currObject) < 3:
            continue
        
        if currObject[0][0] == '_':
            if currObject[0] not in ['_start_', '_camera_block_pm_', '_cylinder_pic_', '_Plane_p_']:
                invisibleObjectCoords.append(np.asarray((float(currObject[2]), float(currObject[3]))))
                invisibleObjectName.append(currObject[0])
    
        elif currObject[0] == 'Origin':
            origin = [float(currObject[2]), float(currObject[3])]
        else:
            visibleObjectCoords.append(np.asarray((float(currObject[2]), float(currObject[3]))))
            visibleObjectName.append(currObject[0])

    invisibleObjectCoords = np.asarray(invisibleObjectCoords)
    visibleObjectCoords = np.asarray(visibleObjectCoords)

    return visibleObjectCoords, visibleObjectName, invisibleObjectCoords, invisibleObjectName, origin
