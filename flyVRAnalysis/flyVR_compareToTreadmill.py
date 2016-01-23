""" Script that loads a single FlyOver trial (assumes use of FlyOver version 9.2 or higher) and compares
 the VR movment against movement computed directly from the TM measurements. """

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'


from Tkinter import Tk
from tkFileDialog import askopenfilename

from flyVR_oneTrialBasicAnalysis import singleVRTrialAnalysis


baseDir = '/Volumes/jayaramanlab/Hannah/Projects/FlyVR/1_Experiments/'

root = Tk()
# prevents root window from appearing
root.withdraw()
# choose experiment
fileToAnalyse = askopenfilename(initialdir=baseDir, title='Select file for analysis')

returnVal = singleVRTrialAnalysis(fileToAnalyse)

# Comparison of flyVR vs. treadmill trajectory..........................................................................

dx1 = FOData[1:, 6]
dy1 = FOData[1:, 7]
dx2 = FOData[1:, 8]
dy2 = FOData[1:, 9]

# parameter definitions
gammaRad = 45*np.pi/180  # absolute angle of cameras to longitudinal axis (of fly)
conversionFactor_pitch = -1.76
conversionFactor_yaw = - 1.69
rBall = float(ballRadius_mm)
pixel2mm = 0.013514

# compute virtual rotation of fly
vFwdBall = - (dy1 + dy2) * np.cos(gammaRad)  # add components along longitudinal axis
vSideBall = - (dy1 - dy2) * np.sin(gammaRad)  # add components along transversal axis
vRotBall = - (dx1 + dx2)/2  # average measured displacement along aximuth

# convert A.U. --> pixel --> mm
vFwdBall = pixel2mm * vFwdBall * conversionFactor_pitch  # use scaling factor for pitch
vSideBall = pixel2mm * vSideBall * ((conversionFactor_yaw + conversionFactor_pitch)/2)  # use mean
vRotBall = pixel2mm * vRotBall * conversionFactor_yaw  # use scaling factor for yaw

# convert to mm/s
vFwdBall = vFwdBall / np.diff(time)
vSideBall = vSideBall / np.diff(time)
vRotBall = vRotBall / np.diff(time)

vRot_TM = - vRotBall / rBall  # mm/s to deg/s

# Assume initial position (0 0 0) = (x-coord, y-coord, theta):
# --> fly in origin, aligned with x axis (head forward)
# During measurement coordinate system is fly-centered, moves with fly.
# Compute all changes along those axes by updating theta and
# projecting the position changes onto the fixed coordinate system

theta = np.cumsum(vRot_TM * np.diff(time))
theta = np.mod((theta + np.pi), 2*np.pi) - np.pi

# movement in x and y direction
yTM_i = vSideBall * np.cos(-theta) - vFwdBall * np.sin(-theta)  # compute increments x_i
yTM = np.cumsum(yTM_i * np.diff(time))  # integrate x_i to get path

xTM_i = vSideBall * np.sin(-theta) + vFwdBall * np.cos(-theta)
xTM = np.cumsum(xTM_i * np.diff(time))

vTrans_TM = np.hypot(xTM_i, yTM_i)


VRTMplot = plt.figure(figsize=(20, 10))
ax = VRTMplot.add_subplot(111)
plt.plot(300 + xTM, yTM, 'k')
plt.plot(xPos_ds, yPos_ds)
ax.set_aspect('equal')

plt.plot(vTrans[0:60]*np.pi)
plt.plot(vTrans_TM[0:60], 'k.-')

plt.hist(vTrans_TM, bins=50)
plt.hist(vTrans, bins=50)

plt.plot(theta[1:100], 'k.')
plt.plot(angle[1:100])

plt.plot(vRot_TM[100:200], 'k.')
plt.plot(vRot[100:200])

# TODO: Save plot in approriate location.
