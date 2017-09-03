""" Script for analysing a single FlyOver trial (assumes use of FlyOver version 9.2 or higher) """

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
