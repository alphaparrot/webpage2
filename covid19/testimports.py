import numpy as np
from scipy.special import factorial,loggamma
import matplotlib.pyplot as plt
import csv, os
from datetime import date, timedelta
import warnings,traceback

if __name__=="__main__":

   os.system("echo Initialized>pythonlog.txt")
   x = np.arange(10)
   os.system("echo numpy>>pythonlog.txt")
