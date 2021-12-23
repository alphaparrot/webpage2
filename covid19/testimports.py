
if __name__=="__main__":
   os.system(' echo "beginning imports">/home/adivp416/public_html/covid19/pythonlog.txt')
   import os 
   os.environ['OPENBLAS_NUM_THREADS'] = '2'
   import numpy as np
   from scipy.special import factorial,loggamma
   import matplotlib.pyplot as plt
   import csv
   from datetime import date, timedelta
   import warnings,traceback

   os.system("echo Initialized>>/home/adivp416/public_html/covid19/pythonlog.txt")
   x = np.arange(10)
   os.system("echo numpy>>/home/adivp416/public_html/covid19/pythonlog.txt")
