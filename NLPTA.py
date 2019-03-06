
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dfply import *
import os
#%%
path = 'D:\lecture\NLPTA\project\FOMCtxt'
os.chdir(r'D:\lecture\NLPTA\project\FOMCtxt')
file = open("1982FOMC19820202meeting.pdf.txt")
file_contents = file.read()
print(file_contents)
file_contents

