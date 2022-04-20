# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""

import pandas as pd 

# file paths
demand_filepath = 'https://drive.google.com/uc?id=1177EifUSKWVpc0eoO9oqJ8obFDA8QeGy&export=download' 
forecast_filepath = 'https://drive.google.com/uc?id=1cTxMU7D0ZMknSylGKP0VDb4g0GLpWUJX&export=download' 
units_filepath  = 'https://drive.google.com/uc?id=1i5VdIDAGoAeVTv8_6do7FhG9_dxggB74&export=download' 


df = pd.read_csv(r'https://docs.google.com/document/d/1177EifUSKWVpc0eoO9oqJ8obFDA8QeGy/export?format=csv')

