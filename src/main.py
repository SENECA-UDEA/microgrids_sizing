# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:14:21 2022

@author: pmayaduque
"""
from classes import Solar, Eolic, Diesel, Battery

import opt as opt
import pandas as pd 
import json
import requests
import numpy as np
import math

# file paths github
demand_filepath = 'https://raw.githubusercontent.com/pmayaduque/MicrogridSizing/main/data/Demanda%20Anual%20Leticia.csv' 
forecast_filepath = 'https://raw.githubusercontent.com/pmayaduque/MicrogridSizing/main/data/Forecast%20anual.csv' 
units_filepath  = 'https://raw.githubusercontent.com/pmayaduque/MicrogridSizing/main/data/parameters_P.json' 
# file paths local
demand_filepath = "../data/Demanda Anual Leticia.csv"
forecast_filepath = '../data/Forecast anual.csv'
units_filepath = "../data/parameters_P.json"


# read data 
forecast_df = pd.read_csv('../data/Forecast anual.csv')
demand_df = pd.read_csv(demand_filepath)
try:
    generators_data =  requests.get(units_filepath)
    generators_data = json.loads(generators_data.text)
except:
    f = open(units_filepath)
    generators_data = json.load(f)
generators = generators_data['generators']
batteries = generators_data['batteries']


# Create generators and batteries
generators_dict = {}
for k in generators:
  if k['tec'] == 'S':
    obj_aux = Solar(*k.values())
  elif k['tec'] == 'W':
    obj_aux = Eolic(*k.values())
  elif k['tec'] == 'D':
    obj_aux = Diesel(*k.values())
  
  generators_dict[k['id_gen']] = obj_aux

batteries_dict = {}
for l in batteries:
    obj_aux = Battery(*l.values())
    batteries_dict[l['id_bat']] = obj_aux
    
# Create technologies list
technologies_dict = dict()
for bat in batteries_dict.values(): 
  if not (bat.tec in technologies_dict.keys()):
    technologies_dict[bat.tec] = set()
    technologies_dict[bat.tec].add(bat.alt)
  else:
    technologies_dict[bat.tec].add(bat.alt)
for gen in generators_dict.values(): 
  if not (gen.tec in technologies_dict.keys()):
    technologies_dict[gen.tec] = set()
    technologies_dict[gen.tec].add(gen.alt)
  else:
    technologies_dict[gen.tec].add(gen.alt)

# Creates renewables dict
renewables_dict = dict()
for gen in generators_dict.values(): 
    if gen.tec == 'S' or gen.tec == 'W': #or gen.tec = 'H'
      if not (gen.tec in renewables_dict.keys()):
          renewables_dict[gen.tec] = set()
          renewables_dict[gen.tec].add(gen.alt)
      else:
          renewables_dict[gen.tec].add(gen.alt)
          
# TODO: this is not needed
max_demand =  max(demand_df.max())
max_rad = max(forecast_df['Rt'])
max_wind = max(forecast_df['Wt'])
l_min, l_max = 2, 2
g_max = 1.1 * max_demand
g_min = 0.3 * max_demand
nn = len(forecast_df)
demand_d = np.zeros(nn)
rad_p = np.zeros(nn)
wind_p = 0
wind_p = pd.DataFrame.mean(forecast_df['Wt'])
demand_d = np.zeros(nn)
rad_p = np.ones(nn)
for val in range(len(forecast_df)):
     demand_d[val] = demand_df['demand'].values[val]
     rad_p[val] = forecast_df['Rt'].values[val]
rad_s = sum(i for i in rad_p)
h_p = rad_s/1000 
l_min, l_max = 2, 2
g_max = 0.8*max_demand
Size = {}
for gen in generators_dict.values():
      if gen.tec == 'S':
         Size[gen.id_gen] = math.ceil((0.1*sum(i for i in demand_d))/(gen.G_test * h_p))
      elif gen.tec == 'W':
          f_plan = wind_p/gen.w_a
          Size[gen.id_gen] = math.ceil((0.1*sum(i for i in demand_d))/(nn*f_plan*20))
      elif gen.tec == 'D':
          Size[gen.id_gen] =  g_max  #para no calcular max_demand siempre

# Create model          
model = opt.make_model(generators_dict, 
                       forecast_df, 
                       batteries_dict, 
                       dict(zip(demand_df.t, demand_df.demand)),
                       technologies_dict, 
                       renewables_dict, 
                       20, 0.2, 0.8,4,3,2,20, Size)    