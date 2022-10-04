# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:51:07 2022

@author: pmayaduque
"""
from src.classes import Solar, Eolic, Diesel, Battery
import pandas as pd
import requests
import json 
import numpy as np
import copy


def read_data(demand_filepath, 
              forecast_filepath,
              units_filepath,
              instance_filepath,
              fiscal_filepath,
              cost_filepath):
    
    forecast_df = pd.read_csv(forecast_filepath)
    demand_df = pd.read_csv(demand_filepath)
    try:
        generators_data =  requests.get(units_filepath)
        generators_data = json.loads(generators_data.text)
    except:
        f = open(units_filepath)
        generators_data = json.load(f)    
    generators = generators_data['generators']
    batteries = generators_data['batteries']
    
    try:
        instance_data =  requests.get(instance_filepath)
        instance_data = json.loads(instance_data.text)
    except:
        f = open(instance_filepath)
        instance_data = json.load(f) 

    try:
        fiscal_data =  requests.get(fiscal_filepath)
        fiscal_data = json.loads(fiscal_data.text)
    except:
        f = open(fiscal_filepath)
        fiscal_data = json.load(f) 

    try:
        cost_data =  requests.get(cost_filepath)
        cost_data = json.loads(cost_data.text)
    except:
        f = open(cost_filepath)
        cost_data = json.load(f) 
        
    return demand_df, forecast_df, generators, batteries, instance_data, fiscal_data, cost_data


def create_objects(generators, batteries, forecast_df, demand_df, instance_data):
    # Create generators and batteries
    generators_dict = {}
    for k in generators:
      if k['tec'] == 'S':
        obj_aux = Solar(*k.values())
        gt = irradiance_panel (forecast_df, instance_data)
        obj_aux.Get_INOCT(instance_data["caso"], instance_data["w"])
        obj_aux.Solargeneration( forecast_df['t_ambt'], gt, instance_data["G_stc"])
        obj_aux.Solarcost()
      elif k['tec'] == 'W':
        obj_aux = Eolic(*k.values())
        obj_aux.Windgeneration(forecast_df['Wt'],instance_data["h2"],instance_data["coef_hel"] )
        obj_aux.Windcost()
      elif k['tec'] == 'D':
        obj_aux = Diesel(*k.values())   
      generators_dict[k['id_gen']] = obj_aux
      
    batteries_dict = {}
    for l in batteries:
        obj_aux = Battery(*l.values())
        batteries_dict[l['id_bat']] = obj_aux
        batteries_dict[l['id_bat']].calculatesoc()
    return generators_dict, batteries_dict


def create_technologies(generators_dict, batteries_dict):
    # Create technologies dictionary
    technologies_dict = dict()
    for bat in batteries_dict.values(): 
      if not (bat.tec in technologies_dict.keys()):
        technologies_dict[bat.tec] = set()
        technologies_dict[bat.tec].add(bat.br)
      else:
        technologies_dict[bat.tec].add(bat.br)
    for gen in generators_dict.values(): 
      if not (gen.tec in technologies_dict.keys()):
        technologies_dict[gen.tec] = set()
        technologies_dict[gen.tec].add(gen.br)
      else:
        technologies_dict[gen.tec].add(gen.br)

    # Creates renewables dict
    renewables_dict = dict()
    for gen in generators_dict.values(): 
        if gen.tec == 'S' or gen.tec == 'W': #or gen.tec = 'H'
          if not (gen.tec in renewables_dict.keys()):
              renewables_dict[gen.tec] = set()
              renewables_dict[gen.tec].add(gen.br)
          else:
              renewables_dict[gen.tec].add(gen.br)
              
    return technologies_dict, renewables_dict
 

#calculate total cost for two stage approach
def calculate_sizingcost(generators_dict, batteries_dict, ir, years, delta, greed):
            expr = 0
            '''
            ref = 0
            
            for gen in generators_dict.values(): 
                if (gen.tec == 'D'):
                    ref = 1
            '''
            for gen in generators_dict.values(): 
                if (gen.tec != 'D'): 
                    #fiscal incentive if not diesel
                    expr += gen.cost_up * delta
                    #if (ref == 0):
                        #expr += gen.cost_up * greed
                else:
                    expr += gen.cost_up
                expr += gen.cost_r 
                expr -= gen.cost_s 
                expr += gen.cost_fopm 
                #expr2 += gen.cost_fopm  
            for bat in batteries_dict.values(): 
                expr += bat.cost_up * delta
                expr += bat.cost_r
                expr -= bat.cost_s
                expr += bat.cost_fopm
                #expr2 += gen.cost_fopm 
             
            CRF = (ir * (1 + ir)**(years))/((1 + ir)**(years)-1)    
            #Operative cost doesn't take into account the crf
            TNPCCRF = expr*CRF 
            #TNCCCRF = expr * (1+ir) + expr2 * ((((inf)**t_years)-1)/inf)
            return TNPCCRF


def calculate_area (sol_actual):
    solution = copy.deepcopy(sol_actual)
    dict_actual = {**solution.generators_dict_sol,**solution.batteries_dict_sol}
    area = 0
    for i in dict_actual.values():
        area += i.area 
    return area


#Calculate energy total, for eevery brand, technology or renewable 
def calculate_energy(batteries_dict, generators_dict, model_results, demand_df):  
   #create auxiliar sets
   column_data = {}
   energy_data = {}
   aux_energy_data = []
   renew_data = {}
   aux_renew_data = []
   total_data = [0]*len(demand_df)
   aux_total_data = []
   brand_data = {}
   aux_brand_data = []
   for bat in batteries_dict.values(): 
       #check that the battery is installed
       if (model_results.descriptive['batteries'][bat.id_bat] == 1):
           column_data[bat.id_bat+'_%'] =  model_results.df_results[bat.id_bat+'_b-'] / model_results.df_results['demand']
           aux_total_data = model_results.df_results[bat.id_bat+'_b-']
           #sum all generation
           total_data += aux_total_data
           #check the key for create or continue in the same dict
           key_energy_total = bat.tec+ 'total'
           key_brand_total = bat.br + 'total'
           if key_energy_total in energy_data:
               aux_energy_data = []
               aux_energy_data = energy_data[key_energy_total] +  model_results.df_results[bat.id_bat+'_b-']
               #fill the dictionary
               energy_data[key_energy_total] = aux_energy_data
           else:
               energy_data[key_energy_total] =  model_results.df_results[bat.id_bat+'_b-']           
    
           if key_brand_total in brand_data:
               aux_brand_data = []
               aux_brand_data = brand_data[key_brand_total] +  model_results.df_results[bat.id_bat+'_b-']
               #fill the dictionary
               brand_data[key_brand_total] = aux_brand_data
           else:
               brand_data[key_brand_total] =  model_results.df_results[bat.id_bat+'_b-']           
      
   for gen in generators_dict.values():
       #check that the generator is installed
       if (model_results.descriptive['generators'][gen.id_gen] == 1):
           column_data[gen.id_gen+'_%'] =  model_results.df_results[gen.id_gen] / model_results.df_results['demand']
           #check the key for create or continue in the same dict
           key_energy_total = gen.tec + 'total'
           key_renew_total = gen.tec + 'total'
           key_brand_total = gen.br + 'total'
           #sum all generation
           total_data += model_results.df_results[gen.id_gen]
           if key_energy_total in energy_data:
               aux_energy_data = []
               aux_energy_data = energy_data[key_energy_total] +  model_results.df_results[gen.id_gen]
               #fill the dictionary
               energy_data[key_energy_total] = aux_energy_data
           else:
               energy_data[key_energy_total] =  model_results.df_results[gen.id_gen]           
           
           if key_brand_total in brand_data:
               aux_brand_data = []
               aux_brand_data = brand_data[key_brand_total] +  model_results.df_results[gen.id_gen]
               #fill the dictionary
               brand_data[key_brand_total] = aux_brand_data
           else:
               brand_data[key_brand_total] =  model_results.df_results[gen.id_gen]           
           if (gen.tec == 'S' or gen.tec == 'W'):
               if key_renew_total in renew_data:
                   aux_renew_data = []
                   aux_renew_data = renew_data[key_renew_total] +  model_results.df_results[gen.id_gen]
                   #fill the dictionary
                   renew_data[key_renew_total] = aux_renew_data
               else:
                   renew_data[key_renew_total] =  model_results.df_results[gen.id_gen]           
   #Create dataframes
   percent_df = pd.DataFrame(column_data, columns=[*column_data.keys()])
   energy_df = pd.DataFrame(energy_data, columns=[*energy_data.keys()])
   renew_df = pd.DataFrame(renew_data, columns=[*renew_data.keys()])
   arraydf = np.array(total_data)
   total_df = pd.DataFrame(arraydf, columns=['Total energy'])
   brand_df = pd.DataFrame(brand_data, columns=[*brand_data.keys()])
   
   return percent_df, energy_df, renew_df, total_df, brand_df

def interest_rate (i_f, inf):
    #inf = inflation
    #i_f = nominal rate
    ir = (i_f - inf)/(1 + inf)
    return ir

def calculate_cost_data(generators, batteries, instance_data,
                        parameters_cost):
    #inflation
    inf = instance_data['inf']
    #nominal rate
    i_f= instance_data['i_f']
    years = instance_data['years']
    ir = interest_rate(i_f,inf)
    #defaul useful life Diesel and batteries = 10
    life_cicle = parameters_cost['life_cicle']
    ran = years/life_cicle
    
    #Calculate tax for remplacement
    tax = 0
    for h in range(1,int(ran)+1):
        tax += 1/((1+ir)**(h*life_cicle))
        
    aux_generators = []
    generators_def = []
    aux_batteries = []
    batteries_def = []
    generators_transformed = copy.deepcopy(generators)
    batteries_transformed = copy.deepcopy(batteries)
    total_transformed = generators_transformed + batteries_transformed
    #Calculate costs with investment cost
    for i in total_transformed:
        if (i['tec'] == 'S'):
            cost_up = i['cost_up']
            aux_generators = []
            aux_generators = i
            aux_generators['cost_r'] = parameters_cost['param_r_solar']
            aux_generators['cost_s'] = cost_up * parameters_cost['param_s_solar'] * (((1 + inf)/(1 + ir))**years)
            aux_generators['cost_fopm'] = cost_up * parameters_cost['param_f_solar'] 
            aux_generators['cost_vopm'] =  parameters_cost['param_v_solar']      
            generators_def.append(copy.deepcopy(aux_generators))
        elif (i['tec'] == 'W'):
            cost_up = i['cost_up']
            aux_generators = []
            aux_generators = i
            aux_generators['cost_r'] = parameters_cost['param_r_wind']  
            aux_generators['cost_s'] = cost_up * parameters_cost['param_s_wind']   * (((1 + inf)/(1 + ir))**years)
            aux_generators['cost_fopm'] =  cost_up * parameters_cost['param_f_wind']  
            aux_generators['cost_vopm'] =  parameters_cost['param_v_wind']  
            generators_def.append(copy.deepcopy(aux_generators))
        elif (i['tec'] == 'D'):
            cost_up = i['cost_up']
            aux_generators = []
            aux_generators = i
            aux_generators['cost_r'] = cost_up * parameters_cost['param_r_diesel']   * tax
            aux_generators['cost_s'] = cost_up * parameters_cost['param_s_diesel']   * (((1 + inf)/(1 + ir))**years)
            aux_generators['cost_fopm'] =  cost_up * parameters_cost['param_f_diesel']
            generators_def.append(copy.deepcopy(aux_generators)) 
        else:
            cost_up = i['cost_up']
            aux_batteries = []
            aux_batteries = i
            aux_batteries['cost_r'] = cost_up * parameters_cost['param_r_bat'] * tax
            aux_batteries['cost_s'] = cost_up * parameters_cost['param_s_bat'] * (((1 + inf)/(1 + ir))**years)
            aux_batteries['cost_fopm'] =  cost_up * parameters_cost['param_f_bat']
            batteries_def.append(copy.deepcopy(aux_batteries))

    return generators_def, batteries_def


def fiscal_incentive (credit, depreciation, corporate_tax, ir, T1, T2):
    #corporate_tax = effective corporate tax income rate
    #Credit = investment tax credit
    #Depreciation = depreciation factor expressed as percentage of investment cost over T2 year
    #ir = Interest rate
    #T1 = Maximum number of years to apply the investment tax credit
    #T2 = useful life of the power generating facility for accelerated depreciation purpose (in year)
    delta = 0
    expr = 0
    for j in range(1,int(T1) + 1):
        expr += credit/((1 + ir)**j)

    for j in range(1,int(T2) + 1):
        expr += depreciation/((1 + ir)**j)
    
    delta = (1/(1-corporate_tax))*(1-corporate_tax*expr)
    
    return delta


    
def irradiance_panel (forecast_df, instance_data):
    if (forecast_df['GHI'].sum() <= 0 or forecast_df['DHI'].sum() <= 0):
        #Default only DNI if it is not GHI or DHI
        gt_data = forecast_df['DNI']
    else:       
        theta_M = instance_data["tilted_angle"]
        a_M = 90 - theta_M
        A_M = instance_data["module_azimuth"]
        TZ = instance_data["time_zone"]
        long = instance_data["longitude"]
        latit = instance_data["latitude"]
        alpha = instance_data["alpha_albedo"]
        gt_data = {}
        for t in list(forecast_df['t'].index.values):
            LT = forecast_df['t'][t]
            SF1 = forecast_df['SF'][t] #Shadow factor
            DNI = forecast_df['DNI'][t] #Direct normal Irradiance
            DHI = forecast_df['DHI'][t] #Diffuse Horizontal Irradiance
            GHI = forecast_df['GHI'][t] #Global horizontal Irradiance
            day = forecast_df['day'][t] #Day of the year
            
            Gs = Get_SolarP01(LT,TZ,day,long,latit) #Sum altitude and sum Azimuth   
            ds = cos_AOI(a_M,A_M,Gs[0],Gs[1]) #COsine incidence angle
            svf = Get_SVF01(theta_M) #Sky view factor
            G_dr = SF1*DNI*ds
            if G_dr < 0:
                G_dr = 0 #negative Direct Irradiance on the PV module as zero
            G_df = svf * DHI #Diffuse irradiancia
            G_alb = alpha*(1 - svf)*GHI #Groud irradiance
            gt_data[t] = G_dr + G_df + G_alb #Total irradiance
    
    gt =  pd.DataFrame(gt_data.items(), columns = ['t','gt']) 
        
    return  gt



def min2hms(hm):
    """conversion min -> (hours, min, sec)
    
    """
    H = int(hm/60)
    M = ((hm/60)-H)*60
    m = int(M)
    S = (M-m)*60
    s = int(S)
    return H,m,s

def hd2hms(hd):
    """decimal time conversion ->
       (hours,minutes,seconds)
    """
    H = int(hd)
    m = (hd - H)*60
    M = int(m)
    s = m-M
    S = int(s*60)
    return H,M,S

def Get_SolarP01(LT,TZ,dia,Long,Latit):
    """LT: local time(hour)
       TZ: time zone
       dia: counted from January 1
       Long: longitude in degrees
       Latit: latitude in degrees
       sun position calculation
       return: Elevation,Azimuth in degrees
       version: 2019-02-05
       ref:https://www.pveducation.org/pvcdrom/2-properties-sunlight/suns-position
    """
    ka = 180/np.pi
    LSTM = 15*(TZ)#Local Standard Time Meridian(LSTM)
    EoT = lambda x:9.87*np.sin(2*x)-7.53*np.cos(x)-1.5*np.sin(x)#x in radians
    B = lambda d:((360/365)*d - 81)*(np.pi/180)
    LT1 = LT*60 #conversion to minutes
    TC = 4*(Long - LSTM) + EoT(B(dia))#Time Correction Factor (TC)
    LST = LT1 + (TC/60)#The Local Solar Time (LST)
    HRA = 15*((LST/60)-12)#Hour Angle (HRA)
    delta = 23.45*np.sin(B(dia))#declination angle (delta)
    Elevation = np.arcsin(np.sin(delta*np.pi/180)*np.sin(Latit*np.pi/180)+np.cos(delta*np.pi/180)*np.cos(Latit*np.pi/180)*np.cos(HRA*np.pi/180))
    
    ##calculate Azimuth 
    ## asumes teta:latitude
    k_num = np.sin(delta*np.pi/180)*np.cos(Latit*np.pi/180)+np.cos(delta*np.pi/180)*np.sin(Latit*np.pi/180)*np.cos(HRA*np.pi/180)
    k_total = k_num/np.cos(Elevation)

    if abs(k_total)>1.0:## vancouver essay
        k_total = k_total/abs(k_total)
    # print(k_total)

    Azimuth = np.arccos(k_total)
    if min2hms(LST)[0]>=12:#Correction after noon
        Azimuth = 2*np.pi - Azimuth

    return Elevation*ka, Azimuth*ka 
            #a_s: Sun altitude (grados)//Elevation
            #A_s: Sun Azimuth (grados)

def cos_AOI(a_M,A_M,a_s,A_s):
    """AOI: angle of incidence
        a_M: Module altitude (grados)
        A_M: Module Azimuth (grados)
        a_s: Sun altitude (grados)
        A_s: Sun Azimuth (grados)
        """
    Ar = A_M - A_s
    c1 = np.cos(a_M*np.pi/180)*np.cos(a_s*np.pi/180)*np.cos(Ar*np.pi/180)
    c2 = np.sin(a_M*np.pi/180)*np.sin(a_s*np.pi/180)
    ct = c1 + c2
    return ct



def Get_SVF01(t_M):
    """t_M: tilted angle of Module (grados)
        Free Horizont model
    """
    svf = (1 + np.cos(t_M*np.pi/180))/2
    return svf








