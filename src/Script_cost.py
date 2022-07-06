# -*- coding: utf-8 -*-

import json 
import requests
import copy
instanceData_filepath = "../data/San_Andres/instance_data_SA.json"
units_filepath = "../data/San_Andres/parameters_SA.json"


try:
    instance_data =  requests.get(instanceData_filepath)
    instance_data = json.loads(instance_data.text)
except:
    f = open(instanceData_filepath)
    instance_data = json.load(f)
try:
    generators_data =  requests.get(units_filepath)
    generators_data = json.loads(generators_data.text)
except:
    f = open(units_filepath)
    generators_data = json.load(f)    
generators = generators_data['generators']
batteries = generators_data['batteries']
ir = (instance_data['i_f'] - instance_data['inf'])/(1 + instance_data['inf'])
inf = instance_data['inf']
years = instance_data['years']
life_cicle = 10
ran = years/life_cicle
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
for i in total_transformed:
    if (i['tec'] == 'S'):
        name = i['id_gen']
        cost_up = i['cost_up']
        cost_r = i['cost_r']
        cost_s = i['cost_s']
        cost_fopm = i['cost_fopm']
        cost_vopm = i['cost_vopm']
        aux_generators = []
        aux_generators = i
        aux_generators['id_gen'] = name
        aux_generators['cost_up'] = cost_up
        aux_generators['cost_r'] = 0
        aux_generators['cost_s'] = cost_up * 0.2 * (((1 + inf)/(1 + ir))**years)
        aux_generators['cost_fopm'] = cost_up * 0.01
        aux_generators['cost_vopm'] =   0     
        generators_def.append(copy.deepcopy(aux_generators))
    elif (i['tec'] == 'W'):
        cost_up = i['cost_up']
        name = i['id_gen']
        aux_generators = []
        aux_generators = i
        aux_generators['id_gen'] = name
        aux_generators['cost_up'] = cost_up
        aux_generators['cost_r'] = 0
        aux_generators['cost_s'] = cost_up * 0.1 * (((1 + inf)/(1 + ir))**years)
        aux_generators['cost_fopm'] =  cost_up * 0.01
        aux_generators['cost_vopm'] =  0  
        generators_def.append(copy.deepcopy(aux_generators))
    elif (i['tec'] == 'D'):
        cost_up = i['cost_up']
        name = i['id_gen']
        aux_generators = []
        aux_generators = i
        aux_generators['id_gen'] = name
        aux_generators['cost_up'] = cost_up
        aux_generators['cost_r'] = cost_up * 0.7 * tax
        aux_generators['cost_s'] = cost_up * 0.3 * (((1 + inf)/(1 + ir))**years)
        aux_generators['cost_fopm'] =  cost_up * 0.1
        generators_def.append(copy.deepcopy(aux_generators)) 
    else:
        cost_up = i['cost_up']
        name = i['id_bat']
        aux_batteries = []
        aux_batteries = i
        aux_batteries['id_bat'] = name
        aux_batteries['cost_up'] = cost_up
        aux_batteries['cost_r'] = cost_up * 0.7 * tax
        aux_batteries['cost_s'] = cost_up * 0.3 * (((1 + inf)/(1 + ir))**years)
        aux_batteries['cost_fopm'] =  cost_up * 0.02
        batteries_def.append(copy.deepcopy(aux_batteries))

total_def = {}
total_def["generators"] = generators_def 
total_def["batteries"] = batteries_def
        
#create the json
with open('..\\data\\San_Andres\\json_cost.json', 'w') as outfile:
    json.dump(total_def, outfile, indent=4)

