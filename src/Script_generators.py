import json 
import requests
import copy

units_filepath = "../data/San_Andres/parameters_SA.json"

try:
    generators_data =  requests.get(units_filepath)
    generators_data = json.loads(generators_data.text)
except:
    f = open(units_filepath)
    generators_data = json.load(f)    
generators = generators_data['generators']
batteries = generators_data['batteries']

aux_generators = []
generators_def = []
aux_batteries = []
batteries_def = []
generators_transformed = copy.deepcopy(generators)
batteries_transformed = copy.deepcopy(batteries)
total_transformed = generators_transformed + batteries_transformed
for i in total_transformed:
    if (i['tec'] == 'S'):
        #max number of cells to be created
        max_cell = int(input(f"Max number of cells, {i['id_gen']}: "))
        #max number of cells to be created
        min_cell = int(input(f"Min number of cells, {i['id_gen']}: "))
        step_cell = int(input(f"Step of creation, {i['id_gen']}: "))
        name = i['id_gen']
        area = i['area']
        Ppv_stc = i['Ppv_stc']
        cost_up = i['cost_up']
        cost_r = i['cost_r']
        cost_s = i['cost_s']
        cost_fopm = i['cost_fopm']
        cost_vopm = i['cost_vopm']
        #create all generators with max, min and step
        for j in range(min_cell, max_cell + 1, step_cell):
            aux_generators = []
            aux_generators = i
            aux_generators['id_gen'] = name + ' n: ' + str(j)
            aux_generators['n'] = j
            aux_generators['area'] = j * area
            aux_generators['Ppv_stc'] = j * Ppv_stc
            aux_generators['cost_up'] = j * cost_up
            aux_generators['cost_r'] = j * cost_r
            aux_generators['cost_s'] = j * cost_s
            aux_generators['cost_fopm'] = j * cost_fopm
            aux_generators['cost_vopm'] = j * cost_vopm            
            generators_def.append(copy.deepcopy(aux_generators))
    elif (i['tec'] == 'W' or i['tec'] == 'D'):
        #Option of create equal generators
        num_gen = int(input(f"Number of equals generators, {i['id_gen']}: "))
        name = i['id_gen']
        for j in range(1, num_gen + 1):
            aux_generators = []
            aux_generators = i
            aux_generators['id_gen'] = name + ' ' + str(j)
            generators_def.append(copy.deepcopy(aux_generators))
    else:
        #Option of create equals batteries
        num_bat = int(input(f"Number of equals batteries, {i['id_bat']}: "))
        name = i['id_bat']
        for j in range(1, num_bat + 1):
            aux_batteries = []
            aux_batteries = i
            aux_batteries['id_bat'] = name + ' ' + str(j)
            batteries_def.append(copy.deepcopy(aux_batteries))
total_def = {}
total_def["generators"] = generators_def 
total_def["batteries"] = batteries_def
        
#create the json
with open('..\\data\\San_Andres\\json_example.json', 'w') as outfile:
    json.dump(total_def, outfile, indent=4)

