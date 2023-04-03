import json 
import requests
import copy

'''
This code is used to create multiple copies of generator and battery data,
with specific properties modified based on user input, 
and write them to a new json file.
 
STEPS
----- 
This code  uses the requests module to get the data from the file path, 
loads it into the items_data variable, and then loads it into the json format.
The code then separates the generators and batteries into their own variables.

The code iterates through the total_transformed variable, which 
is a combination of generators and batteries. If the technology key of an item 
in total_transformed equals 'S', the code prompts the user for a max number
 of cells, a min number of cells, and a step of creation for that generator.
 It then creates new generators, with the parameters keys modified based
 on the user input, and appends them to the generators_def list.

If the 'tec' key equals 'W' or 'D', the code prompts the user for a number 
of equal generators, and creates that number of generators with the id_gen 
key modified based on the user input, and appends them to the generators_def 
list.

The code prompts the user for a number of equal batteries, creates that number
 of batteries with the id_bat key modified based on the user input, 
 and appends them to the batteries_def list.

Finally, the code creates a new json file and writes the generators_def
and batteries_def lists to it

'''
#set the origin of data
units_filepath = "../../data/Leticia/parameters_Leticia.json"

#load the data
try:
    items_data =  requests.get(units_filepath)
    items_data = json.loads(items_data.text)
except:
    f = open(units_filepath)
    items_data = json.load(f)    
generators = items_data['generators']
batteries = items_data['batteries']

aux_generators = []
generators_def = [] #definitive list of generators
aux_batteries = []
batteries_def = [] #definitive list of batteries
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
with open('../../data/json_generated.json',
          'w') as outfile:
    
    json.dump(total_def, outfile, indent = 4)

