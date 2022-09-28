# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 18:10:16 2022

@author: sebas
"""


import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import copy


df = pd.read_excel('anova_time_w1to432_costfase2.xlsx')

df.columns = df.columns.str.replace(' ', '_')

my_list = list(df)

significant = ['TOTAL_TIME','ITERATIONS_MEAN_SOLVER_TIME','LCOE','MEAN_GENERATION_WIND',
               'MEAN_GENERATION_SOLAR', 'MEAN_GENERATION_DIESEL','MEAN_RENEWABLE']

significant2 = ['TOTAL_TIME','LCOE',
                'MEAN_GENERATION_WIND', 'MEAN_GENERATION_BATTERY',
               'MEAN_GENERATION_SOLAR', 'MEAN_GENERATION_DIESEL',
               'COST_VOPM','TNPC','LPSP_MEAN','UTILIZED_AREA','LEN_GENERATORS']


#boxplot = df.boxplot(column=['TOTAL_TIME'],by="gap", grid=False)
#boxplot = df.boxplot(column=['LCOE'],by="NSE")
#boxplot = df.boxplot(column=['LCOE'],by="Demand_percent")
#boxplot = df.boxplot(column=['LCOE'],by="W_cost")
#boxplot.plot()

#plt.show()
sns.boxplot(x='fuel_cost', y='lcoe', data=df, hue = 'NSE')
sns.boxplot(x='Item_cost', y='LCOE', data=df)


dfs1 = []
dfs2 = []
dfs3 = []
dfs4 = []
dfs5 = []
dfs6 = []

for i in significant:
    aux_df1 = []
    model1_aov_table = []
    name1 = i + " ~  C(Iterations, Sum) + C(Tlpsp, Sum) + C(Distribution, Sum) + C(gap, Sum) + C(Len_demand, Sum)  + C(add_function, Sum) + C(RAD, Sum)"
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=2)
    model1_aov_table[i] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs1.append(aux_df1)

for i in significant:
    aux_df1 = []
    model1_aov_table = []
    name1 = i + "~ C(Iterations, Sum) + C(Tlpsp, Sum) + C(Distribution, Sum) + C(gap, Sum) + C(Len_demand, Sum)  + C(add_function, Sum) + C(RAD, Sum)     + C(Iterations, Sum) * C(gap, Sum)+ C(Iterations, Sum) * C(Tlpsp, Sum) + C(Iterations, Sum) * C(Distribution, Sum)+ C(Len_demand, Sum) * C(RAD, Sum) + C(Len_demand, Sum)  * C(add_function, Sum)+ C(add_function, Sum) * C(RAD, Sum)+ C(Tlpsp, Sum) * C(gap, Sum)+ C(Tlpsp, Sum) * C(Len_demand, Sum) + C(Tlpsp, Sum) * C(add_function, Sum)+ C(Distribution, Sum) * C(Len_demand, Sum)"    
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=2)
    model1_aov_table[i] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs2.append(aux_df1)


for i in significant:
    aux_df1 = []
    model1_aov_table = []
    name1 = i + "~ C(Iterations, Sum) + C(Tlpsp, Sum) + C(Distribution, Sum) + C(gap, Sum) + C(Len_demand, Sum)  + C(add_function, Sum) + C(RAD, Sum)     + C(Iterations, Sum) * C(gap, Sum)+ C(Iterations, Sum) * C(Tlpsp, Sum) + C(Iterations, Sum) * C(Distribution, Sum)+ C(Len_demand, Sum) * C(RAD, Sum) + C(Len_demand, Sum)  * C(add_function, Sum)+ C(add_function, Sum) * C(RAD, Sum)+ C(Tlpsp, Sum) * C(gap, Sum)+ C(Tlpsp, Sum) * C(Len_demand, Sum) + C(Tlpsp, Sum) * C(add_function, Sum)+ C(Distribution, Sum) * C(Len_demand, Sum)"    
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=3)
    model1_aov_table[i] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs5.append(aux_df1)


for i in range (31,57):
    aux_df1 = []
    model1_aov_table = []
    name1 = my_list[i] + " ~ C(Iterations, Sum) + C(Tlpsp, Sum) + C(Distribution, Sum) + C(gap, Sum) + C(Len_demand, Sum)  + C(add_function, Sum) + C(RAD, Sum)"
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=2)
    model1_aov_table[my_list[i]] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs3.append(aux_df1)

for i in range (31,57):
    aux_df1 = []
    model1_aov_table = []
    name1 = my_list[i] + " ~ C(Iterations, Sum) + C(Tlpsp, Sum) + C(Distribution, Sum) + C(gap, Sum) + C(Len_demand, Sum)  + C(add_function, Sum) + C(RAD, Sum)     + C(Iterations, Sum) * C(gap, Sum)+ C(Iterations, Sum) * C(Tlpsp, Sum) + C(Iterations, Sum) * C(Distribution, Sum)+ C(Len_demand, Sum) * C(RAD, Sum) + C(Len_demand, Sum)  * C(add_function, Sum)+ C(add_function, Sum) * C(RAD, Sum)+ C(Tlpsp, Sum) * C(gap, Sum)+ C(Tlpsp, Sum) * C(Len_demand, Sum) + C(Tlpsp, Sum) * C(add_function, Sum)+ C(Distribution, Sum) * C(Len_demand, Sum)"    
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=2)
    model1_aov_table[my_list[i]] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs4.append(aux_df1)


for i in range (31,57):
    aux_df1 = []
    model1_aov_table = []
    name1 = my_list[i] + " ~ C(Iterations, Sum) + C(Tlpsp, Sum) + C(Distribution, Sum) + C(gap, Sum) + C(Len_demand, Sum)  + C(add_function, Sum) + C(RAD, Sum)     + C(Iterations, Sum) * C(gap, Sum)+ C(Iterations, Sum) * C(Tlpsp, Sum) + C(Iterations, Sum) * C(Distribution, Sum)+ C(Len_demand, Sum) * C(RAD, Sum) + C(Len_demand, Sum)  * C(add_function, Sum)+ C(add_function, Sum) * C(RAD, Sum)+ C(Tlpsp, Sum) * C(gap, Sum)+ C(Tlpsp, Sum) * C(Len_demand, Sum) + C(Tlpsp, Sum) * C(add_function, Sum)+ C(Distribution, Sum) * C(Len_demand, Sum)"    
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=2)
    model1_aov_table[my_list[i]] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs6.append(aux_df1)



    
#crear Excel
def multiple_dfs(df_list, sheets, file_name):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')   
    col = 0
    for dataframe in df_list:
        dataframe.to_excel(writer,sheet_name=sheets,startrow=0 , startcol=col)   
        col = col + 7 
    writer.save()


# run function
multiple_dfs(dfs1, 'ExecTime', '1factor.xlsx')
multiple_dfs(dfs2, 'ExecTime', '2factores.xlsx')
multiple_dfs(dfs3, 'ExecTime', '1factorall.xlsx')
multiple_dfs(dfs4, 'ExecTime', '2factorall.xlsx')
multiple_dfs(dfs5, 'ExecTime', '2factorest3.xlsx')
multiple_dfs(dfs6, 'ExecTime', '2factoresallt3.xlsx')





'''
for i in significant2:
    aux_df1 = []
    model1_aov_table = []
    name1 = i + " ~ C(fuel_cost, Sum) + C(W_cost, Sum) + C(Area, Sum) + C(delta, Sum) + C(NSE, Sum)  + C(Demand_percent, Sum) + C(Item_cost, Sum)"
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=2)
    model1_aov_table[i] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs1.append(aux_df1)

for i in significant2:
    aux_df1 = []
    model1_aov_table = []
    name1 = i + " ~ C(fuel_cost, Sum) + C(W_cost, Sum) + C(Area, Sum) + C(delta, Sum) + C(NSE, Sum)  + C(Demand_percent, Sum) + C(Item_cost, Sum) + C(fuel_cost, Sum) * C(W_cost, Sum) + C(fuel_cost, Sum) *C(delta, Sum) + C(fuel_cost, Sum) * C(NSE, Sum) + C(fuel_cost, Sum) * C(Demand_percent, Sum)  + C(fuel_cost, Sum) * C(Item_cost, Sum) + C(W_cost, Sum) * C(delta, Sum) + C(W_cost, Sum) * C(NSE, Sum) + C(W_cost, Sum) * C(Demand_percent, Sum)  + C(W_cost, Sum)  * C(Item_cost, Sum) + C(Area, Sum) * C(delta, Sum) + C(delta, Sum) * C(Demand_percent, Sum)  + C(NSE, Sum) * C(Demand_percent, Sum) "    
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=2)
    model1_aov_table[i] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs2.append(aux_df1)

for i in range (30,56):
    aux_df1 = []
    model1_aov_table = []
    name1 = my_list[i] + " ~ C(fuel_cost, Sum) + C(W_cost, Sum) + C(Area, Sum) + C(delta, Sum) + C(NSE, Sum)  + C(Demand_percent, Sum) + C(Item_cost, Sum)"
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=3)
    model1_aov_table[my_list[i]] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs3.append(aux_df1)

for i in range (30,56):
    aux_df1 = []
    model1_aov_table = []
    name1 = my_list[i] + " ~ C(fuel_cost, Sum) + C(W_cost, Sum) + C(Area, Sum) + C(delta, Sum) + C(NSE, Sum)  + C(Demand_percent, Sum) + C(Item_cost, Sum) + C(fuel_cost, Sum) * C(W_cost, Sum) + C(fuel_cost, Sum) *C(delta, Sum) + C(fuel_cost, Sum) * C(NSE, Sum) + C(fuel_cost, Sum) * C(Demand_percent, Sum)  + C(fuel_cost, Sum) * C(Item_cost, Sum) + C(W_cost, Sum) * C(delta, Sum) + C(W_cost, Sum) * C(NSE, Sum) + C(W_cost, Sum) * C(Demand_percent, Sum)  + C(W_cost, Sum)  * C(Item_cost, Sum) + C(Area, Sum) * C(delta, Sum) + C(delta, Sum) * C(Demand_percent, Sum)  + C(NSE, Sum) * C(Demand_percent, Sum) "    
    model1 = ols (name1, data=df).fit()
    model1_aov_table = sm.stats.anova_lm(model1, typ=3)
    model1_aov_table[my_list[i]] = ''
    aux_df1 = copy.deepcopy(model1_aov_table)
    dfs4.append(aux_df1)



multiple_dfs(dfs1, 'ExecTime', 'ANOVA1factorFASE2.xlsx')
multiple_dfs(dfs2, 'ExecTime', 'ANOVA2factoresFASE2.xlsx')

'''