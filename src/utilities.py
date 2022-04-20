# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:51:07 2022

@author: pmayaduque
"""

def generation(gen, t, forecast_df, Size):

      if gen.tec == 'S':
         g_rule = Size * gen.ef * gen.G_test * (forecast_df['Rt'][t]/gen.R_test) 
         #falta considerar área del panel
      elif gen.tec == 'W':
          if forecast_df['Wt'][t] < gen.w_min:
              g_rule = 0
          elif forecast_df['Wt'][t] < gen.w_a:
              g_rule =  (Size * (1/2) * gen.p * gen.s * (forecast_df['Wt'][t]**3) * gen.ef * gen.n )/1000
              #p en otros papers es densidad del aíre, preguntar a Mateo, por qué divide por 1000?
          elif forecast_df['Wt'][t] <= gen.w_max:
              g_rule = (Size * (1/2) * gen.p * gen.s * (gen.w_a**3) * gen.ef * gen.n )/1000
          else:
              g_rule = 0
      elif gen.tec == 'D':
         g_rule =  Size
      return g_rule