from Heat_Sink_calculation import heat_sink_design
import numpy as np
import pandas as pd

data=[]

tdp_range= np.arange(30,251,10)
v_air_range=np.arange(0.5,15.5,0.5)
k_tim_range=np.arange(1,13,1)

for tdp in tdp_range:
    for v in v_air_range:
        for k in k_tim_range:
            R_total,T_j= heat_sink_design(tdp,v,k)

            data.append([
                tdp,v,k,R_total,T_j
            ])
df=pd.DataFrame(data,columns=[
    "TDP","air_velocity","tim_conductivity","thermal_resistance","junction_temperature"
])

df.to_csv("thermal_dataset.csv", index=False)

print(df.head())
print("total rows=",len(df))