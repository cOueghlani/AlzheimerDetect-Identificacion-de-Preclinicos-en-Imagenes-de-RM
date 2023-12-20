import numpy as np
from CFSmethod.CFS import cfs
import pandas as pd

#Read data from a specify sheet in Excel --> [537 rows x 208 columns]
bd = pd.read_excel(r'/home/cristina/Documentos/TFG/ALL_neurocloud_estudio_ALDAPA.xlsx',
                   sheet_name = 'normalizedCorrelation'
                   )

# Remove the rows = 'NA' of the column C. Now --> [360 rows x 208 columns]
bd_sinFilasNa = bd[bd['CSF_Result'].notna()] 

#Remove the columns C, D, E, F. Now --> [360 rows x 204 columns]:
bd_sinLas4columnasSinNa = bd_sinFilasNa.drop(['CSF_Result', 'CSF_AB42_Cutoff_LUD/Sant_pau', 'CSF_TTAU_Cutoff_LUD/Sant_pau', 'CSF_PTAU_Cutoff_LUD/Sant_pau'],
                                            axis=1)


                                # .dtypes --> checking the data types using data_frame.dtypes method  

bd_sinLas4columnasSinNa = bd_sinLas4columnasSinNa.to_numpy()
bd_sinFilasNa.to_numpy()

x = bd_sinLas4columnasSinNa     # Todas las columnas de bd menos la columna C D E F
y = bd_sinFilasNa.iloc[:, 2]    # Sólo la columna C, sin las filas NA 

x = x.astype('float64')

features = cfs(x, y)            # La función devolverá cuáles de las columnas del primer parámetro están 
                                # más correladas con el segundo parámetro (la clase):

print(features)