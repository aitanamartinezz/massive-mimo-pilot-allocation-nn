from main import tau_p, N, K, L, R, system_NMSE_rand, system_NMSE_opt
from prueba import predicciones_asignaciones_opt
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import numpy as np


for i in range(len(predicciones_asignaciones_opt)):
   print(f'Prediction for index {i}: {predicciones_asignaciones_opt[i]}')
   nmse_pred = functionComputeNMSE_uplink(tau_p, N, K, L, R, predicciones_asignaciones_opt[i])
   print (nmse_pred)
   if i == len(predicciones_asignaciones_opt) - 1:
      nmse_pred_calculado = np.mean(nmse_pred[0])
      print(f'Average NMSE for pred assignments (from data): {nmse_pred_calculado}')



#rand = np.mean(system_NMSE_rand)
#opt = np.mean(system_NMSE_opt)
#print(f'Average NMSE for rand assignments (from data): {rand}')
#print(f'Average NMSE for optimal assignments (from data): {opt}')