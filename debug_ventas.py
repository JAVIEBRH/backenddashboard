import requests
import json

try:
    print("Probando endpoint de ventas diarias...")
    response = requests.get('http://localhost:8001/ventas-diarias')
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"Response Text: {response.text[:500]}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Ventas hoy: ${data.get('ventas_hoy', 0):,}")
        print(f"Ventas mes anterior: ${data.get('ventas_mismo_dia_mes_anterior', 0):,}")
        print(f"Porcentaje cambio: {data.get('porcentaje_cambio', 0)}%")
    else:
        print("Error en el endpoint")
        
except Exception as e:
    print(f"Error: {e}") 