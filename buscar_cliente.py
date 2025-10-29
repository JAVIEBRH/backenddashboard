#!/usr/bin/env python3
"""
Script para buscar clientes con nombres similares a Walker Martinez
"""

import requests
import pandas as pd
from datetime import datetime
import re

# Configuración
ENDPOINT_PEDIDOS = "https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def buscar_cliente_similar(nombre_busqueda):
    """Busca clientes con nombres similares"""
    try:
        response = requests.get(ENDPOINT_PEDIDOS, headers=HEADERS, timeout=10)
        response.raise_for_status()
        pedidos = response.json()
    except Exception as e:
        print(f"Error obteniendo pedidos: {e}")
        return
    
    df = pd.DataFrame(pedidos)
    
    # Buscar en todos los locales, no solo Aguas Ancud
    print(f"Buscando clientes similares a: '{nombre_busqueda}'")
    print(f"Total de pedidos en todos los locales: {len(df)}")
    
    # Buscar por diferentes variaciones del nombre
    nombre_lower = nombre_busqueda.lower()
    palabras_busqueda = nombre_lower.split()
    
    # Buscar clientes que contengan alguna de las palabras
    clientes_encontrados = set()
    
    for _, pedido in df.iterrows():
        usuario = str(pedido.get('usuario', '')).lower()
        
        # Verificar si contiene alguna palabra del nombre buscado
        if any(palabra in usuario for palabra in palabras_busqueda):
            clientes_encontrados.add(pedido['usuario'])
    
    print(f"\nClientes encontrados con palabras similares:")
    for cliente in sorted(clientes_encontrados):
        print(f"  - {cliente}")
    
    # Mostrar detalles de los clientes encontrados
    if clientes_encontrados:
        print(f"\nDetalles de los clientes encontrados:")
        for cliente_nombre in sorted(clientes_encontrados):
            pedidos_cliente = df[df['usuario'] == cliente_nombre]
            print(f"\n  Cliente: {cliente_nombre}")
            print(f"  Total de pedidos: {len(pedidos_cliente)}")
            
            # Mostrar últimos 3 pedidos
            pedidos_cliente['fecha_dt'] = pd.to_datetime(pedidos_cliente['fecha'], format='%d-%m-%Y', errors='coerce')
            pedidos_cliente = pedidos_cliente.sort_values('fecha_dt', ascending=False)
            
            for idx, pedido in pedidos_cliente.head(3).iterrows():
                fecha = pedido.get('fecha', 'N/A')
                local = pedido.get('nombrelocal', 'N/A')
                precio = pedido.get('precio', 0)
                print(f"    - {fecha} | {local} | ${precio}")
    
    # Buscar específicamente por "walker" y "martinez"
    print(f"\nBuscando específicamente por 'walker' y 'martinez':")
    walker_martinez = df[df['usuario'].str.contains('walker|martinez', case=False, na=False)]
    
    if not walker_martinez.empty:
        print(f"Encontrados {len(walker_martinez)} pedidos con 'walker' o 'martinez':")
        for _, pedido in walker_martinez.iterrows():
            print(f"  - {pedido['usuario']} | {pedido['fecha']} | {pedido['nombrelocal']}")
    else:
        print("No se encontraron pedidos con 'walker' o 'martinez'")

def main():
    print("=== BÚSQUEDA DE CLIENTE WALKER MARTINEZ ===\n")
    
    # Buscar por diferentes variaciones del nombre
    nombres_busqueda = [
        "Walker Martinez 264",
        "Walker Martinez",
        "walker martinez",
        "walker",
        "martinez"
    ]
    
    for nombre in nombres_busqueda:
        buscar_cliente_similar(nombre)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 