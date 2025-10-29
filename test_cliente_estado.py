#!/usr/bin/env python3
"""
Script para diagnosticar el problema con el estado de clientes activos/inactivos
"""

import requests
import pandas as pd
from datetime import datetime
import json

# Configuración
ENDPOINT_PEDIDOS = "https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def parse_fecha(fecha_str):
    """Convierte fecha del formato DD-MM-YYYY a datetime"""
    try:
        return datetime.strptime(fecha_str, "%d-%m-%Y")
    except:
        return None

def calcular_estado_cliente(fecha_ultimo):
    """Calcula si un cliente está activo o inactivo basado en su último pedido"""
    if not fecha_ultimo or fecha_ultimo.strip() == '':
        return 'Inactivo'
    
    fecha = parse_fecha(fecha_ultimo)
    if not fecha:
        return 'Inactivo'
    
    hoy = datetime.now()
    diff_dias = (hoy - fecha).days
    
    return 'Activo' if diff_dias <= 75 else 'Inactivo'

def main():
    print("=== DIAGNÓSTICO DE ESTADO DE CLIENTES ===\n")
    
    # 1. Obtener datos de pedidos
    try:
        print("1. Obteniendo datos de pedidos...")
        response = requests.get(ENDPOINT_PEDIDOS, headers=HEADERS, timeout=10)
        response.raise_for_status()
        pedidos = response.json()
        print(f"   ✓ Pedidos obtenidos: {len(pedidos)} registros")
    except Exception as e:
        print(f"   ✗ Error obteniendo pedidos: {e}")
        return
    
    # 2. Filtrar solo Aguas Ancud
    df = pd.DataFrame(pedidos)
    if 'nombrelocal' in df.columns:
        df_ancud = df[df['nombrelocal'].str.strip().str.lower() == 'aguas ancud']
        print(f"   ✓ Pedidos Aguas Ancud: {len(df_ancud)} registros")
    else:
        print("   ✗ No se encontró columna 'nombrelocal'")
        return
    
    if df_ancud.empty:
        print("   ✗ No hay pedidos de Aguas Ancud")
        return
    
    # 3. Buscar específicamente a Walker Martinez 264
    print("\n2. Buscando a Walker Martinez 264...")
    walker_pedidos = df_ancud[df_ancud['usuario'].str.contains('walker martinez 264', case=False, na=False)]
    
    if not walker_pedidos.empty:
        print(f"   ✓ Encontrados {len(walker_pedidos)} pedidos de Walker Martinez 264")
        
        # Mostrar todos los pedidos de Walker
        for idx, pedido in walker_pedidos.iterrows():
            fecha_pedido = pedido.get('fecha', 'N/A')
            fecha_parsed = parse_fecha(fecha_pedido)
            estado = calcular_estado_cliente(fecha_pedido)
            
            print(f"   - Pedido {idx}: {fecha_pedido} -> Estado: {estado}")
            if fecha_parsed:
                dias_desde = (datetime.now() - fecha_parsed).days
                print(f"     Días desde el pedido: {dias_desde}")
    else:
        print("   ✗ No se encontraron pedidos de Walker Martinez 264")
    
    # 4. Analizar todos los clientes
    print("\n3. Analizando estado de todos los clientes...")
    
    # Obtener último pedido por cliente
    df_ancud['fecha_dt'] = pd.to_datetime(df_ancud['fecha'], format='%d-%m-%Y', errors='coerce')
    df_ancud = df_ancud.sort_values('fecha_dt', ascending=False)
    clientes_unicos = df_ancud.drop_duplicates('usuario', keep='first')
    
    print(f"   ✓ Total de clientes únicos: {len(clientes_unicos)}")
    
    # Calcular estados
    clientes_unicos['estado'] = clientes_unicos['fecha'].apply(calcular_estado_cliente)
    
    activos = clientes_unicos[clientes_unicos['estado'] == 'Activo']
    inactivos = clientes_unicos[clientes_unicos['estado'] == 'Inactivo']
    
    print(f"   - Clientes Activos: {len(activos)}")
    print(f"   - Clientes Inactivos: {len(inactivos)}")
    
    # 5. Mostrar clientes activos recientes (últimos 7 días)
    print("\n4. Clientes con pedidos en los últimos 7 días:")
    hoy = datetime.now()
    clientes_recientes = clientes_unicos[clientes_unicos['fecha_dt'].notna()]
    clientes_recientes['dias_desde'] = (hoy - clientes_recientes['fecha_dt']).dt.days
    
    ultimos_7_dias = clientes_recientes[clientes_recientes['dias_desde'] <= 7]
    
    if not ultimos_7_dias.empty:
        for _, cliente in ultimos_7_dias.iterrows():
            print(f"   - {cliente['usuario']}: {cliente['fecha']} (hace {cliente['dias_desde']} días)")
    else:
        print("   - No hay clientes con pedidos en los últimos 7 días")
    
    # 6. Verificar fechas problemáticas
    print("\n5. Verificando fechas problemáticas...")
    fechas_invalidas = clientes_unicos[clientes_unicos['fecha_dt'].isna()]
    
    if not fechas_invalidas.empty:
        print(f"   ⚠️  {len(fechas_invalidas)} clientes con fechas inválidas:")
        for _, cliente in fechas_invalidas.head(5).iterrows():
            print(f"      - {cliente['usuario']}: '{cliente['fecha']}'")
    else:
        print("   ✓ Todas las fechas son válidas")
    
    # 7. Resumen del problema
    print("\n=== RESUMEN DEL PROBLEMA ===")
    print("El sistema considera un cliente como:")
    print("- ACTIVO: Si su último pedido fue hace 75 días o menos")
    print("- INACTIVO: Si su último pedido fue hace más de 75 días")
    print("\nSi Walker Martinez 264 realizó un pedido hoy pero aparece como inactivo,")
    print("posibles causas:")
    print("1. La fecha del pedido no se está parseando correctamente")
    print("2. El pedido no está siendo reconocido como de Aguas Ancud")
    print("3. Hay un problema en la lógica de cálculo de días")
    print("4. El pedido está siendo filtrado por alguna razón")

if __name__ == "__main__":
    main() 