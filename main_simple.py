from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
from typing import List, Dict
from datetime import datetime
import json

app = FastAPI(title="API Aguas Ancud Simple", version="1.0")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENDPOINT_PEDIDOS = "https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def parse_fecha(fecha_str):
    """Convierte fecha de múltiples formatos a datetime"""
    try:
        if isinstance(fecha_str, str) and fecha_str.strip():
            fecha_str = fecha_str.strip()
            
            # Intentar diferentes formatos
            formatos = [
                "%d-%m-%Y",  # DD-MM-YYYY
                "%Y-%m-%d",  # YYYY-MM-DD
                "%d/%m/%Y",  # DD/MM/YYYY
                "%Y/%m/%d",  # YYYY/MM/DD
                "%d-%m-%y",  # DD-MM-YY
                "%y-%m-%d",  # YY-MM-DD
            ]
            
            for formato in formatos:
                try:
                    return datetime.strptime(fecha_str, formato)
                except ValueError:
                    continue
            
            print(f"⚠️ No se pudo parsear fecha: '{fecha_str}'")
            return None
        return None
    except Exception as e:
        print(f"Error parseando fecha '{fecha_str}': {e}")
        return None

@app.get("/test")
def test_endpoint():
    return {"message": "Server is working", "ventas_hoy": 22000}

@app.get("/kpis", response_model=Dict)
def get_kpis():
    """Calcular KPIs principales de Aguas Ancud"""
    print("=== INICIO ENDPOINT KPIs ===")
    try:
        print("Intentando obtener pedidos desde:", ENDPOINT_PEDIDOS)
        response = requests.get(ENDPOINT_PEDIDOS, headers=HEADERS, timeout=10)
        print(f"Status code: {response.status_code}")
        response.raise_for_status()
        
        try:
            pedidos = response.json()
            print(f"Pedidos obtenidos: {len(pedidos)} registros")
            
            # Debug: Mostrar algunos pedidos de ejemplo
            if len(pedidos) > 0:
                print("=== MUESTRA DE PEDIDOS ===")
                for i, pedido in enumerate(pedidos[:3]):
                    print(f"Pedido {i+1}: {pedido}")
                print("=== FIN MUESTRA ===")
        except Exception as e:
            print("Error al parsear JSON de pedidos:", e)
            return {"error": "Error parsing JSON"}
            
    except Exception as e:
        print("Error al obtener pedidos para KPIs:", e)
        return {"error": f"Error: {str(e)}"}
    
    df = pd.DataFrame(pedidos)
    print(f"Total de pedidos para KPIs antes del filtro: {len(df)}")
    
    if 'nombrelocal' in df.columns:
        df = df[df['nombrelocal'] == 'Aguas Ancud']
        print(f"Pedidos para KPIs después del filtro Aguas Ancud: {len(df)}")
    
    if df.empty or 'fecha' not in df.columns:
        print("DataFrame vacío o sin columna fecha")
        return {"error": "No hay datos válidos"}
    
    try:
        # Convertir fechas correctamente
        df['fecha_parsed'] = df['fecha'].apply(parse_fecha)
        df = df.dropna(subset=['fecha_parsed'])
        print(f"Pedidos con fechas válidas: {len(df)}")
        
        # Debug: Mostrar rango de fechas disponibles
        if len(df) > 0:
            fecha_min = df['fecha_parsed'].min()
            fecha_max = df['fecha_parsed'].max()
            print(f"📅 Rango de fechas disponibles: {fecha_min} a {fecha_max}")
            
            # Mostrar distribución por mes
            df['mes_anio'] = df['fecha_parsed'].dt.to_period('M')
            distribucion_meses = df['mes_anio'].value_counts().sort_index(ascending=False)
            print("📊 Distribución de pedidos por mes:")
            for mes, cantidad in distribucion_meses.head(10).items():
                print(f"   {mes}: {cantidad} pedidos")
        
        # Convertir precios
        df['precio'] = pd.to_numeric(df['precio'], errors='coerce').fillna(0)
        df['cantidad'] = df['precio'] // 2000
        
        # Agregar columna cliente basada en usuario
        if 'usuario' in df.columns:
            df['cliente'] = df['usuario']
        
        # Calcular fechas para filtros
        hoy = datetime.now()
        mes_actual = hoy.month
        anio_actual = hoy.year
        
        print(f"🗓️ Fecha actual: {hoy}")
        print(f"📅 Mes actual: {mes_actual}, Año actual: {anio_actual}")
        
        # Mes pasado
        if mes_actual == 1:
            mes_pasado = 12
            anio_pasado = anio_actual - 1
        else:
            mes_pasado = mes_actual - 1
            anio_pasado = anio_actual
        
        print(f"📅 Mes pasado: {mes_pasado}, Año pasado: {anio_pasado}")
        
        # Filtrar pedidos por mes - USAR SIEMPRE LOS 2 MESES MÁS RECIENTES
        print("🔄 Buscando los 2 meses más recientes con datos...")
        
        # Obtener el mes más reciente con datos
        df['mes_anio'] = df['fecha_parsed'].dt.to_period('M')
        meses_con_datos = df['mes_anio'].value_counts().sort_index(ascending=False)
        
        if len(meses_con_datos) >= 2:
            mes_reciente = meses_con_datos.index[0]
            mes_anterior = meses_con_datos.index[1]
            
            # Convertir Period a datetime para filtrado
            mes_reciente_dt = mes_reciente.to_timestamp()
            mes_anterior_dt = mes_anterior.to_timestamp()
            
            pedidos_mes = df[
                (df['fecha_parsed'].dt.month == mes_reciente_dt.month) & 
                (df['fecha_parsed'].dt.year == mes_reciente_dt.year)
            ]
            pedidos_mes_pasado = df[
                (df['fecha_parsed'].dt.month == mes_anterior_dt.month) & 
                (df['fecha_parsed'].dt.year == mes_anterior_dt.year)
            ]
            
            print(f"✅ Usando datos de {mes_reciente} como mes actual ({len(pedidos_mes)} pedidos)")
            print(f"✅ Usando datos de {mes_anterior} como mes anterior ({len(pedidos_mes_pasado)} pedidos)")
        else:
            print("⚠️ No hay suficientes meses con datos, usando datos simulados")
            # Usar datos simulados basados en el total disponible
            pedidos_mes = df.sample(min(100, len(df)))
            pedidos_mes_pasado = df.sample(min(80, len(df)))
        
        print(f"Pedidos mes actual: {len(pedidos_mes)}")
        print(f"Pedidos mes pasado: {len(pedidos_mes_pasado)}")
        
        # Debug: Mostrar algunos pedidos del mes actual
        if len(pedidos_mes) > 0:
            print("=== MUESTRA PEDIDOS MES ACTUAL ===")
            print(pedidos_mes[['fecha', 'precio', 'cantidad', 'usuario']].head())
            print("=== FIN MUESTRA ===")
        
        # Calcular KPIs
        ventas_mes = pedidos_mes['precio'].sum()
        ventas_mes_pasado = pedidos_mes_pasado['precio'].sum()
        total_bidones_mes = pedidos_mes['cantidad'].sum()
        total_bidones_mes_pasado = pedidos_mes_pasado['cantidad'].sum()
        
        print(f"Ventas mes actual: ${ventas_mes:,}")
        print(f"Ventas mes pasado: ${ventas_mes_pasado:,}")
        print(f"Bidones mes actual: {total_bidones_mes}")
        print(f"Bidones mes pasado: {total_bidones_mes_pasado}")
        
        # Cálculo de costos según especificaciones
        cuota_camion = 260000  # Costo fijo mensual del camión
        costo_tapa = 51  # Costo por tapa (sin IVA)
        precio_venta_bidon = 2000
        
        # Costo por bidón: 1 tapa + IVA
        costo_tapa_con_iva = costo_tapa * 1.19  # 51 + 19% IVA = 60.69 pesos
        costos_variables = costo_tapa_con_iva * total_bidones_mes  # Costos por bidones vendidos
        costos_reales = cuota_camion + costos_variables  # Costos fijos + variables
        
        # Cálculo de IVA
        iva_ventas = ventas_mes * 0.19  # IVA de las ventas
        iva_tapas = (costo_tapa * total_bidones_mes) * 0.19  # IVA de las tapas compradas
        iva = iva_ventas - iva_tapas  # IVA neto a pagar
        
        # Cálculo de IVA del mes pasado
        iva_ventas_mes_pasado = ventas_mes_pasado * 0.19
        iva_tapas_mes_pasado = (costo_tapa * total_bidones_mes_pasado) * 0.19
        iva_mes_pasado = iva_ventas_mes_pasado - iva_tapas_mes_pasado
        
        # Cálculo de utilidad: Ventas - Costos (sin restar IVA, ya que los costos ya incluyen IVA)
        utilidad = ventas_mes - costos_reales
        
        # Cálculo de utilidad del mes pasado
        costos_mes_pasado = cuota_camion + (costo_tapa_con_iva * total_bidones_mes_pasado)
        utilidad_mes_pasado = ventas_mes_pasado - costos_mes_pasado
        
        # Cálculo de ticket promedio del mes pasado
        def calcularTicketPromedio(ventas, pedidos):
            return ventas / pedidos if pedidos > 0 else 0
        
        ticket_promedio_mes_pasado = calcularTicketPromedio(ventas_mes_pasado, len(pedidos_mes_pasado))
        
        # Cálculo de clientes activos del mes pasado
        clientes_activos_mes_pasado = len(pedidos_mes_pasado['usuario'].unique()) if 'usuario' in pedidos_mes_pasado.columns else 0
        
        # Cálculo de clientes inactivos del mes pasado (aproximación)
        clientes_inactivos_mes_pasado = max(0, round(clientes_activos_mes_pasado * 0.2))
        
        # Cálculo punto de equilibrio
        try:
            # Punto de equilibrio: cuota camión / (precio venta - costo por bidón con IVA)
            punto_equilibrio = int(round(cuota_camion / (precio_venta_bidon - costo_tapa_con_iva)))
        except ZeroDivisionError:
            punto_equilibrio = 0
        
        # Cálculo de capacidad utilizada
        capacidad_total_litros = 30000  # 30.000 litros por mes
        litros_vendidos = total_bidones_mes * 20  # Cada bidón = 20 litros
        capacidad_utilizada_porcentaje = min(100, (litros_vendidos / capacidad_total_litros) * 100)
        
        # Cálculo clientes activos últimos 2 meses
        clientes_ultimos_2m = pd.concat([pedidos_mes, pedidos_mes_pasado])
        clientes_activos = len(clientes_ultimos_2m['usuario'].unique()) if 'usuario' in clientes_ultimos_2m.columns else 0
        
        resultado = {
            "ventas_mes": int(ventas_mes),
            "ventas_mes_pasado": int(ventas_mes_pasado),
            "total_pedidos_mes": len(pedidos_mes),
            "total_pedidos_mes_pasado": len(pedidos_mes_pasado),
            "total_litros_mes": int(total_bidones_mes * 20),
            "litros_vendidos_mes_pasado": int(total_bidones_mes_pasado * 20),
            "costos_reales": int(costos_reales),
            "iva": int(iva),
            "iva_mes_pasado": int(iva_mes_pasado),
            "utilidad": int(utilidad),
            "utilidad_mes_pasado": int(utilidad_mes_pasado),
            "ticket_promedio_mes_pasado": int(ticket_promedio_mes_pasado),
            "clientes_activos_mes_pasado": clientes_activos_mes_pasado,
            "clientes_inactivos_mes_pasado": clientes_inactivos_mes_pasado,
            "punto_equilibrio": punto_equilibrio,
            "clientes_activos": clientes_activos,
            "capacidad_utilizada": round(capacidad_utilizada_porcentaje, 1),
            "litros_vendidos": int(litros_vendidos),
            "capacidad_total": capacidad_total_litros,
        }
        
        print("=== RESULTADO KPIs ===")
        print(resultado)
        print("=== FIN ENDPOINT KPIs ===")
        return resultado
        
    except Exception as e:
        print(f"Error en cálculo de KPIs: {e}")
        return {"error": f"Error en cálculo: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 