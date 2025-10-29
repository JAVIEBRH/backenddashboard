#!/usr/bin/env python3
"""
Sistema para mejorar el predictor basado en análisis de efectividad real
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np

# Configuración
ENDPOINT_PEDIDOS = "https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def parse_fecha(fecha_str):
    """Convierte fecha del formato DD-MM-YYYY a datetime"""
    try:
        return datetime.strptime(fecha_str, "%d-%m-%Y")
    except:
        return None

def obtener_datos_reales():
    """Obtiene datos reales de pedidos"""
    try:
        response = requests.get(ENDPOINT_PEDIDOS, headers=HEADERS, timeout=10)
        response.raise_for_status()
        pedidos = response.json()
        
        df = pd.DataFrame(pedidos)
        if 'nombrelocal' in df.columns:
            df = df[df['nombrelocal'].str.strip().str.lower() == 'aguas ancud']
        
        if df.empty or 'fecha' not in df.columns:
            return pd.DataFrame()
        
        # Convertir fechas
        df['fecha_parsed'] = df['fecha'].apply(parse_fecha)
        df = df.dropna(subset=['fecha_parsed'])
        
        # Convertir precios
        df['precio'] = pd.to_numeric(df['precio'], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        print(f"Error obteniendo datos reales: {e}")
        return pd.DataFrame()

def analizar_patrones_reales(df):
    """Analiza patrones reales de los datos"""
    print("=== ANÁLISIS DE PATRONES REALES ===")
    
    # Agregar columnas de análisis
    df['mes'] = df['fecha_parsed'].dt.month
    df['dia_semana'] = df['fecha_parsed'].dt.dayofweek
    df['anio'] = df['fecha_parsed'].dt.year
    df['semana'] = df['fecha_parsed'].dt.isocalendar().week
    
    # 1. Análisis por día de la semana
    print("\n1. PATRONES POR DÍA DE LA SEMANA:")
    pedidos_por_dia = df.groupby('dia_semana').size()
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    for dia in range(7):
        if dia in pedidos_por_dia.index:
            pedidos = pedidos_por_dia[dia]
            print(f"   {dias_semana[dia]}: {pedidos} pedidos")
    
    # 2. Análisis por mes
    print("\n2. PATRONES POR MES:")
    pedidos_por_mes = df.groupby('mes').size()
    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    for mes in range(1, 13):
        if mes in pedidos_por_mes.index:
            pedidos = pedidos_por_mes[mes]
            print(f"   {meses[mes-1]}: {pedidos} pedidos")
    
    # 3. Análisis de tendencia temporal
    print("\n3. TENDENCIA TEMPORAL:")
    pedidos_por_fecha = df.groupby(df['fecha_parsed'].dt.date).size().reset_index()
    pedidos_por_fecha.columns = ['fecha', 'pedidos']
    pedidos_por_fecha['fecha'] = pd.to_datetime(pedidos_por_fecha['fecha'])
    pedidos_por_fecha = pedidos_por_fecha.sort_values('fecha')
    
    # Calcular promedio móvil de 7 días
    pedidos_por_fecha['promedio_7d'] = pedidos_por_fecha['pedidos'].rolling(window=7, min_periods=1).mean()
    
    print(f"   Promedio general: {pedidos_por_fecha['pedidos'].mean():.1f} pedidos/día")
    print(f"   Mediana: {pedidos_por_fecha['pedidos'].median():.1f} pedidos/día")
    print(f"   Desviación estándar: {pedidos_por_fecha['pedidos'].std():.1f}")
    print(f"   Mínimo: {pedidos_por_fecha['pedidos'].min()} pedidos")
    print(f"   Máximo: {pedidos_por_fecha['pedidos'].max()} pedidos")
    
    # 4. Análisis de variabilidad
    print("\n4. ANÁLISIS DE VARIABILIDAD:")
    variabilidad_por_dia = {}
    for dia in range(7):
        if dia in pedidos_por_dia.index:
            datos_dia = df[df['dia_semana'] == dia]
            pedidos_por_fecha_dia = datos_dia.groupby(datos_dia['fecha_parsed'].dt.date).size()
            variabilidad = pedidos_por_fecha_dia.std() / pedidos_por_fecha_dia.mean() if pedidos_por_fecha_dia.mean() > 0 else 0
            variabilidad_por_dia[dia] = variabilidad
            print(f"   {dias_semana[dia]}: {variabilidad:.2f} (coeficiente de variación)")
    
    return {
        'pedidos_por_dia': pedidos_por_dia.to_dict(),
        'pedidos_por_mes': pedidos_por_mes.to_dict(),
        'promedio_general': pedidos_por_fecha['pedidos'].mean(),
        'mediana': pedidos_por_fecha['pedidos'].median(),
        'desviacion_estandar': pedidos_por_fecha['pedidos'].std(),
        'variabilidad_por_dia': variabilidad_por_dia,
        'datos_por_fecha': pedidos_por_fecha
    }

def generar_factores_mejorados(patrones):
    """Genera factores mejorados basados en patrones reales"""
    print("\n=== GENERANDO FACTORES MEJORADOS ===")
    
    # 1. Factores por día de la semana (normalizados)
    pedidos_por_dia = patrones['pedidos_por_dia']
    promedio_diario = patrones['promedio_general']
    
    factores_dia_mejorados = {}
    for dia in range(7):
        if dia in pedidos_por_dia:
            factor = pedidos_por_dia[dia] / promedio_diario
            factores_dia_mejorados[dia] = round(factor, 3)
        else:
            factores_dia_mejorados[dia] = 1.0
    
    print("1. FACTORES POR DÍA DE LA SEMANA:")
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    for dia, factor in factores_dia_mejorados.items():
        print(f"   {dias_semana[dia]}: {factor}")
    
    # 2. Factores por mes (normalizados)
    pedidos_por_mes = patrones['pedidos_por_mes']
    promedio_mensual = sum(pedidos_por_mes.values()) / len(pedidos_por_mes) if pedidos_por_mes else 1
    
    factores_mes_mejorados = {}
    for mes in range(1, 13):
        if mes in pedidos_por_mes:
            factor = pedidos_por_mes[mes] / promedio_mensual
            factores_mes_mejorados[mes-1] = round(factor, 3)  # mes-1 para JavaScript
        else:
            factores_mes_mejorados[mes-1] = 1.0
    
    print("\n2. FACTORES POR MES:")
    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    for mes in range(12):
        print(f"   {meses[mes]}: {factores_mes_mejorados[mes]}")
    
    # 3. Factor de variabilidad
    variabilidad_promedio = sum(patrones['variabilidad_por_dia'].values()) / len(patrones['variabilidad_por_dia'])
    factor_variabilidad = min(1.5, max(0.5, 1 / (1 + variabilidad_promedio)))
    
    print(f"\n3. FACTOR DE VARIABILIDAD: {factor_variabilidad:.3f}")
    print(f"   (Basado en coeficiente de variación promedio: {variabilidad_promedio:.3f})")
    
    # 4. Factores de tipo de cliente (ajustados)
    factores_tipo_mejorados = {
        'recurrente': 0.8,  # Reducido porque estaba sobreestimando
        'residencial': 1.0,  # Base
        'nuevo': 0.6,        # Reducido
        'empresa': 0.6       # Reducido
    }
    
    print("\n4. FACTORES POR TIPO DE CLIENTE:")
    for tipo, factor in factores_tipo_mejorados.items():
        print(f"   {tipo}: {factor}")
    
    return {
        'factores_dia_semana': factores_dia_mejorados,
        'factores_temporada': factores_mes_mejorados,
        'factores_tipo_cliente': factores_tipo_mejorados,
        'factor_variabilidad': factor_variabilidad,
        'promedio_base_real': patrones['promedio_general'],
        'mediana_real': patrones['mediana']
    }

def crear_predictor_mejorado(factores_mejorados):
    """Crea un predictor mejorado con los nuevos factores"""
    print("\n=== PREDICTOR MEJORADO ===")
    
    def predecir_mejorado(fecha_objetivo, tipo_cliente):
        """Función de predicción mejorada"""
        fecha = datetime.strptime(fecha_objetivo, "%Y-%m-%d")
        dia_semana = fecha.weekday()
        mes = fecha.month - 1
        
        # Factores mejorados
        factor_dia = factores_mejorados['factores_dia_semana'].get(dia_semana, 1.0)
        factor_mes = factores_mejorados['factores_temporada'].get(mes, 1.0)
        factor_tipo = factores_mejorados['factores_tipo_cliente'].get(tipo_cliente, 1.0)
        factor_variabilidad = factores_mejorados['factor_variabilidad']
        
        # Usar mediana en lugar de promedio para ser más conservador
        base_prediccion = factores_mejorados['mediana_real']
        
        # Cálculo mejorado
        prediccion = base_prediccion * factor_dia * factor_mes * factor_tipo * factor_variabilidad
        
        # Aplicar factor de seguridad (ser más conservador)
        factor_seguridad = 0.8
        prediccion_final = round(prediccion * factor_seguridad)
        
        return {
            'prediccion': prediccion_final,
            'factores': {
                'base': base_prediccion,
                'factor_dia': factor_dia,
                'factor_mes': factor_mes,
                'factor_tipo': factor_tipo,
                'factor_variabilidad': factor_variabilidad,
                'factor_seguridad': factor_seguridad
            }
        }
    
    return predecir_mejorado

def probar_predictor_mejorado(predictor_mejorado, df):
    """Prueba el predictor mejorado con datos reales"""
    print("\n=== PRUEBA DEL PREDICTOR MEJORADO ===")
    
    # Usar últimos 30 días para prueba
    hoy = datetime.now()
    fecha_inicio = hoy - timedelta(days=30)
    df_reciente = df[df['fecha_parsed'] >= fecha_inicio].copy()
    
    if df_reciente.empty:
        print("No hay datos recientes para probar")
        return
    
    # Agrupar por fecha
    pedidos_por_fecha = df_reciente.groupby(df_reciente['fecha_parsed'].dt.date).size().reset_index()
    pedidos_por_fecha.columns = ['fecha', 'pedidos_reales']
    
    resultados_prueba = []
    
    for _, row in pedidos_por_fecha.iterrows():
        fecha_str = row['fecha'].strftime('%Y-%m-%d')
        pedidos_reales = row['pedidos_reales']
        
        # Probar con tipo 'residencial' (el más estable)
        prediccion = predictor_mejorado(fecha_str, 'residencial')
        
        diferencia = abs(pedidos_reales - prediccion['prediccion'])
        porcentaje_error = (diferencia / pedidos_reales * 100) if pedidos_reales > 0 else 0
        
        resultados_prueba.append({
            'fecha': fecha_str,
            'real': pedidos_reales,
            'prediccion': prediccion['prediccion'],
            'error_porcentual': porcentaje_error,
            'factores': prediccion['factores']
        })
    
    # Calcular métricas
    errores = [r['error_porcentual'] for r in resultados_prueba]
    error_promedio = sum(errores) / len(errores)
    predicciones_excelentes = len([e for e in errores if e <= 20])
    predicciones_buenas = len([e for e in errores if e <= 40])
    
    print(f"📊 RESULTADOS DE LA PRUEBA:")
    print(f"   - Fechas probadas: {len(resultados_prueba)}")
    print(f"   - Error promedio: {error_promedio:.1f}%")
    print(f"   - Predicciones excelentes (≤20% error): {predicciones_excelentes}/{len(resultados_prueba)} ({predicciones_excelentes/len(resultados_prueba)*100:.1f}%)")
    print(f"   - Predicciones buenas (≤40% error): {predicciones_buenas}/{len(resultados_prueba)} ({predicciones_buenas/len(resultados_prueba)*100:.1f}%)")
    
    # Mostrar mejores y peores predicciones
    mejor_prediccion = min(resultados_prueba, key=lambda x: x['error_porcentual'])
    peor_prediccion = max(resultados_prueba, key=lambda x: x['error_porcentual'])
    
    print(f"\n🏆 MEJOR PREDICCIÓN:")
    print(f"   - Fecha: {mejor_prediccion['fecha']}")
    print(f"   - Real: {mejor_prediccion['real']} vs Predicción: {mejor_prediccion['prediccion']}")
    print(f"   - Error: {mejor_prediccion['error_porcentual']:.1f}%")
    
    print(f"\n⚠️  PEOR PREDICCIÓN:")
    print(f"   - Fecha: {peor_prediccion['fecha']}")
    print(f"   - Real: {peor_prediccion['real']} vs Predicción: {peor_prediccion['prediccion']}")
    print(f"   - Error: {peor_prediccion['error_porcentual']:.1f}%")
    
    return {
        'error_promedio': error_promedio,
        'predicciones_excelentes': predicciones_excelentes,
        'predicciones_buenas': predicciones_buenas,
        'total_predicciones': len(resultados_prueba),
        'resultados_detallados': resultados_prueba
    }

def main():
    print("=== SISTEMA DE MEJORA DEL PREDICTOR ===\n")
    
    # 1. Obtener datos reales
    print("1. Obteniendo datos reales...")
    df = obtener_datos_reales()
    if df.empty:
        print("No se pudieron obtener datos")
        return
    
    print(f"   ✓ {len(df)} pedidos obtenidos")
    
    # 2. Analizar patrones reales
    print("\n2. Analizando patrones reales...")
    patrones = analizar_patrones_reales(df)
    
    # 3. Generar factores mejorados
    print("\n3. Generando factores mejorados...")
    factores_mejorados = generar_factores_mejorados(patrones)
    
    # 4. Crear predictor mejorado
    print("\n4. Creando predictor mejorado...")
    predictor_mejorado = crear_predictor_mejorado(factores_mejorados)
    
    # 5. Probar predictor mejorado
    print("\n5. Probando predictor mejorado...")
    resultados_prueba = probar_predictor_mejorado(predictor_mejorado, df)
    
    # 6. Guardar resultados
    print("\n6. Guardando resultados...")
    resultados_completos = {
        'fecha_analisis': datetime.now().isoformat(),
        'patrones_reales': patrones,
        'factores_mejorados': factores_mejorados,
        'resultados_prueba': resultados_prueba
    }
    
    with open('predictor_mejorado.json', 'w', encoding='utf-8') as f:
        json.dump(resultados_completos, f, indent=2, ensure_ascii=False)
    
    print("   ✓ Resultados guardados en 'predictor_mejorado.json'")
    
    # 7. Resumen final
    print("\n=== RESUMEN FINAL ===")
    if resultados_prueba['error_promedio'] < 30:
        print("✅ El predictor mejorado funciona EXCELENTE")
    elif resultados_prueba['error_promedio'] < 50:
        print("🟡 El predictor mejorado funciona BIEN")
    else:
        print("🔴 El predictor mejorado aún necesita ajustes")
    
    print(f"📊 Error promedio: {resultados_prueba['error_promedio']:.1f}%")
    print(f"📊 Predicciones excelentes: {resultados_prueba['predicciones_excelentes']}/{resultados_prueba['total_predicciones']}")

if __name__ == "__main__":
    main() 