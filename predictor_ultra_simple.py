#!/usr/bin/env python3
"""
Predictor Ultra Simple - Basado en mediana real de pedidos
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
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
        
        return df
    except Exception as e:
        print(f"Error obteniendo datos reales: {e}")
        return pd.DataFrame()

def crear_predictor_ultra_simple(df):
    """Crea un predictor ultra simple basado en mediana real"""
    print("=== PREDICTOR ULTRA SIMPLE ===\n")
    
    # Calcular estadísticas básicas
    pedidos_por_fecha = df.groupby(df['fecha_parsed'].dt.date).size()
    
    mediana_real = pedidos_por_fecha.median()
    promedio_real = pedidos_por_fecha.mean()
    
    print(f"📊 ESTADÍSTICAS REALES:")
    print(f"   - Mediana diaria: {mediana_real:.1f} pedidos")
    print(f"   - Promedio diario: {promedio_real:.1f} pedidos")
    print(f"   - Mínimo: {pedidos_por_fecha.min()} pedidos")
    print(f"   - Máximo: {pedidos_por_fecha.max()} pedidos")
    
    # Análisis por día de la semana
    df['dia_semana'] = df['fecha_parsed'].dt.dayofweek
    pedidos_por_dia = df.groupby('dia_semana').size()
    
    # Calcular mediana por día de la semana
    medianas_por_dia = {}
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    print(f"\n📅 MEDIANA POR DÍA DE LA SEMANA:")
    for dia in range(7):
        datos_dia = df[df['dia_semana'] == dia]
        if not datos_dia.empty:
            pedidos_dia = datos_dia.groupby(datos_dia['fecha_parsed'].dt.date).size()
            mediana_dia = pedidos_dia.median()
            medianas_por_dia[dia] = mediana_dia
            print(f"   {dias_semana[dia]}: {mediana_dia:.1f} pedidos")
        else:
            medianas_por_dia[dia] = mediana_real
            print(f"   {dias_semana[dia]}: {mediana_real:.1f} pedidos (sin datos)")
    
    def predecir_ultra_simple(fecha_objetivo, tipo_cliente="residencial"):
        """Predicción ultra simple basada en mediana real"""
        fecha = datetime.strptime(fecha_objetivo, "%Y-%m-%d")
        dia_semana = fecha.weekday()
        
        # Usar mediana del día específico o mediana general
        base_prediccion = medianas_por_dia.get(dia_semana, mediana_real)
        
        # Factor de tipo de cliente (muy simple)
        factor_tipo = {
            'recurrente': 1.1,    # 10% más
            'residencial': 1.0,   # Base
            'nuevo': 0.9,         # 10% menos
            'empresa': 0.9        # 10% menos
        }.get(tipo_cliente, 1.0)
        
        # Predicción simple
        prediccion = round(base_prediccion * factor_tipo)
        
        # Asegurar mínimo de 1 pedido
        prediccion = max(1, prediccion)
        
        return {
            'prediccion': prediccion,
            'confianza': 80,
            'base_mediana': base_prediccion,
            'factor_tipo': factor_tipo
        }
    
    return predecir_ultra_simple, {
        'mediana_general': mediana_real,
        'promedio_general': promedio_real,
        'medianas_por_dia': medianas_por_dia
    }

def probar_predictor_ultra_simple(predictor, df):
    """Prueba el predictor ultra simple"""
    print("\n=== PRUEBA DEL PREDICTOR ULTRA SIMPLE ===\n")
    
    # Usar últimos 30 días
    hoy = datetime.now()
    fecha_inicio = hoy - timedelta(days=30)
    df_reciente = df[df['fecha_parsed'] >= fecha_inicio].copy()
    
    if df_reciente.empty:
        print("No hay datos recientes")
        return
    
    # Agrupar por fecha
    pedidos_por_fecha = df_reciente.groupby(df_reciente['fecha_parsed'].dt.date).size().reset_index()
    pedidos_por_fecha.columns = ['fecha', 'pedidos_reales']
    
    resultados = []
    
    for _, row in pedidos_por_fecha.iterrows():
        fecha_str = row['fecha'].strftime('%Y-%m-%d')
        pedidos_reales = row['pedidos_reales']
        
        prediccion = predictor(fecha_str, 'residencial')
        
        diferencia = abs(pedidos_reales - prediccion['prediccion'])
        porcentaje_error = (diferencia / pedidos_reales * 100) if pedidos_reales > 0 else 0
        
        resultados.append({
            'fecha': fecha_str,
            'real': pedidos_reales,
            'prediccion': prediccion['prediccion'],
            'error_porcentual': porcentaje_error
        })
    
    # Calcular métricas
    errores = [r['error_porcentual'] for r in resultados]
    error_promedio = sum(errores) / len(errores)
    predicciones_excelentes = len([e for e in errores if e <= 30])
    predicciones_buenas = len([e for e in errores if e <= 60])
    predicciones_aceptables = len([e for e in errores if e <= 100])
    
    print(f"📊 RESULTADOS:")
    print(f"   - Fechas probadas: {len(resultados)}")
    print(f"   - Error promedio: {error_promedio:.1f}%")
    print(f"   - Predicciones excelentes (≤30% error): {predicciones_excelentes}/{len(resultados)} ({predicciones_excelentes/len(resultados)*100:.1f}%)")
    print(f"   - Predicciones buenas (≤60% error): {predicciones_buenas}/{len(resultados)} ({predicciones_buenas/len(resultados)*100:.1f}%)")
    print(f"   - Predicciones aceptables (≤100% error): {predicciones_aceptables}/{len(resultados)} ({predicciones_aceptables/len(resultados)*100:.1f}%)")
    
    # Mejor y peor predicción
    mejor = min(resultados, key=lambda x: x['error_porcentual'])
    peor = max(resultados, key=lambda x: x['error_porcentual'])
    
    print(f"\n🏆 MEJOR PREDICCIÓN:")
    print(f"   - Fecha: {mejor['fecha']}")
    print(f"   - Real: {mejor['real']} vs Predicción: {mejor['prediccion']}")
    print(f"   - Error: {mejor['error_porcentual']:.1f}%")
    
    print(f"\n⚠️  PEOR PREDICCIÓN:")
    print(f"   - Fecha: {peor['fecha']}")
    print(f"   - Real: {peor['real']} vs Predicción: {peor['prediccion']}")
    print(f"   - Error: {peor['error_porcentual']:.1f}%")
    
    # Predicciones recientes
    print(f"\n📋 PREDICCIONES RECIENTES:")
    for resultado in resultados[-5:]:
        print(f"   {resultado['fecha']}: Real {resultado['real']} vs Pred {resultado['prediccion']} (Error: {resultado['error_porcentual']:.1f}%)")
    
    return {
        'error_promedio': error_promedio,
        'predicciones_excelentes': predicciones_excelentes,
        'predicciones_buenas': predicciones_buenas,
        'predicciones_aceptables': predicciones_aceptables,
        'total_predicciones': len(resultados),
        'resultados_detallados': resultados
    }

def generar_recomendacion_final(resultados):
    """Genera recomendación final"""
    print(f"\n=== RECOMENDACIÓN FINAL ===\n")
    
    error_promedio = resultados['error_promedio']
    
    if error_promedio < 40:
        print("✅ EXCELENTE: El predictor ultra simple funciona muy bien")
        print("   - Implementar inmediatamente")
        print("   - Monitorear resultados")
    elif error_promedio < 70:
        print("🟡 BUENO: El predictor funciona aceptablemente")
        print("   - Implementar con monitoreo")
        print("   - Ajustar según resultados")
    elif error_promedio < 100:
        print("🟠 MODERADO: El predictor necesita ajustes")
        print("   - Revisar lógica")
        print("   - Considerar factores adicionales")
    else:
        print("🔴 CRÍTICO: El predictor no es confiable")
        print("   - No implementar")
        print("   - Revisar completamente el enfoque")

def main():
    print("=== PREDICTOR ULTRA SIMPLE ===\n")
    
    # 1. Obtener datos
    print("1. Obteniendo datos reales...")
    df = obtener_datos_reales()
    if df.empty:
        print("No se pudieron obtener datos")
        return
    
    print(f"   ✓ {len(df)} pedidos obtenidos")
    
    # 2. Crear predictor
    print("\n2. Creando predictor ultra simple...")
    predictor, estadisticas = crear_predictor_ultra_simple(df)
    
    # 3. Probar predictor
    print("\n3. Probando predictor...")
    resultados = probar_predictor_ultra_simple(predictor, df)
    
    # 4. Recomendación final
    generar_recomendacion_final(resultados)
    
    # 5. Guardar resultados
    print(f"\n4. Guardando resultados...")
    resultados_completos = {
        'fecha_analisis': datetime.now().isoformat(),
        'estadisticas': estadisticas,
        'resultados_prueba': resultados
    }
    
    with open('predictor_ultra_simple_resultados.json', 'w', encoding='utf-8') as f:
        json.dump(resultados_completos, f, indent=2, ensure_ascii=False)
    
    print("   ✓ Resultados guardados en 'predictor_ultra_simple_resultados.json'")
    
    # 6. Resumen final
    print(f"\n=== RESUMEN FINAL ===")
    print(f"📊 Error promedio: {resultados['error_promedio']:.1f}%")
    print(f"📊 Predicciones excelentes: {resultados['predicciones_excelentes']}/{resultados['total_predicciones']}")
    print(f"📊 Predicciones buenas: {resultados['predicciones_buenas']}/{resultados['total_predicciones']}")
    
    if resultados['error_promedio'] < 40:
        print("✅ IMPLEMENTAR INMEDIATAMENTE")
    elif resultados['error_promedio'] < 70:
        print("🟡 IMPLEMENTAR CON MONITOREO")
    else:
        print("🔴 NO IMPLEMENTAR - NECESITA MEJORAS")

if __name__ == "__main__":
    main() 