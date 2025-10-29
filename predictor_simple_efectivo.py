#!/usr/bin/env python3
"""
Predictor Simple y Efectivo basado en datos reales de Aguas Ancud
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

def crear_predictor_simple(df):
    """Crea un predictor simple basado en datos reales"""
    print("=== CREANDO PREDICTOR SIMPLE Y EFECTIVO ===\n")
    
    # 1. Calcular estadísticas básicas
    pedidos_por_fecha = df.groupby(df['fecha_parsed'].dt.date).size()
    
    promedio_general = pedidos_por_fecha.mean()
    mediana_general = pedidos_por_fecha.median()
    desviacion = pedidos_por_fecha.std()
    
    print(f"📊 ESTADÍSTICAS REALES:")
    print(f"   - Promedio diario: {promedio_general:.1f} pedidos")
    print(f"   - Mediana diaria: {mediana_general:.1f} pedidos")
    print(f"   - Desviación estándar: {desviacion:.1f}")
    print(f"   - Mínimo: {pedidos_por_fecha.min()} pedidos")
    print(f"   - Máximo: {pedidos_por_fecha.max()} pedidos")
    
    # 2. Calcular factores por día de la semana (normalizados correctamente)
    df['dia_semana'] = df['fecha_parsed'].dt.dayofweek
    pedidos_por_dia = df.groupby('dia_semana').size()
    
    # Normalizar por el promedio general
    factores_dia = {}
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    print(f"\n📅 FACTORES POR DÍA DE LA SEMANA:")
    for dia in range(7):
        if dia in pedidos_por_dia.index:
            factor = pedidos_por_dia[dia] / promedio_general
            factores_dia[dia] = round(factor, 3)
            print(f"   {dias_semana[dia]}: {factores_dia[dia]} ({pedidos_por_dia[dia]} pedidos)")
        else:
            factores_dia[dia] = 1.0
            print(f"   {dias_semana[dia]}: 1.000 (sin datos)")
    
    # 3. Calcular factores por mes (normalizados correctamente)
    df['mes'] = df['fecha_parsed'].dt.month
    pedidos_por_mes = df.groupby('mes').size()
    
    factores_mes = {}
    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    print(f"\n📆 FACTORES POR MES:")
    for mes in range(1, 13):
        if mes in pedidos_por_mes.index:
            factor = pedidos_por_mes[mes] / (promedio_general * 30)  # Normalizar por día promedio
            factores_mes[mes-1] = round(factor, 3)  # mes-1 para JavaScript
            print(f"   {meses[mes-1]}: {factores_mes[mes-1]} ({pedidos_por_mes[mes]} pedidos)")
        else:
            factores_mes[mes-1] = 1.0
            print(f"   {meses[mes-1]}: 1.000 (sin datos)")
    
    # 4. Crear función de predicción simple
    def predecir_simple(fecha_objetivo, tipo_cliente="residencial"):
        """Predicción simple y efectiva"""
        fecha = datetime.strptime(fecha_objetivo, "%Y-%m-%d")
        dia_semana = fecha.weekday()
        mes = fecha.month - 1
        
        # Usar mediana como base (más estable que promedio)
        base_prediccion = mediana_general
        
        # Aplicar factores
        factor_dia = factores_dia.get(dia_semana, 1.0)
        factor_mes = factores_mes.get(mes, 1.0)
        
        # Factor de tipo de cliente (simplificado)
        factor_tipo = {
            'recurrente': 1.2,    # 20% más
            'residencial': 1.0,   # Base
            'nuevo': 0.8,         # 20% menos
            'empresa': 0.8        # 20% menos
        }.get(tipo_cliente, 1.0)
        
        # Cálculo simple
        prediccion = base_prediccion * factor_dia * factor_mes * factor_tipo
        
        # Aplicar factor de seguridad (ser conservador)
        factor_seguridad = 0.9
        prediccion_final = round(prediccion * factor_seguridad)
        
        # Asegurar que no sea negativo
        prediccion_final = max(1, prediccion_final)
        
        return {
            'prediccion': prediccion_final,
            'confianza': 75,  # Confianza moderada
            'factores': {
                'base': base_prediccion,
                'factor_dia': factor_dia,
                'factor_mes': factor_mes,
                'factor_tipo': factor_tipo,
                'factor_seguridad': factor_seguridad
            }
        }
    
    return predecir_simple, {
        'promedio_general': promedio_general,
        'mediana_general': mediana_general,
        'desviacion': desviacion,
        'factores_dia': factores_dia,
        'factores_mes': factores_mes
    }

def probar_predictor_simple(predictor_simple, df):
    """Prueba el predictor simple con datos reales"""
    print("\n=== PRUEBA DEL PREDICTOR SIMPLE ===\n")
    
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
        
        # Probar con tipo 'residencial'
        prediccion = predictor_simple(fecha_str, 'residencial')
        
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
    predicciones_excelentes = len([e for e in errores if e <= 25])
    predicciones_buenas = len([e for e in errores if e <= 50])
    predicciones_aceptables = len([e for e in errores if e <= 75])
    
    print(f"📊 RESULTADOS DE LA PRUEBA:")
    print(f"   - Fechas probadas: {len(resultados_prueba)}")
    print(f"   - Error promedio: {error_promedio:.1f}%")
    print(f"   - Predicciones excelentes (≤25% error): {predicciones_excelentes}/{len(resultados_prueba)} ({predicciones_excelentes/len(resultados_prueba)*100:.1f}%)")
    print(f"   - Predicciones buenas (≤50% error): {predicciones_buenas}/{len(resultados_prueba)} ({predicciones_buenas/len(resultados_prueba)*100:.1f}%)")
    print(f"   - Predicciones aceptables (≤75% error): {predicciones_aceptables}/{len(resultados_prueba)} ({predicciones_aceptables/len(resultados_prueba)*100:.1f}%)")
    
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
    
    # Mostrar algunas predicciones recientes
    print(f"\n📋 PREDICCIONES RECIENTES:")
    for resultado in resultados_prueba[-5:]:  # Últimas 5 predicciones
        print(f"   {resultado['fecha']}: Real {resultado['real']} vs Pred {resultado['prediccion']} (Error: {resultado['error_porcentual']:.1f}%)")
    
    return {
        'error_promedio': error_promedio,
        'predicciones_excelentes': predicciones_excelentes,
        'predicciones_buenas': predicciones_buenas,
        'predicciones_aceptables': predicciones_aceptables,
        'total_predicciones': len(resultados_prueba),
        'resultados_detallados': resultados_prueba
    }

def generar_recomendaciones(resultados_prueba):
    """Genera recomendaciones basadas en los resultados"""
    print(f"\n=== RECOMENDACIONES ===\n")
    
    error_promedio = resultados_prueba['error_promedio']
    
    if error_promedio < 30:
        print("✅ EXCELENTE: El predictor simple funciona muy bien")
        print("   - Mantener la lógica actual")
        print("   - Considerar agregar más factores si es necesario")
    elif error_promedio < 50:
        print("🟡 BUENO: El predictor funciona bien")
        print("   - Revisar factores de mes")
        print("   - Ajustar factor de seguridad")
    elif error_promedio < 75:
        print("🟠 MODERADO: El predictor necesita ajustes")
        print("   - Revisar todos los factores")
        print("   - Considerar factores externos")
    else:
        print("🔴 CRÍTICO: El predictor necesita mejoras significativas")
        print("   - Revisar completamente la lógica")
        print("   - Considerar enfoque diferente")
    
    # Recomendaciones específicas
    print(f"\n📋 RECOMENDACIONES ESPECÍFICAS:")
    print(f"   - Usar mediana en lugar de promedio para mayor estabilidad")
    print(f"   - Aplicar factor de seguridad conservador (0.9)")
    print(f"   - Simplificar factores de tipo de cliente")
    print(f"   - Validar predicciones con datos reales regularmente")

def main():
    print("=== PREDICTOR SIMPLE Y EFECTIVO ===\n")
    
    # 1. Obtener datos reales
    print("1. Obteniendo datos reales...")
    df = obtener_datos_reales()
    if df.empty:
        print("No se pudieron obtener datos")
        return
    
    print(f"   ✓ {len(df)} pedidos obtenidos")
    
    # 2. Crear predictor simple
    print("\n2. Creando predictor simple...")
    predictor_simple, estadisticas = crear_predictor_simple(df)
    
    # 3. Probar predictor
    print("\n3. Probando predictor...")
    resultados_prueba = probar_predictor_simple(predictor_simple, df)
    
    # 4. Generar recomendaciones
    generar_recomendaciones(resultados_prueba)
    
    # 5. Guardar resultados
    print(f"\n4. Guardando resultados...")
    resultados_completos = {
        'fecha_analisis': datetime.now().isoformat(),
        'estadisticas': estadisticas,
        'resultados_prueba': resultados_prueba
    }
    
    with open('predictor_simple_resultados.json', 'w', encoding='utf-8') as f:
        json.dump(resultados_completos, f, indent=2, ensure_ascii=False)
    
    print("   ✓ Resultados guardados en 'predictor_simple_resultados.json'")
    
    # 6. Resumen final
    print(f"\n=== RESUMEN FINAL ===")
    print(f"📊 Error promedio: {resultados_prueba['error_promedio']:.1f}%")
    print(f"📊 Predicciones excelentes: {resultados_prueba['predicciones_excelentes']}/{resultados_prueba['total_predicciones']}")
    print(f"📊 Predicciones buenas: {resultados_prueba['predicciones_buenas']}/{resultados_prueba['total_predicciones']}")
    
    if resultados_prueba['error_promedio'] < 30:
        print("✅ El predictor simple es EXCELENTE")
    elif resultados_prueba['error_promedio'] < 50:
        print("🟡 El predictor simple es BUENO")
    else:
        print("🔴 El predictor simple necesita mejoras")

if __name__ == "__main__":
    main() 