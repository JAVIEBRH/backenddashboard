import requests
import pandas as pd
from datetime import datetime, timedelta
import json

def obtener_datos_reales_3_meses():
    """Obtiene datos reales de los últimos 3 meses para entrenar el predictor"""
    try:
        # Obtener datos de la API
        response = requests.get("https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php", 
                              headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        pedidos = response.json()
        
        # Procesar datos
        df = pd.DataFrame(pedidos)
        if 'nombrelocal' in df.columns:
            df = df[df['nombrelocal'] == 'Aguas Ancud']
        
        if df.empty:
            return {"error": "No hay datos suficientes"}
        
        # Convertir fechas
        df['fecha_parsed'] = df['fecha'].apply(lambda x: datetime.strptime(x, "%d-%m-%Y") if x else None)
        df = df.dropna(subset=['fecha_parsed'])
        
        # Filtrar últimos 3 meses
        fecha_limite = datetime.now() - timedelta(days=90)
        df_3_meses = df[df['fecha_parsed'] >= fecha_limite].copy()
        
        if df_3_meses.empty:
            return {"error": "No hay datos de los últimos 3 meses"}
        
        return df_3_meses
        
    except Exception as e:
        return {"error": f"Error obteniendo datos: {str(e)}"}

def calcular_factores_reales(df):
    """Calcula factores reales basados en datos históricos"""
    try:
        # Agrupar por fecha
        pedidos_por_dia = df.groupby(df['fecha_parsed'].dt.date).size()
        
        # Estadísticas generales
        media_general = pedidos_por_dia.mean()
        mediana_general = pedidos_por_dia.median()
        desviacion_general = pedidos_por_dia.std()
        
        # Factores por día de semana
        df['dia_semana'] = df['fecha_parsed'].dt.weekday()
        factores_dia_semana = {}
        
        dias_nombre = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        for dia in range(7):
            pedidos_dia = df[df['dia_semana'] == dia]
            if len(pedidos_dia) > 0:
                pedidos_por_dia_especifico = pedidos_dia.groupby(pedidos_dia['fecha_parsed'].dt.date).size()
                media_dia = pedidos_por_dia_especifico.mean()
                factor_dia = media_dia / media_general if media_general > 0 else 1.0
                factores_dia_semana[dias_nombre[dia]] = round(factor_dia, 3)
            else:
                factores_dia_semana[dias_nombre[dia]] = 1.0
        
        # Factores por mes
        df['mes'] = df['fecha_parsed'].dt.month
        factores_mes = {}
        
        meses_nombre = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                       'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        
        for mes in range(1, 13):
            pedidos_mes = df[df['mes'] == mes]
            if len(pedidos_mes) > 0:
                pedidos_por_mes_especifico = pedidos_mes.groupby(pedidos_mes['fecha_parsed'].dt.date).size()
                media_mes = pedidos_por_mes_especifico.mean()
                factor_mes = media_mes / media_general if media_general > 0 else 1.0
                factores_mes[meses_nombre[mes-1]] = round(factor_mes, 3)
            else:
                factores_mes[meses_nombre[mes-1]] = 1.0
        
        # Análisis de tendencias simple
        fechas_ordenadas = sorted(pedidos_por_dia.index)
        factor_tendencia = 1.0
        
        if len(fechas_ordenadas) >= 14:  # Al menos 2 semanas
            primera_semana = fechas_ordenadas[:7]
            ultima_semana = fechas_ordenadas[-7:]
            
            try:
                promedio_primera = sum(pedidos_por_dia[fecha] for fecha in primera_semana) / 7
                promedio_ultima = sum(pedidos_por_dia[fecha] for fecha in ultima_semana) / 7
                
                if promedio_primera > 0:
                    factor_tendencia = promedio_ultima / promedio_primera
                    factor_tendencia = round(factor_tendencia, 3)
            except:
                factor_tendencia = 1.0
        
        # Calcular efectividad estimada
        coeficiente_variacion = desviacion_general / media_general if media_general > 0 else 0
        
        if coeficiente_variacion < 0.2:
            efectividad = 95
        elif coeficiente_variacion < 0.3:
            efectividad = 85
        elif coeficiente_variacion < 0.4:
            efectividad = 75
        elif coeficiente_variacion < 0.5:
            efectividad = 65
        else:
            efectividad = 55
        
        # Generar recomendaciones
        recomendaciones = []
        
        if coeficiente_variacion < 0.2:
            recomendaciones.append("✅ Datos muy estables: El predictor tendrá alta efectividad")
        elif coeficiente_variacion < 0.3:
            recomendaciones.append("✅ Datos estables: Buena efectividad esperada")
        elif coeficiente_variacion < 0.4:
            recomendaciones.append("⚠️ Datos moderadamente variables: Efectividad aceptable")
        else:
            recomendaciones.append("⚠️ Datos muy variables: Considerar factores adicionales")
        
        if factor_tendencia > 1.1:
            recomendaciones.append("📈 Tendencia creciente detectada: Ajustar factores al alza")
        elif factor_tendencia < 0.9:
            recomendaciones.append("📉 Tendencia decreciente detectada: Ajustar factores a la baja")
        else:
            recomendaciones.append("➡️ Tendencia estable: Factores actuales apropiados")
        
        # Análisis de días con mayor/menor demanda
        for dia, factor in factores_dia_semana.items():
            if factor > 1.2:
                recomendaciones.append(f"📊 {dia}: Alta demanda (factor {factor})")
            elif factor < 0.8:
                recomendaciones.append(f"📊 {dia}: Baja demanda (factor {factor})")
        
        return {
            "fecha_entrenamiento": datetime.now().isoformat(),
            "periodo_analisis": "3 meses",
            "total_dias_analizados": len(pedidos_por_dia),
            "total_pedidos_analizados": len(df),
            "estadisticas_generales": {
                "media": round(media_general, 2),
                "mediana": round(mediana_general, 2),
                "desviacion_estandar": round(desviacion_general, 2),
                "coeficiente_variacion": round(coeficiente_variacion, 3)
            },
            "factores_dia_semana": factores_dia_semana,
            "factores_mes": factores_mes,
            "factor_tendencia": factor_tendencia,
            "efectividad_estimada": efectividad,
            "recomendaciones_entrenamiento": recomendaciones
        }
        
    except Exception as e:
        return {"error": f"Error calculando factores: {str(e)}"}

def guardar_factores_entrenados(factores):
    """Guarda los factores entrenados en un archivo JSON"""
    try:
        with open('factores_entrenados.json', 'w', encoding='utf-8') as f:
            json.dump(factores, f, ensure_ascii=False, indent=2)
        print("✅ Factores entrenados guardados en 'factores_entrenados.json'")
    except Exception as e:
        print(f"❌ Error guardando factores: {str(e)}")

def entrenar_predictor():
    """Función principal para entrenar el predictor"""
    print("🚀 Iniciando entrenamiento del predictor con datos reales...")
    print("=" * 60)
    
    # Obtener datos reales
    print("📊 Obteniendo datos de los últimos 3 meses...")
    df = obtener_datos_reales_3_meses()
    
    if isinstance(df, dict) and "error" in df:
        print(f"❌ Error: {df['error']}")
        return
    
    print(f"✅ Datos obtenidos: {len(df)} pedidos en {len(df.groupby(df['fecha_parsed'].dt.date))} días")
    
    # Calcular factores
    print("\n🔧 Calculando factores reales...")
    factores = calcular_factores_reales(df)
    
    if "error" in factores:
        print(f"❌ Error: {factores['error']}")
        return
    
    # Mostrar resultados
    print("\n📈 RESULTADOS DEL ENTRENAMIENTO:")
    print("=" * 60)
    
    print(f"📅 Fecha de entrenamiento: {factores['fecha_entrenamiento']}")
    print(f"⏱️ Período analizado: {factores['periodo_analisis']}")
    print(f"📊 Días analizados: {factores['total_dias_analizados']}")
    print(f"📦 Pedidos analizados: {factores['total_pedidos_analizados']}")
    
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    stats = factores['estadisticas_generales']
    print(f"   • Media: {stats['media']} pedidos/día")
    print(f"   • Mediana: {stats['mediana']} pedidos/día")
    print(f"   • Desviación estándar: {stats['desviacion_estandar']}")
    print(f"   • Coeficiente de variación: {stats['coeficiente_variacion']}")
    
    print(f"\n🎯 EFECTIVIDAD ESTIMADA: {factores['efectividad_estimada']}%")
    
    print(f"\n📅 FACTORES POR DÍA DE SEMANA:")
    for dia, factor in factores['factores_dia_semana'].items():
        print(f"   • {dia}: {factor}")
    
    print(f"\n📈 FACTOR DE TENDENCIA: {factores['factor_tendencia']}")
    
    print(f"\n💡 RECOMENDACIONES:")
    for rec in factores['recomendaciones_entrenamiento']:
        print(f"   • {rec}")
    
    # Guardar factores
    print(f"\n💾 Guardando factores entrenados...")
    guardar_factores_entrenados(factores)
    
    print("\n✅ ¡Entrenamiento completado exitosamente!")
    print("🎉 El predictor ahora está optimizado con datos reales")

if __name__ == "__main__":
    entrenar_predictor() 