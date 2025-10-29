import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json
from typing import Dict, List

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

def calcular_factores_reales(df: pd.DataFrame) -> Dict:
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
        
        for dia in range(7):
            pedidos_dia = df[df['dia_semana'] == dia]
            if len(pedidos_dia) > 0:
                pedidos_por_dia_especifico = pedidos_dia.groupby(pedidos_dia['fecha_parsed'].dt.date).size()
                media_dia = pedidos_por_dia_especifico.mean()
                factor_dia = media_dia / media_general if media_general > 0 else 1.0
                factores_dia_semana[dia] = round(factor_dia, 3)
            else:
                factores_dia_semana[dia] = 1.0
        
        # Factores por mes
        df['mes'] = df['fecha_parsed'].dt.month
        factores_mes = {}
        
        for mes in range(1, 13):
            pedidos_mes = df[df['mes'] == mes]
            if len(pedidos_mes) > 0:
                pedidos_por_mes_especifico = pedidos_mes.groupby(pedidos_mes['fecha_parsed'].dt.date).size()
                media_mes = pedidos_por_mes_especifico.mean()
                factor_mes = media_mes / media_general if media_general > 0 else 1.0
                factores_mes[mes] = round(factor_mes, 3)
            else:
                factores_mes[mes] = 1.0
        
        # Factores por tipo de cliente (si hay datos)
        factores_tipo_cliente = {
            'residencial': 1.0,
            'recurrente': 1.2,
            'nuevo': 0.8,
            'empresa': 1.1,
            'vip': 1.25
        }
        
        # Análisis de tendencias
        fechas_ordenadas = sorted(pedidos_por_dia.index)
        if len(fechas_ordenadas) >= 2:
            # Calcular tendencia lineal
            x = np.arange(len(fechas_ordenadas))
            y = [pedidos_por_dia.loc[fecha] for fecha in fechas_ordenadas]
            
            if len(y) > 1:
                try:
                    z = np.polyfit(x, y, 1)
                    pendiente = z[0]
                    factor_tendencia = 1.0 + (pendiente / media_general) if media_general > 0 else 1.0
                except:
                    factor_tendencia = 1.0
            else:
                factor_tendencia = 1.0
        else:
            factor_tendencia = 1.0
        
        # Análisis de estacionalidad semanal
        estacionalidad_semanal = {}
        for dia in range(7):
            dias_del_mismo_tipo = [fecha for fecha in pedidos_por_dia.index if fecha.weekday() == dia]
            if dias_del_mismo_tipo:
                promedio_dia = np.mean([pedidos_por_dia.loc[fecha] for fecha in dias_del_mismo_tipo])
                estacionalidad_semanal[dia] = round(promedio_dia, 2)
            else:
                estacionalidad_semanal[dia] = media_general
        
        return {
            "fecha_entrenamiento": datetime.now().isoformat(),
            "periodo_analisis": "3 meses",
            "total_dias_analizados": len(pedidos_por_dia),
            "total_pedidos_analizados": len(df),
            "estadisticas_generales": {
                "media": round(media_general, 2),
                "mediana": round(mediana_general, 2),
                "desviacion_estandar": round(desviacion_general, 2),
                "coeficiente_variacion": round(desviacion_general / media_general, 3) if media_general > 0 else 0
            },
            "factores_dia_semana": factores_dia_semana,
            "factores_mes": factores_mes,
            "factores_tipo_cliente": factores_tipo_cliente,
            "factor_tendencia": round(factor_tendencia, 3),
            "estacionalidad_semanal": estacionalidad_semanal,
            "efectividad_estimada": calcular_efectividad_estimada(desviacion_general, media_general),
            "recomendaciones_entrenamiento": generar_recomendaciones_entrenamiento(df, pedidos_por_dia)
        }
        
    except Exception as e:
        return {"error": f"Error calculando factores: {str(e)}"}

def calcular_efectividad_estimada(desviacion: float, media: float) -> int:
    """Calcula la efectividad estimada basada en la variabilidad de los datos"""
    if media == 0:
        return 60
    
    coeficiente_variacion = desviacion / media
    
    if coeficiente_variacion < 0.2:
        return 95  # Muy estable
    elif coeficiente_variacion < 0.3:
        return 85  # Estable
    elif coeficiente_variacion < 0.4:
        return 75  # Moderadamente estable
    elif coeficiente_variacion < 0.5:
        return 65  # Variable
    else:
        return 55  # Muy variable

def generar_recomendaciones_entrenamiento(df: pd.DataFrame, pedidos_por_dia: pd.Series) -> List[str]:
    """Genera recomendaciones basadas en el análisis de datos"""
    recomendaciones = []
    
    # Análisis de variabilidad
    media = pedidos_por_dia.mean()
    desviacion = pedidos_por_dia.std()
    coeficiente_variacion = desviacion / media if media > 0 else 0
    
    if coeficiente_variacion < 0.2:
        recomendaciones.append("✅ Datos muy estables: El predictor tendrá alta efectividad")
    elif coeficiente_variacion < 0.3:
        recomendaciones.append("✅ Datos estables: Buena efectividad esperada")
    elif coeficiente_variacion < 0.4:
        recomendaciones.append("⚠️ Datos moderadamente variables: Efectividad aceptable")
    else:
        recomendaciones.append("⚠️ Datos muy variables: Considerar factores adicionales")
    
    # Análisis de tendencias
    fechas_ordenadas = sorted(pedidos_por_dia.index)
    if len(fechas_ordenadas) >= 7:
        ultima_semana = fechas_ordenadas[-7:]
        primera_semana = fechas_ordenadas[:7]
        
        promedio_ultima = np.mean([pedidos_por_dia.loc[fecha] for fecha in ultima_semana])
        promedio_primera = np.mean([pedidos_por_dia.loc[fecha] for fecha in primera_semana])
        
        if promedio_ultima > promedio_primera * 1.1:
            recomendaciones.append("📈 Tendencia creciente detectada: Ajustar factores al alza")
        elif promedio_ultima < promedio_primera * 0.9:
            recomendaciones.append("📉 Tendencia decreciente detectada: Ajustar factores a la baja")
        else:
            recomendaciones.append("➡️ Tendencia estable: Factores actuales apropiados")
    
    # Análisis de días de la semana
    df['dia_semana'] = df['fecha_parsed'].dt.weekday()
    for dia in range(7):
        pedidos_dia = df[df['dia_semana'] == dia]
        if len(pedidos_dia) > 0:
            pedidos_por_dia_especifico = pedidos_dia.groupby(pedidos_dia['fecha_parsed'].dt.date).size()
            media_dia = pedidos_por_dia_especifico.mean()
            if media_dia > media * 1.2:
                recomendaciones.append(f"📊 Día {dia}: Alta demanda (promedio {media_dia:.1f} pedidos)")
            elif media_dia < media * 0.8:
                recomendaciones.append(f"📊 Día {dia}: Baja demanda (promedio {media_dia:.1f} pedidos)")
    
    return recomendaciones

def guardar_factores_entrenados(factores: Dict):
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
        dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        print(f"   • {dias[dia]}: {factor}")
    
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