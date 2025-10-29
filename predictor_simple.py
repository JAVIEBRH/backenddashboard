import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List

def calcular_estadisticas_bidones(df: pd.DataFrame) -> Dict:
    """Calcula estadísticas de bidones por pedido para análisis híbrido"""
    try:
        # Calcular bidones por pedido basado en precio
        def calcular_bidones_por_pedido(row):
            precio = float(row.get('precio', 0))
            if precio > 0:
                return max(1, round(precio / 2000))  # $2,000 por bidón
            return 1
        
        df['bidones_estimados'] = df.apply(calcular_bidones_por_pedido, axis=1)
        
        # Estadísticas generales
        promedio_bidones = df['bidones_estimados'].mean()
        mediana_bidones = df['bidones_estimados'].median()
        desviacion_bidones = df['bidones_estimados'].std()
        
        # Estadísticas por día de semana
        df['dia_semana'] = df['fecha_parsed'].dt.weekday()
        bidones_por_dia_semana = df.groupby('dia_semana')['bidones_estimados'].agg(['mean', 'median', 'std']).to_dict()
        
        # Estadísticas por mes
        df['mes'] = df['fecha_parsed'].dt.month
        bidones_por_mes = df.groupby('mes')['bidones_estimados'].agg(['mean', 'median', 'std']).to_dict()
        
        # Patrones de volumen
        pedidos_pequenos = (df['bidones_estimados'] <= 2).sum() / len(df) * 100
        pedidos_medianos = ((df['bidones_estimados'] > 2) & (df['bidones_estimados'] <= 5)).sum() / len(df) * 100
        pedidos_grandes = (df['bidones_estimados'] > 5).sum() / len(df) * 100
        
        return {
            "promedio_bidones_por_pedido": round(promedio_bidones, 2),
            "mediana_bidones_por_pedido": round(mediana_bidones, 2),
            "desviacion_bidones": round(desviacion_bidones, 2),
            "bidones_por_dia_semana": bidones_por_dia_semana,
            "bidones_por_mes": bidones_por_mes,
            "patrones_volumen": {
                "pedidos_pequenos_1_2_bidones": round(pedidos_pequenos, 1),
                "pedidos_medianos_3_5_bidones": round(pedidos_medianos, 1),
                "pedidos_grandes_mas_5_bidones": round(pedidos_grandes, 1)
            },
            "total_pedidos_analizados": len(df)
        }
    except Exception as e:
        return {"error": f"Error calculando estadísticas de bidones: {str(e)}"}

def get_predictor_simple(fecha: str, tipo_cliente: str = "residencial") -> Dict:
    """Predictor simple pero efectivo con mejoras implementadas"""
    try:
        # Obtener datos
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
        
        # Obtener últimos 60 días
        fecha_limite = datetime.now() - timedelta(days=60)
        df_reciente = df[df['fecha_parsed'] >= fecha_limite].copy()
        
        if df_reciente.empty:
            return {"error": "No hay datos recientes"}
        
        # Calcular estadísticas base
        pedidos_por_dia = df_reciente.groupby(df_reciente['fecha_parsed'].dt.date).size()
        media = pedidos_por_dia.mean()
        mediana = pedidos_por_dia.median()
        desviacion = pedidos_por_dia.std()
        
        # Calcular estadísticas de bidones (análisis híbrido)
        estadisticas_bidones = calcular_estadisticas_bidones(df_reciente)
        
        # Factores por tipo de cliente
        factores_tipo = {
            'residencial': 1.0,
            'recurrente': 1.2,
            'nuevo': 0.8,
            'empresa': 1.1,
            'vip': 1.25
        }
        
        # Variables exógenas
        fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
        es_finde = fecha_dt.weekday() in [5, 6]
        factor_finde = 0.8 if es_finde else 1.0
        
        # Factor estacional
        mes = fecha_dt.month
        if mes in [12, 1, 2]:  # Verano
            factor_estacional = 1.2
        elif mes in [6, 7, 8]:  # Invierno
            factor_estacional = 0.9
        else:
            factor_estacional = 1.0
        
        # Calcular predicción
        factor_tipo = factores_tipo.get(tipo_cliente, 1.0)
        prediccion_base = mediana * factor_tipo * factor_finde * factor_estacional
        prediccion_final = max(1, round(prediccion_base))
        
        # Calcular rango de confianza
        rango_inferior = max(1, round(prediccion_final - desviacion))
        rango_superior = round(prediccion_final + desviacion)
        
        # Calcular predicción de bidones (análisis híbrido)
        if 'promedio_bidones_por_pedido' in estadisticas_bidones:
            promedio_bidones = estadisticas_bidones['promedio_bidones_por_pedido']
            desviacion_bidones = estadisticas_bidones.get('desviacion_bidones', 1.0)
            
            # Predicción de bidones basada en pedidos
            prediccion_bidones_min = max(1, round(prediccion_final * (promedio_bidones - desviacion_bidones)))
            prediccion_bidones_max = round(prediccion_final * (promedio_bidones + desviacion_bidones))
            prediccion_bidones_media = round(prediccion_final * promedio_bidones)
        else:
            # Valores por defecto si no hay datos de bidones
            prediccion_bidones_min = prediccion_final
            prediccion_bidones_max = prediccion_final * 3
            prediccion_bidones_media = prediccion_final * 2
        
        # Calcular efectividad estimada
        variabilidad = desviacion / media if media > 0 else 0
        efectividad_base = 85
        if variabilidad < 0.3:
            efectividad_base += 10
        elif variabilidad > 0.5:
            efectividad_base -= 15
        
        efectividad_final = max(60, min(95, efectividad_base))
        
        # Generar recomendaciones
        recomendaciones = []
        if prediccion_final > 15:
            recomendaciones.append("Alto volumen esperado: Considerar refuerzo de personal")
        elif prediccion_final < 5:
            recomendaciones.append("Bajo volumen: Optimizar rutas de entrega")
        
        if es_finde:
            recomendaciones.append("Fin de semana: Demanda típicamente menor")
        
        # Recomendaciones basadas en bidones
        if 'promedio_bidones_por_pedido' in estadisticas_bidones:
            promedio_bidones = estadisticas_bidones['promedio_bidones_por_pedido']
            if prediccion_bidones_media > 30:
                recomendaciones.append(f"Preparar {prediccion_bidones_media} bidones: Alto volumen de inventario requerido")
            elif prediccion_bidones_media < 10:
                recomendaciones.append(f"Preparar {prediccion_bidones_media} bidones: Volumen estándar")
            
            if promedio_bidones > 3:
                recomendaciones.append(f"Promedio alto de {promedio_bidones} bidones/pedido: Considerar vehículos de mayor capacidad")
            elif promedio_bidones < 2:
                recomendaciones.append(f"Promedio bajo de {promedio_bidones} bidones/pedido: Optimizar rutas para múltiples entregas")
        
        if not recomendaciones:
            recomendaciones.append("Operación estándar recomendada")
        
        return {
            "fecha": fecha,
            "tipo_cliente": tipo_cliente,
            "prediccion": prediccion_final,
            "rango_confianza": [rango_inferior, rango_superior],
            "nivel_confianza": 75,
            "es_anomalia": False,
            "prediccion_bidones": {
                "rango_estimado": [prediccion_bidones_min, prediccion_bidones_max],
                "valor_medio": prediccion_bidones_media,
                "promedio_por_pedido": estadisticas_bidones.get('promedio_bidones_por_pedido', 2.0),
                "factor_conversion": estadisticas_bidones.get('promedio_bidones_por_pedido', 2.0)
            },
            "factores": {
                "base": mediana,
                "factor_tipo": factor_tipo,
                "factor_finde": factor_finde,
                "factor_estacional": factor_estacional,
                "dia_semana": fecha_dt.weekday()
            },
            "estadisticas_base": {
                "media": media,
                "mediana": mediana,
                "desviacion": desviacion,
                "total_dias": len(pedidos_por_dia)
            },
            "estadisticas_bidones": estadisticas_bidones,
            "anomalias_detectadas": 0,
            "analisis_vip": {
                "total_vip": 0,
                "probabilidad_alta": 0,
                "probabilidad_media": 0,
                "probabilidad_baja": 0,
                "clientes_destacados": [],
                "factor_vip": 1.25
            },
            "variables_exogenas": {
                "es_feriado": es_finde,
                "es_finde": es_finde,
                "factor_estacional": factor_estacional,
                "mes": mes,
                "dia_semana": fecha_dt.weekday(),
                "variables_personalizadas": {}
            },
            "efectividad_estimada": efectividad_final,
            "recomendaciones": recomendaciones,
            "tendencia": {
                "tendencia": "estable",
                "factor": 1.0,
                "pendiente": 0
            }
        }
        
    except Exception as e:
        return {"error": f"Error en predictor: {str(e)}"}

if __name__ == "__main__":
    # Probar el predictor
    resultado = get_predictor_simple("2025-01-20", "vip")
    print("Resultado:", resultado) 