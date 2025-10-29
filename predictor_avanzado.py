import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple

class PredictorAvanzado:
    def __init__(self):
        self.factores_entrenados = self.cargar_factores_entrenados()
        self.datos_historicos = self.obtener_datos_historicos()
        
    def cargar_factores_entrenados(self) -> Dict:
        """Carga los factores entrenados"""
        try:
            with open('factores_entrenados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def obtener_datos_historicos(self) -> List:
        """Obtiene datos históricos para análisis avanzado"""
        try:
            response = requests.get("https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php", 
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            pedidos = response.json()
            
            # Filtrar Aguas Ancud y procesar fechas
            pedidos_ancud = []
            for pedido in pedidos:
                if pedido.get('nombrelocal') == 'Aguas Ancud':
                    try:
                        fecha_str = pedido.get('fecha', '')
                        if fecha_str:
                            fecha = datetime.strptime(fecha_str, "%d-%m-%Y")
                            pedido['fecha_parsed'] = fecha
                            pedidos_ancud.append(pedido)
                    except:
                        continue
            
            return pedidos_ancud
        except:
            return []
    
    def calcular_factor_estacional(self, fecha: datetime) -> float:
        """Calcula factor estacional basado en la época del año"""
        mes = fecha.month
        
        # Factores estacionales basados en datos reales
        factores_estacionales = {
            1: 1.1,   # Enero - Verano, alta demanda
            2: 1.05,  # Febrero - Verano
            3: 1.0,   # Marzo - Fin de verano
            4: 0.95,  # Abril - Otoño
            5: 1.0,   # Mayo - Otoño
            6: 0.95,  # Junio - Invierno
            7: 1.0,   # Julio - Invierno
            8: 1.05,  # Agosto - Invierno
            9: 1.0,   # Septiembre - Primavera
            10: 1.05, # Octubre - Primavera
            11: 1.1,  # Noviembre - Primavera
            12: 1.15  # Diciembre - Verano, fiestas
        }
        
        return factores_estacionales.get(mes, 1.0)
    
    def calcular_factor_clima(self, fecha: datetime) -> float:
        """Calcula factor climático (simulado)"""
        # En una implementación real, se conectaría con API del clima
        mes = fecha.month
        
        # Simulación de factores climáticos
        if mes in [12, 1, 2]:  # Verano
            return 1.15  # Mayor demanda por calor
        elif mes in [6, 7, 8]:  # Invierno
            return 0.95  # Menor demanda
        else:
            return 1.0
    
    def calcular_factor_eventos(self, fecha: datetime) -> float:
        """Calcula factor por eventos especiales"""
        # Eventos que pueden afectar la demanda
        eventos = {
            (12, 25): 0.8,   # Navidad - menor demanda
            (12, 31): 0.7,   # Año nuevo - menor demanda
            (1, 1): 0.6,     # Año nuevo - menor demanda
            (9, 18): 1.2,    # Fiestas patrias - mayor demanda
            (5, 1): 0.9,     # Día del trabajo
        }
        
        return eventos.get((fecha.month, fecha.day), 1.0)
    
    def calcular_factor_tendencia_avanzada(self, fecha: datetime) -> float:
        """Calcula factor de tendencia más sofisticado"""
        if not self.factores_entrenados:
            return 1.0
        
        factor_base = self.factores_entrenados.get('factor_tendencia', 1.0)
        
        # Ajustar tendencia según la distancia temporal
        dias_desde_entrenamiento = (fecha - datetime.fromisoformat(
            self.factores_entrenados['fecha_entrenamiento'].split('T')[0]
        )).days
        
        # La tendencia se mantiene por 30 días, luego se estabiliza
        if dias_desde_entrenamiento <= 30:
            return factor_base
        else:
            # Gradualmente estabilizar la tendencia
            factor_ajuste = 1.0 + (factor_base - 1.0) * 0.8
            return round(factor_ajuste, 3)
    
    def calcular_factor_cliente_tipo(self, tipo_cliente: str = "residencial") -> float:
        """Calcula factor según tipo de cliente"""
        factores_cliente = {
            'residencial': 1.0,
            'recurrente': 1.2,
            'nuevo': 0.8,
            'empresa': 1.1,
            'vip': 1.25
        }
        
        return factores_cliente.get(tipo_cliente, 1.0)
    
    def calcular_factor_semana_mes(self, fecha: datetime) -> float:
        """Calcula factor según la semana del mes"""
        semana_mes = (fecha.day - 1) // 7 + 1
        
        # Patrones típicos: primera semana (pagos), última semana (mayor demanda)
        if semana_mes == 1:
            return 0.95  # Primera semana - menor demanda
        elif semana_mes == 4:
            return 1.05  # Última semana - mayor demanda
        else:
            return 1.0
    
    def calcular_factor_feriados(self, fecha: datetime) -> float:
        """Calcula factor por feriados"""
        # Feriados chilenos (simplificado)
        feriados = [
            (1, 1),   # Año nuevo
            (5, 1),   # Día del trabajo
            (9, 18),  # Fiestas patrias
            (9, 19),  # Glorias del ejército
            (12, 25), # Navidad
        ]
        
        if (fecha.month, fecha.day) in feriados:
            return 0.7  # Menor demanda en feriados
        else:
            return 1.0
    
    def predecir_demanda_avanzada(self, fecha: datetime, tipo_cliente: str = "residencial") -> Dict:
        """Predicción avanzada con múltiples factores"""
        try:
            # Base de predicción
            media_base = self.factores_entrenados.get('estadisticas_generales', {}).get('media', 6.39)
            
            # Factores múltiples
            factor_dia = self.factores_entrenados.get('factores_dia_semana', {}).get(
                ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][fecha.weekday()], 1.0
            )
            
            factor_estacional = self.calcular_factor_estacional(fecha)
            factor_clima = self.calcular_factor_clima(fecha)
            factor_eventos = self.calcular_factor_eventos(fecha)
            factor_tendencia = self.calcular_factor_tendencia_avanzada(fecha)
            factor_cliente = self.calcular_factor_cliente_tipo(tipo_cliente)
            factor_semana = self.calcular_factor_semana_mes(fecha)
            factor_feriados = self.calcular_factor_feriados(fecha)
            
            # Factor combinado
            factor_total = (factor_dia * factor_estacional * factor_clima * 
                          factor_eventos * factor_tendencia * factor_cliente * 
                          factor_semana * factor_feriados)
            
            # Predicción base
            prediccion_base = media_base * factor_total
            
            # Calcular rango de confianza
            desviacion = self.factores_entrenados.get('estadisticas_generales', {}).get('desviacion_estandar', 2.94)
            rango_min = max(0, prediccion_base - desviacion)
            rango_max = prediccion_base + desviacion
            
            # Predicción de bidones
            precio_promedio = 2000  # Precio por bidón
            bidones_estimados = prediccion_base * 1.5  # Factor de conversión
            rango_bidones_min = max(0, bidones_estimados - (desviacion * 1.5))
            rango_bidones_max = bidones_estimados + (desviacion * 1.5)
            
            return {
                "fecha_prediccion": fecha.strftime("%d-%m-%Y"),
                "prediccion_pedidos": {
                    "valor_esperado": round(prediccion_base, 1),
                    "rango_min": round(rango_min, 1),
                    "rango_max": round(rango_max, 1),
                    "confianza": self.factores_entrenados.get('efectividad_estimada', 65)
                },
                "prediccion_bidones": {
                    "valor_esperado": round(bidones_estimados, 1),
                    "rango_min": round(rango_bidones_min, 1),
                    "rango_max": round(rango_bidones_max, 1),
                    "precio_estimado": round(bidones_estimados * precio_promedio, 0)
                },
                "factores_aplicados": {
                    "dia_semana": factor_dia,
                    "estacional": factor_estacional,
                    "clima": factor_clima,
                    "eventos": factor_eventos,
                    "tendencia": factor_tendencia,
                    "tipo_cliente": factor_cliente,
                    "semana_mes": factor_semana,
                    "feriados": factor_feriados,
                    "factor_total": round(factor_total, 3)
                },
                "recomendaciones": self.generar_recomendaciones_avanzadas(fecha, prediccion_base, factor_total)
            }
            
        except Exception as e:
            return {"error": f"Error en predicción avanzada: {str(e)}"}
    
    def generar_recomendaciones_avanzadas(self, fecha: datetime, prediccion: float, factor_total: float) -> List[str]:
        """Genera recomendaciones avanzadas basadas en la predicción"""
        recomendaciones = []
        
        # Análisis de demanda
        if prediccion > 8:
            recomendaciones.append("🔥 Alta demanda esperada: Preparar inventario extra")
        elif prediccion < 4:
            recomendaciones.append("📉 Baja demanda esperada: Considerar promociones")
        
        # Análisis de factores
        if factor_total > 1.2:
            recomendaciones.append("📈 Factores favorables: Buena oportunidad para ventas")
        elif factor_total < 0.8:
            recomendaciones.append("⚠️ Factores desfavorables: Planificar estrategias de compensación")
        
        # Recomendaciones específicas por día
        dia_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][fecha.weekday()]
        if dia_semana == 'Lunes':
            recomendaciones.append("📅 Lunes: Preparar para alta demanda típica")
        elif dia_semana == 'Sábado':
            recomendaciones.append("📅 Sábado: Demanda reducida, planificar actividades alternativas")
        
        # Recomendaciones por época
        mes = fecha.month
        if mes in [12, 1, 2]:
            recomendaciones.append("☀️ Verano: Aumentar capacidad de producción")
        elif mes in [6, 7, 8]:
            recomendaciones.append("❄️ Invierno: Mantener inventario estable")
        
        return recomendaciones
    
    def predecir_tendencia_futura(self, dias_futuros: int = 30) -> Dict:
        """Predice tendencia para los próximos días"""
        try:
            predicciones = []
            fecha_actual = datetime.now()
            
            for i in range(dias_futuros):
                fecha_futura = fecha_actual + timedelta(days=i)
                prediccion = self.predecir_demanda_avanzada(fecha_futura)
                
                if "error" not in prediccion:
                    predicciones.append({
                        "fecha": fecha_futura.strftime("%d-%m-%Y"),
                        "pedidos": prediccion["prediccion_pedidos"]["valor_esperado"],
                        "bidones": prediccion["prediccion_bidones"]["valor_esperado"]
                    })
            
            if predicciones:
                # Calcular tendencia
                pedidos = [p["pedidos"] for p in predicciones]
                bidones = [p["bidones"] for p in predicciones]
                
                tendencia_pedidos = "creciente" if pedidos[-1] > pedidos[0] else "decreciente" if pedidos[-1] < pedidos[0] else "estable"
                tendencia_bidones = "creciente" if bidones[-1] > bidones[0] else "decreciente" if bidones[-1] < bidones[0] else "estable"
                
                return {
                    "periodo_prediccion": f"{dias_futuros} días",
                    "predicciones_diarias": predicciones,
                    "resumen": {
                        "promedio_pedidos": round(sum(pedidos) / len(pedidos), 1),
                        "promedio_bidones": round(sum(bidones) / len(bidones), 1),
                        "tendencia_pedidos": tendencia_pedidos,
                        "tendencia_bidones": tendencia_bidones
                    }
                }
            
            return {"error": "No se pudieron generar predicciones"}
            
        except Exception as e:
            return {"error": f"Error en predicción de tendencia: {str(e)}"}

# Función para probar el predictor avanzado
def probar_predictor_avanzado():
    """Función para probar el predictor avanzado"""
    predictor = PredictorAvanzado()
    
    # Predicción para mañana
    fecha_mañana = datetime.now() + timedelta(days=1)
    prediccion = predictor.predecir_demanda_avanzada(fecha_mañana)
    
    print("🚀 PREDICTOR AVANZADO - PRUEBA")
    print("=" * 50)
    
    if "error" not in prediccion:
        print(f"📅 Fecha: {prediccion['fecha_prediccion']}")
        print(f"📦 Pedidos esperados: {prediccion['prediccion_pedidos']['valor_esperado']}")
        print(f"🛢️ Bidones esperados: {prediccion['prediccion_bidones']['valor_esperado']}")
        print(f"💰 Ingresos estimados: ${prediccion['prediccion_bidones']['precio_estimado']:,}")
        print(f"🎯 Confianza: {prediccion['prediccion_pedidos']['confianza']}%")
        
        print(f"\n📊 FACTORES APLICADOS:")
        for factor, valor in prediccion['factores_aplicados'].items():
            print(f"   • {factor}: {valor}")
        
        print(f"\n💡 RECOMENDACIONES:")
        for rec in prediccion['recomendaciones']:
            print(f"   • {rec}")
    else:
        print(f"❌ Error: {prediccion['error']}")

if __name__ == "__main__":
    probar_predictor_avanzado() 