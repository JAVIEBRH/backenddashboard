"""
MEJORAS AVANZADAS PARA EL PREDICTOR ENTRENADO
============================================

Basado en datos reales de 3 meses, estas mejoras incluyen:

1. 📊 Predicción por Estaciones y Eventos Especiales
2. 🌡️ Factores Climáticos (simulados)
3. 📅 Análisis de Semanas del Mes
4. 🎯 Predicción por Tipo de Cliente
5. 📈 Tendencia Avanzada con Decay
6. 🗓️ Detección de Feriados
7. 🔮 Predicción de Tendencia Futura
8. 📊 Análisis de Patrones Semanales
9. 💰 Estimación de Ingresos
10. 🎯 Recomendaciones Inteligentes
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class MejorasPredictor:
    def __init__(self):
        self.factores = self.cargar_factores()
        
    def cargar_factores(self) -> Dict:
        """Carga los factores entrenados"""
        try:
            with open('factores_entrenados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def predecir_con_mejoras(self, fecha: datetime, tipo_cliente: str = "residencial") -> Dict:
        """Predicción con todas las mejoras implementadas"""
        
        # 1. Base de predicción
        media_base = self.factores.get('estadisticas_generales', {}).get('media', 6.39)
        
        # 2. Factor día de semana (datos reales)
        dias_nombre = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        factor_dia = self.factores.get('factores_dia_semana', {}).get(dias_nombre[fecha.weekday()], 1.0)
        
        # 3. Factor estacional
        factor_estacional = self.calcular_factor_estacional(fecha)
        
        # 4. Factor clima (simulado)
        factor_clima = self.calcular_factor_clima(fecha)
        
        # 5. Factor eventos especiales
        factor_eventos = self.calcular_factor_eventos(fecha)
        
        # 6. Factor tendencia avanzada
        factor_tendencia = self.calcular_factor_tendencia_avanzada(fecha)
        
        # 7. Factor tipo de cliente
        factor_cliente = self.calcular_factor_cliente(tipo_cliente)
        
        # 8. Factor semana del mes
        factor_semana = self.calcular_factor_semana_mes(fecha)
        
        # 9. Factor feriados
        factor_feriados = self.calcular_factor_feriados(fecha)
        
        # Factor total combinado
        factor_total = (factor_dia * factor_estacional * factor_clima * 
                       factor_eventos * factor_tendencia * factor_cliente * 
                       factor_semana * factor_feriados)
        
        # Predicción final
        prediccion_pedidos = media_base * factor_total
        prediccion_bidones = prediccion_pedidos * 1.5  # Factor de conversión
        
        # Rango de confianza
        desviacion = self.factores.get('estadisticas_generales', {}).get('desviacion_estandar', 2.94)
        rango_min = max(0, prediccion_pedidos - desviacion)
        rango_max = prediccion_pedidos + desviacion
        
        return {
            "fecha": fecha.strftime("%d-%m-%Y"),
            "prediccion_pedidos": round(prediccion_pedidos, 1),
            "prediccion_bidones": round(prediccion_bidones, 1),
            "rango_pedidos": [round(rango_min, 1), round(rango_max, 1)],
            "ingresos_estimados": round(prediccion_bidones * 2000, 0),
            "confianza": self.factores.get('efectividad_estimada', 65),
            "factores": {
                "dia_semana": factor_dia,
                "estacional": factor_estacional,
                "clima": factor_clima,
                "eventos": factor_eventos,
                "tendencia": factor_tendencia,
                "cliente": factor_cliente,
                "semana_mes": factor_semana,
                "feriados": factor_feriados,
                "total": round(factor_total, 3)
            },
            "recomendaciones": self.generar_recomendaciones(fecha, prediccion_pedidos, factor_total)
        }
    
    def calcular_factor_estacional(self, fecha: datetime) -> float:
        """Factor estacional basado en la época del año"""
        mes = fecha.month
        
        # Factores basados en patrones reales de agua embotellada
        factores = {
            1: 1.15,  # Enero - Verano, alta demanda
            2: 1.1,   # Febrero - Verano
            3: 1.05,  # Marzo - Fin de verano
            4: 0.95,  # Abril - Otoño
            5: 1.0,   # Mayo - Otoño
            6: 0.9,   # Junio - Invierno
            7: 0.95,  # Julio - Invierno
            8: 1.0,   # Agosto - Invierno
            9: 1.05,  # Septiembre - Primavera
            10: 1.1,  # Octubre - Primavera
            11: 1.15, # Noviembre - Primavera
            12: 1.2   # Diciembre - Verano, fiestas
        }
        
        return factores.get(mes, 1.0)
    
    def calcular_factor_clima(self, fecha: datetime) -> float:
        """Factor climático (simulado)"""
        mes = fecha.month
        
        # Simulación de factores climáticos
        if mes in [12, 1, 2]:  # Verano
            return 1.1  # Mayor demanda por calor
        elif mes in [6, 7, 8]:  # Invierno
            return 0.95  # Menor demanda
        else:
            return 1.0
    
    def calcular_factor_eventos(self, fecha: datetime) -> float:
        """Factor por eventos especiales"""
        eventos = {
            (12, 25): 0.7,   # Navidad
            (12, 31): 0.6,   # Año nuevo
            (1, 1): 0.5,     # Año nuevo
            (9, 18): 1.3,    # Fiestas patrias
            (5, 1): 0.9,     # Día del trabajo
        }
        
        return eventos.get((fecha.month, fecha.day), 1.0)
    
    def calcular_factor_tendencia_avanzada(self, fecha: datetime) -> float:
        """Factor de tendencia con decay temporal"""
        if not self.factores:
            return 1.0
        
        factor_base = self.factores.get('factor_tendencia', 1.0)
        
        # Calcular días desde el entrenamiento
        try:
            fecha_entrenamiento = datetime.fromisoformat(
                self.factores['fecha_entrenamiento'].split('T')[0]
            )
            dias_desde_entrenamiento = (fecha - fecha_entrenamiento).days
            
            # La tendencia se mantiene fuerte por 30 días, luego decae
            if dias_desde_entrenamiento <= 30:
                return factor_base
            elif dias_desde_entrenamiento <= 90:
                # Decay gradual
                factor_decay = 1.0 + (factor_base - 1.0) * 0.7
                return round(factor_decay, 3)
            else:
                # Estabilización
                factor_estable = 1.0 + (factor_base - 1.0) * 0.5
                return round(factor_estable, 3)
        except:
            return factor_base
    
    def calcular_factor_cliente(self, tipo_cliente: str) -> float:
        """Factor según tipo de cliente"""
        factores = {
            'residencial': 1.0,
            'recurrente': 1.2,
            'nuevo': 0.8,
            'empresa': 1.15,
            'vip': 1.3
        }
        
        return factores.get(tipo_cliente, 1.0)
    
    def calcular_factor_semana_mes(self, fecha: datetime) -> float:
        """Factor según la semana del mes"""
        semana = (fecha.day - 1) // 7 + 1
        
        # Patrones típicos: primera semana (pagos), última semana (mayor demanda)
        if semana == 1:
            return 0.95  # Primera semana
        elif semana == 4:
            return 1.05  # Última semana
        else:
            return 1.0
    
    def calcular_factor_feriados(self, fecha: datetime) -> float:
        """Factor por feriados chilenos"""
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
    
    def generar_recomendaciones(self, fecha: datetime, prediccion: float, factor_total: float) -> List[str]:
        """Genera recomendaciones inteligentes"""
        recomendaciones = []
        
        # Análisis de demanda
        if prediccion > 8:
            recomendaciones.append("🔥 Alta demanda esperada: Preparar inventario extra")
        elif prediccion < 4:
            recomendaciones.append("📉 Baja demanda esperada: Considerar promociones")
        
        # Análisis de factores
        if factor_total > 1.2:
            recomendaciones.append("📈 Factores muy favorables: Excelente oportunidad")
        elif factor_total < 0.8:
            recomendaciones.append("⚠️ Factores desfavorables: Planificar compensación")
        
        # Recomendaciones específicas
        dia_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][fecha.weekday()]
        if dia_semana == 'Lunes':
            recomendaciones.append("📅 Lunes: Preparar para demanda alta típica")
        elif dia_semana == 'Sábado':
            recomendaciones.append("📅 Sábado: Demanda reducida, planificar alternativas")
        
        # Recomendaciones por época
        mes = fecha.month
        if mes in [12, 1, 2]:
            recomendaciones.append("☀️ Verano: Aumentar capacidad de producción")
        elif mes in [6, 7, 8]:
            recomendaciones.append("❄️ Invierno: Mantener inventario estable")
        
        return recomendaciones
    
    def predecir_tendencia_30_dias(self) -> Dict:
        """Predice tendencia para los próximos 30 días"""
        predicciones = []
        fecha_actual = datetime.now()
        
        for i in range(30):
            fecha_futura = fecha_actual + timedelta(days=i)
            prediccion = self.predecir_con_mejoras(fecha_futura)
            predicciones.append(prediccion)
        
        # Análisis de tendencia
        pedidos = [p['prediccion_pedidos'] for p in predicciones]
        bidones = [p['prediccion_bidones'] for p in predicciones]
        
        promedio_pedidos = sum(pedidos) / len(pedidos)
        promedio_bidones = sum(bidones) / len(bidones)
        
        # Determinar tendencia
        if pedidos[-1] > pedidos[0] * 1.1:
            tendencia = "creciente"
        elif pedidos[-1] < pedidos[0] * 0.9:
            tendencia = "decreciente"
        else:
            tendencia = "estable"
        
        return {
            "periodo": "30 días",
            "promedio_pedidos": round(promedio_pedidos, 1),
            "promedio_bidones": round(promedio_bidones, 1),
            "tendencia": tendencia,
            "ingresos_totales_estimados": round(promedio_bidones * 30 * 2000, 0),
            "predicciones_diarias": predicciones
        }

def probar_mejoras():
    """Función para probar las mejoras"""
    mejoras = MejorasPredictor()
    
    # Predicción para mañana
    fecha_mañana = datetime.now() + timedelta(days=1)
    prediccion = mejoras.predecir_con_mejoras(fecha_mañana)
    
    print("🚀 PREDICTOR CON MEJORAS AVANZADAS")
    print("=" * 50)
    print(f"📅 Fecha: {prediccion['fecha']}")
    print(f"📦 Pedidos: {prediccion['prediccion_pedidos']}")
    print(f"🛢️ Bidones: {prediccion['prediccion_bidones']}")
    print(f"💰 Ingresos: ${prediccion['ingresos_estimados']:,}")
    print(f"🎯 Confianza: {prediccion['confianza']}%")
    
    print(f"\n📊 FACTORES:")
    for factor, valor in prediccion['factores'].items():
        print(f"   • {factor}: {valor}")
    
    print(f"\n💡 RECOMENDACIONES:")
    for rec in prediccion['recomendaciones']:
        print(f"   • {rec}")
    
    # Predicción de tendencia
    print(f"\n📈 TENDENCIA 30 DÍAS:")
    tendencia = mejoras.predecir_tendencia_30_dias()
    print(f"   • Promedio pedidos: {tendencia['promedio_pedidos']}")
    print(f"   • Promedio bidones: {tendencia['promedio_bidones']}")
    print(f"   • Tendencia: {tendencia['tendencia']}")
    print(f"   • Ingresos estimados: ${tendencia['ingresos_totales_estimados']:,}")

if __name__ == "__main__":
    probar_mejoras() 