import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional
import pandas as pd

class PredictorTracker:
    """Sistema de tracking para monitorear efectividad del predictor"""
    
    def __init__(self, data_file: str = "predictor_tracking.json"):
        self.data_file = data_file
        self.tracking_data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Carga datos de tracking existentes"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"predicciones": [], "metricas": {}}
        return {"predicciones": [], "metricas": {}}
    
    def _save_data(self):
        """Guarda datos de tracking"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.tracking_data, f, indent=2, ensure_ascii=False)
    
    def registrar_prediccion(self, fecha: str, tipo_cliente: str, prediccion: int, 
                           factores: Dict, efectividad_estimada: float):
        """Registra una nueva predicción"""
        prediccion_data = {
            "fecha": fecha,
            "tipo_cliente": tipo_cliente,
            "prediccion": prediccion,
            "factores": factores,
            "efectividad_estimada": efectividad_estimada,
            "fecha_registro": datetime.now().isoformat(),
            "pedidos_reales": None,  # Se llenará después
            "error_porcentual": None,
            "efectividad_real": None,
            "verificada": False
        }
        
        # Verificar si ya existe una predicción para esta fecha y tipo
        for i, pred in enumerate(self.tracking_data["predicciones"]):
            if pred["fecha"] == fecha and pred["tipo_cliente"] == tipo_cliente:
                self.tracking_data["predicciones"][i] = prediccion_data
                break
        else:
            self.tracking_data["predicciones"].append(prediccion_data)
        
        self._save_data()
        print(f"✅ Predicción registrada: {fecha} - {tipo_cliente} - {prediccion} pedidos")
    
    def registrar_pedidos_reales(self, fecha: str, pedidos_reales: int, tipo_cliente: str = "general"):
        """Registra los pedidos reales para una fecha"""
        fecha_iso = fecha if "-" in fecha else datetime.strptime(fecha, "%d-%m-%Y").strftime("%Y-%m-%d")
        
        # Buscar predicciones para esta fecha
        predicciones_actualizadas = 0
        for prediccion in self.tracking_data["predicciones"]:
            if prediccion["fecha"] == fecha_iso and not prediccion["verificada"]:
                prediccion["pedidos_reales"] = pedidos_reales
                prediccion["verificada"] = True
                
                # Calcular error
                if prediccion["prediccion"] > 0 and pedidos_reales > 0:
                    error = abs(prediccion["prediccion"] - pedidos_reales) / pedidos_reales * 100
                    prediccion["error_porcentual"] = round(error, 1)
                    
                    # Calcular efectividad real
                    if error <= 15:
                        efectividad = 100
                    elif error <= 30:
                        efectividad = 75
                    elif error <= 50:
                        efectividad = 50
                    else:
                        efectividad = 25
                    
                    prediccion["efectividad_real"] = efectividad
                    predicciones_actualizadas += 1
        
        self._save_data()
        print(f"✅ Pedidos reales registrados: {fecha} - {pedidos_reales} pedidos ({predicciones_actualizadas} predicciones actualizadas)")
    
    def obtener_metricas(self) -> Dict:
        """Calcula métricas de efectividad del predictor"""
        predicciones_verificadas = [p for p in self.tracking_data["predicciones"] if p["verificada"]]
        
        if not predicciones_verificadas:
            return {
                "total_predicciones": 0,
                "predicciones_verificadas": 0,
                "error_promedio": 0,
                "efectividad_promedio": 0,
                "predicciones_excelentes": 0,
                "predicciones_buenas": 0,
                "predicciones_aceptables": 0,
                "predicciones_pobres": 0
            }
        
        errores = [p["error_porcentual"] for p in predicciones_verificadas if p["error_porcentual"] is not None]
        efectividades = [p["efectividad_real"] for p in predicciones_verificadas if p["efectividad_real"] is not None]
        
        # Clasificar predicciones
        excelentes = sum(1 for e in errores if e <= 15)
        buenas = sum(1 for e in errores if 15 < e <= 30)
        aceptables = sum(1 for e in errores if 30 < e <= 50)
        pobres = sum(1 for e in errores if e > 50)
        
        metricas = {
            "total_predicciones": len(self.tracking_data["predicciones"]),
            "predicciones_verificadas": len(predicciones_verificadas),
            "error_promedio": round(sum(errores) / len(errores), 1) if errores else 0,
            "efectividad_promedio": round(sum(efectividades) / len(efectividades), 1) if efectividades else 0,
            "predicciones_excelentes": excelentes,
            "predicciones_buenas": buenas,
            "predicciones_aceptables": aceptables,
            "predicciones_pobres": pobres,
            "porcentaje_excelentes": round(excelentes / len(errores) * 100, 1) if errores else 0,
            "porcentaje_buenas": round(buenas / len(errores) * 100, 1) if errores else 0,
            "porcentaje_aceptables": round(aceptables / len(errores) * 100, 1) if errores else 0,
            "porcentaje_pobres": round(pobres / len(errores) * 100, 1) if errores else 0
        }
        
        # Guardar métricas
        self.tracking_data["metricas"] = metricas
        self._save_data()
        
        return metricas
    
    def obtener_ultimas_predicciones(self, dias: int = 7) -> List[Dict]:
        """Obtiene las últimas predicciones"""
        predicciones_ordenadas = sorted(
            self.tracking_data["predicciones"], 
            key=lambda x: x["fecha"], 
            reverse=True
        )
        return predicciones_ordenadas[:dias]
    
    def generar_reporte_diario(self) -> Dict:
        """Genera un reporte diario de efectividad"""
        metricas = self.obtener_metricas()
        ultimas_predicciones = self.obtener_ultimas_predicciones(7)
        
        return {
            "fecha_reporte": datetime.now().isoformat(),
            "metricas_generales": metricas,
            "ultimas_predicciones": ultimas_predicciones,
            "recomendaciones": self._generar_recomendaciones(metricas)
        }
    
    def _generar_recomendaciones(self, metricas: Dict) -> List[str]:
        """Genera recomendaciones basadas en las métricas"""
        recomendaciones = []
        
        if metricas["efectividad_promedio"] < 50:
            recomendaciones.append("⚠️ Efectividad baja: Considerar recalibración de factores")
        
        if metricas["porcentaje_pobres"] > 30:
            recomendaciones.append("⚠️ Muchas predicciones pobres: Revisar modelo base")
        
        if metricas["predicciones_verificadas"] < 10:
            recomendaciones.append("📊 Necesitas más datos: Continuar monitoreo diario")
        
        if metricas["efectividad_promedio"] > 70:
            recomendaciones.append("✅ Excelente rendimiento: Sistema funcionando bien")
        
        if not recomendaciones:
            recomendaciones.append("📈 Rendimiento estable: Continuar monitoreo")
        
        return recomendaciones

# Instancia global del tracker
predictor_tracker = PredictorTracker() 