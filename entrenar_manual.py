import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict

def obtener_datos_reales():
    """Obtiene datos reales de los últimos 3 meses"""
    try:
        response = requests.get("https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php", 
                              headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        pedidos = response.json()
        
        # Filtrar Aguas Ancud
        pedidos_ancud = [p for p in pedidos if p.get('nombrelocal') == 'Aguas Ancud']
        
        if not pedidos_ancud:
            return {"error": "No hay datos de Aguas Ancud"}
        
        return pedidos_ancud
        
    except Exception as e:
        return {"error": f"Error obteniendo datos: {str(e)}"}

def procesar_datos_manual(pedidos):
    """Procesa los datos de forma manual"""
    try:
        # Filtrar últimos 3 meses
        fecha_limite = datetime.now() - timedelta(days=90)
        pedidos_recientes = []
        
        for pedido in pedidos:
            try:
                fecha_str = pedido.get('fecha', '')
                if fecha_str:
                    fecha = datetime.strptime(fecha_str, "%d-%m-%Y")
                    if fecha >= fecha_limite:
                        pedido['fecha_parsed'] = fecha
                        pedidos_recientes.append(pedido)
            except:
                continue
        
        if not pedidos_recientes:
            return {"error": "No hay datos de los últimos 3 meses"}
        
        return pedidos_recientes
        
    except Exception as e:
        return {"error": f"Error procesando datos: {str(e)}"}

def calcular_estadisticas_manual(pedidos):
    """Calcula estadísticas de forma manual"""
    try:
        # Agrupar por fecha
        pedidos_por_fecha = defaultdict(list)
        for pedido in pedidos:
            fecha = pedido['fecha_parsed'].date()
            pedidos_por_fecha[fecha].append(pedido)
        
        # Calcular pedidos por día
        pedidos_por_dia = {fecha: len(pedidos_list) for fecha, pedidos_list in pedidos_por_fecha.items()}
        
        # Estadísticas generales
        valores = list(pedidos_por_dia.values())
        total_dias = len(valores)
        total_pedidos = sum(valores)
        
        if total_dias == 0:
            return {"error": "No hay datos válidos"}
        
        media = total_pedidos / total_dias
        valores_ordenados = sorted(valores)
        mediana = valores_ordenados[total_dias // 2] if total_dias % 2 == 1 else (valores_ordenados[total_dias // 2 - 1] + valores_ordenados[total_dias // 2]) / 2
        
        # Desviación estándar
        varianza = sum((x - media) ** 2 for x in valores) / total_dias
        desviacion = varianza ** 0.5
        
        # Factores por día de semana
        factores_dia = defaultdict(list)
        for fecha, cantidad in pedidos_por_dia.items():
            dia_semana = fecha.weekday()
            factores_dia[dia_semana].append(cantidad)
        
        factores_dia_semana = {}
        dias_nombre = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        for dia in range(7):
            if dia in factores_dia:
                media_dia = sum(factores_dia[dia]) / len(factores_dia[dia])
                factor = media_dia / media if media > 0 else 1.0
                factores_dia_semana[dias_nombre[dia]] = round(factor, 3)
            else:
                factores_dia_semana[dias_nombre[dia]] = 1.0
        
        # Factores por mes
        factores_mes = defaultdict(list)
        for fecha, cantidad in pedidos_por_dia.items():
            mes = fecha.month
            factores_mes[mes].append(cantidad)
        
        factores_mes_final = {}
        meses_nombre = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                       'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        
        for mes in range(1, 13):
            if mes in factores_mes:
                media_mes = sum(factores_mes[mes]) / len(factores_mes[mes])
                factor = media_mes / media if media > 0 else 1.0
                factores_mes_final[meses_nombre[mes-1]] = round(factor, 3)
            else:
                factores_mes_final[meses_nombre[mes-1]] = 1.0
        
        # Análisis de tendencias
        fechas_ordenadas = sorted(pedidos_por_dia.keys())
        factor_tendencia = 1.0
        
        if len(fechas_ordenadas) >= 14:
            primera_semana = fechas_ordenadas[:7]
            ultima_semana = fechas_ordenadas[-7:]
            
            promedio_primera = sum(pedidos_por_dia[fecha] for fecha in primera_semana) / 7
            promedio_ultima = sum(pedidos_por_dia[fecha] for fecha in ultima_semana) / 7
            
            if promedio_primera > 0:
                factor_tendencia = promedio_ultima / promedio_primera
                factor_tendencia = round(factor_tendencia, 3)
        
        # Efectividad estimada
        coeficiente_variacion = desviacion / media if media > 0 else 0
        
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
        
        # Recomendaciones
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
        
        # Análisis de días
        for dia, factor in factores_dia_semana.items():
            if factor > 1.2:
                recomendaciones.append(f"📊 {dia}: Alta demanda (factor {factor})")
            elif factor < 0.8:
                recomendaciones.append(f"📊 {dia}: Baja demanda (factor {factor})")
        
        return {
            "fecha_entrenamiento": datetime.now().isoformat(),
            "periodo_analisis": "3 meses",
            "total_dias_analizados": total_dias,
            "total_pedidos_analizados": total_pedidos,
            "estadisticas_generales": {
                "media": round(media, 2),
                "mediana": round(mediana, 2),
                "desviacion_estandar": round(desviacion, 2),
                "coeficiente_variacion": round(coeficiente_variacion, 3)
            },
            "factores_dia_semana": factores_dia_semana,
            "factores_mes": factores_mes_final,
            "factor_tendencia": factor_tendencia,
            "efectividad_estimada": efectividad,
            "recomendaciones_entrenamiento": recomendaciones
        }
        
    except Exception as e:
        return {"error": f"Error calculando estadísticas: {str(e)}"}

def guardar_factores(factores):
    """Guarda los factores en un archivo JSON"""
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
    
    # Obtener datos
    print("📊 Obteniendo datos de los últimos 3 meses...")
    pedidos = obtener_datos_reales()
    
    if isinstance(pedidos, dict) and "error" in pedidos:
        print(f"❌ Error: {pedidos['error']}")
        return
    
    print(f"✅ Datos obtenidos: {len(pedidos)} pedidos totales")
    
    # Procesar datos
    print("\n🔧 Procesando datos...")
    pedidos_procesados = procesar_datos_manual(pedidos)
    
    if isinstance(pedidos_procesados, dict) and "error" in pedidos_procesados:
        print(f"❌ Error: {pedidos_procesados['error']}")
        return
    
    print(f"✅ Datos procesados: {len(pedidos_procesados)} pedidos en los últimos 3 meses")
    
    # Calcular estadísticas
    print("\n📈 Calculando estadísticas...")
    factores = calcular_estadisticas_manual(pedidos_procesados)
    
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
    guardar_factores(factores)
    
    print("\n✅ ¡Entrenamiento completado exitosamente!")
    print("🎉 El predictor ahora está optimizado con datos reales")

if __name__ == "__main__":
    entrenar_predictor() 