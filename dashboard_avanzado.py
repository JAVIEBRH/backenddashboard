import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class DashboardAvanzado:
    def __init__(self):
        self.factores_entrenados = self.cargar_factores()
        self.datos_historicos = self.obtener_datos_historicos()
        self.configurar_graficos()
        
    def cargar_factores(self):
        """Carga los factores entrenados"""
        try:
            with open('factores_entrenados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def obtener_datos_historicos(self):
        """Obtiene datos históricos para análisis"""
        try:
            response = requests.get("https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php", 
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            pedidos = response.json()
            
            # Filtrar Aguas Ancud
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
    
    def configurar_graficos(self):
        """Configura el estilo de los gráficos"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generar_datos_analisis(self) -> Dict:
        """Genera datos para análisis del dashboard"""
        if not self.datos_historicos:
            return {}
        
        # Crear DataFrame
        df = pd.DataFrame(self.datos_historicos)
        df['fecha_parsed'] = pd.to_datetime(df['fecha_parsed'])
        df['fecha'] = df['fecha_parsed'].dt.date
        
        # Agrupar por fecha
        pedidos_por_dia = df.groupby('fecha').size().reset_index(name='pedidos')
        pedidos_por_dia['fecha'] = pd.to_datetime(pedidos_por_dia['fecha'])
        
        # Calcular métricas adicionales
        pedidos_por_dia['bidones_estimados'] = pedidos_por_dia['pedidos'] * 1.5
        pedidos_por_dia['ingresos_estimados'] = pedidos_por_dia['bidones_estimados'] * 2000
        pedidos_por_dia['dia_semana'] = pedidos_por_dia['fecha'].dt.weekday
        pedidos_por_dia['mes'] = pedidos_por_dia['fecha'].dt.month
        pedidos_por_dia['semana'] = pedidos_por_dia['fecha'].dt.isocalendar().week
        
        return {
            'df': pedidos_por_dia,
            'total_dias': len(pedidos_por_dia),
            'total_pedidos': pedidos_por_dia['pedidos'].sum(),
            'total_bidones': pedidos_por_dia['bidones_estimados'].sum(),
            'total_ingresos': pedidos_por_dia['ingresos_estimados'].sum()
        }
    
    def generar_grafico_tendencias(self) -> str:
        """Genera gráfico de tendencias temporales"""
        try:
            datos = self.generar_datos_analisis()
            if not datos:
                return ""
            
            df = datos['df']
            
            # Crear figura
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Gráfico de pedidos por día
            ax1.plot(df['fecha'], df['pedidos'], marker='o', linewidth=2, markersize=4)
            ax1.set_title('📈 Tendencia de Pedidos Diarios', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Número de Pedidos')
            ax1.grid(True, alpha=0.3)
            
            # Línea de tendencia
            z = np.polyfit(range(len(df)), df['pedidos'], 1)
            p = np.poly1d(z)
            ax1.plot(df['fecha'], p(range(len(df))), "r--", alpha=0.8, label='Tendencia')
            ax1.legend()
            
            # Gráfico de ingresos
            ax2.plot(df['fecha'], df['ingresos_estimados'], marker='s', linewidth=2, markersize=4, color='green')
            ax2.set_title('💰 Tendencia de Ingresos Diarios', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Ingresos ($)')
            ax2.set_xlabel('Fecha')
            ax2.grid(True, alpha=0.3)
            
            # Formatear eje Y para ingresos
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.tight_layout()
            
            # Convertir a base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            imagen_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return imagen_base64
            
        except Exception as e:
            print(f"❌ Error generando gráfico de tendencias: {str(e)}")
            return ""
    
    def generar_grafico_dias_semana(self) -> str:
        """Genera gráfico de demanda por día de la semana"""
        try:
            datos = self.generar_datos_analisis()
            if not datos:
                return ""
            
            df = datos['df']
            
            # Agrupar por día de semana
            dias_semana = df.groupby('dia_semana').agg({
                'pedidos': ['mean', 'std', 'count'],
                'ingresos_estimados': 'mean'
            }).round(2)
            
            dias_nombre = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            
            # Crear figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico de barras - Pedidos promedio
            x_pos = range(len(dias_nombre))
            pedidos_promedio = [dias_semana.loc[i, ('pedidos', 'mean')] for i in range(7)]
            
            bars = ax1.bar(x_pos, pedidos_promedio, color=sns.color_palette("husl", 7))
            ax1.set_title('📊 Promedio de Pedidos por Día de la Semana', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Pedidos Promedio')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(dias_nombre, rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Agregar valores en las barras
            for bar, valor in zip(bars, pedidos_promedio):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{valor:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Gráfico de ingresos promedio
            ingresos_promedio = [dias_semana.loc[i, ('ingresos_estimados', 'mean')] for i in range(7)]
            
            bars2 = ax2.bar(x_pos, ingresos_promedio, color=sns.color_palette("viridis", 7))
            ax2.set_title('💰 Ingresos Promedio por Día de la Semana', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Ingresos Promedio ($)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(dias_nombre, rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Agregar valores en las barras
            for bar, valor in zip(bars2, ingresos_promedio):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
                        f'${valor:,.0f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Convertir a base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            imagen_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return imagen_base64
            
        except Exception as e:
            print(f"❌ Error generando gráfico días semana: {str(e)}")
            return ""
    
    def generar_grafico_meses(self) -> str:
        """Genera gráfico de demanda por meses"""
        try:
            datos = self.generar_datos_analisis()
            if not datos:
                return ""
            
            df = datos['df']
            
            # Agrupar por mes
            meses_data = df.groupby('mes').agg({
                'pedidos': ['mean', 'sum'],
                'ingresos_estimados': ['mean', 'sum']
            }).round(2)
            
            meses_nombre = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            
            # Crear figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico de pedidos totales por mes
            x_pos = range(12)
            pedidos_totales = [meses_data.loc[i, ('pedidos', 'sum')] if i in meses_data.index else 0 for i in range(1, 13)]
            
            bars = ax1.bar(x_pos, pedidos_totales, color=sns.color_palette("plasma", 12))
            ax1.set_title('📅 Pedidos Totales por Mes', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Total de Pedidos')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(meses_nombre)
            ax1.grid(True, alpha=0.3)
            
            # Agregar valores en las barras
            for bar, valor in zip(bars, pedidos_totales):
                if valor > 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                            f'{valor:.0f}', ha='center', va='bottom', fontweight='bold')
            
            # Gráfico de ingresos totales por mes
            ingresos_totales = [meses_data.loc[i, ('ingresos_estimados', 'sum')] if i in meses_data.index else 0 for i in range(1, 13)]
            
            bars2 = ax2.bar(x_pos, ingresos_totales, color=sns.color_palette("magma", 12))
            ax2.set_title('💰 Ingresos Totales por Mes', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Ingresos Totales ($)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(meses_nombre)
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Agregar valores en las barras
            for bar, valor in zip(bars2, ingresos_totales):
                if valor > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                            f'${valor:,.0f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Convertir a base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            imagen_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return imagen_base64
            
        except Exception as e:
            print(f"❌ Error generando gráfico meses: {str(e)}")
            return ""
    
    def generar_analisis_rentabilidad(self) -> Dict:
        """Genera análisis de rentabilidad"""
        try:
            datos = self.generar_datos_analisis()
            if not datos:
                return {}
            
            df = datos['df']
            
            # Métricas de rentabilidad
            ingresos_totales = datos['total_ingresos']
            pedidos_totales = datos['total_pedidos']
            bidones_totales = datos['total_bidones']
            
            # Costos estimados (simulados)
            costo_bidon = 800  # Costo por bidón
            costo_operacion = 200  # Costo operacional por pedido
            costos_totales = (bidones_totales * costo_bidon) + (pedidos_totales * costo_operacion)
            
            # Cálculos de rentabilidad
            utilidad_bruta = ingresos_totales - costos_totales
            margen_bruto = (utilidad_bruta / ingresos_totales) * 100 if ingresos_totales > 0 else 0
            roi = (utilidad_bruta / costos_totales) * 100 if costos_totales > 0 else 0
            
            # Análisis por día de semana
            rentabilidad_dias = {}
            for dia in range(7):
                datos_dia = df[df['dia_semana'] == dia]
                if len(datos_dia) > 0:
                    ingresos_dia = datos_dia['ingresos_estimados'].sum()
                    bidones_dia = datos_dia['bidones_estimados'].sum()
                    pedidos_dia = datos_dia['pedidos'].sum()
                    
                    costos_dia = (bidones_dia * costo_bidon) + (pedidos_dia * costo_operacion)
                    utilidad_dia = ingresos_dia - costos_dia
                    margen_dia = (utilidad_dia / ingresos_dia) * 100 if ingresos_dia > 0 else 0
                    
                    rentabilidad_dias[dia] = {
                        'ingresos': ingresos_dia,
                        'costos': costos_dia,
                        'utilidad': utilidad_dia,
                        'margen': margen_dia,
                        'pedidos': pedidos_dia
                    }
            
            return {
                'metricas_generales': {
                    'ingresos_totales': ingresos_totales,
                    'costos_totales': costos_totales,
                    'utilidad_bruta': utilidad_bruta,
                    'margen_bruto': margen_bruto,
                    'roi': roi,
                    'pedidos_totales': pedidos_totales,
                    'bidones_totales': bidones_totales
                },
                'rentabilidad_dias': rentabilidad_dias,
                'dias_nombre': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            }
            
        except Exception as e:
            print(f"❌ Error en análisis de rentabilidad: {str(e)}")
            return {}
    
    def generar_comparativa_historica(self) -> Dict:
        """Genera comparativa histórica"""
        try:
            datos = self.generar_datos_analisis()
            if not datos:
                return {}
            
            df = datos['df']
            
            # Comparativa por períodos
            df['periodo'] = df['fecha'].dt.to_period('M')
            periodos_data = df.groupby('periodo').agg({
                'pedidos': ['sum', 'mean'],
                'ingresos_estimados': ['sum', 'mean'],
                'bidones_estimados': 'sum'
            }).round(2)
            
            # Últimos 3 meses
            ultimos_3_meses = periodos_data.tail(3)
            
            # Comparativa con período anterior
            if len(periodos_data) >= 2:
                mes_actual = periodos_data.iloc[-1]
                mes_anterior = periodos_data.iloc[-2]
                
                variacion_pedidos = ((mes_actual[('pedidos', 'sum')] - mes_anterior[('pedidos', 'sum')]) / 
                                   mes_anterior[('pedidos', 'sum')]) * 100
                variacion_ingresos = ((mes_actual[('ingresos_estimados', 'sum')] - mes_anterior[('ingresos_estimados', 'sum')]) / 
                                    mes_anterior[('ingresos_estimados', 'sum')]) * 100
            else:
                variacion_pedidos = 0
                variacion_ingresos = 0
            
            return {
                'ultimos_3_meses': ultimos_3_meses.to_dict(),
                'variacion_mes_anterior': {
                    'pedidos': variacion_pedidos,
                    'ingresos': variacion_ingresos
                },
                'tendencia': 'creciente' if variacion_pedidos > 0 else 'decreciente' if variacion_pedidos < 0 else 'estable'
            }
            
        except Exception as e:
            print(f"❌ Error en comparativa histórica: {str(e)}")
            return {}
    
    def generar_dashboard_completo(self) -> Dict:
        """Genera dashboard completo con todos los análisis"""
        print("📊 Generando Dashboard Avanzado...")
        
        dashboard = {
            'fecha_generacion': datetime.now().isoformat(),
            'graficos': {
                'tendencias': self.generar_grafico_tendencias(),
                'dias_semana': self.generar_grafico_dias_semana(),
                'meses': self.generar_grafico_meses()
            },
            'analisis_rentabilidad': self.generar_analisis_rentabilidad(),
            'comparativa_historica': self.generar_comparativa_historica(),
            'resumen_ejecutivo': self.generar_resumen_ejecutivo()
        }
        
        # Guardar dashboard
        try:
            with open('dashboard_avanzado.json', 'w', encoding='utf-8') as f:
                json.dump(dashboard, f, ensure_ascii=False, indent=2)
            print("✅ Dashboard guardado en 'dashboard_avanzado.json'")
        except Exception as e:
            print(f"❌ Error guardando dashboard: {str(e)}")
        
        return dashboard
    
    def generar_resumen_ejecutivo(self) -> Dict:
        """Genera resumen ejecutivo del dashboard"""
        datos = self.generar_datos_analisis()
        rentabilidad = self.generar_analisis_rentabilidad()
        comparativa = self.generar_comparativa_historica()
        
        if not datos:
            return {}
        
        # KPIs principales
        kpis = {
            'total_pedidos': datos['total_pedidos'],
            'total_ingresos': datos['total_ingresos'],
            'promedio_diario': datos['total_pedidos'] / datos['total_dias'],
            'margen_bruto': rentabilidad.get('metricas_generales', {}).get('margen_bruto', 0),
            'tendencia': comparativa.get('tendencia', 'estable')
        }
        
        # Hallazgos principales
        hallazgos = []
        
        # Análisis de días de la semana
        if rentabilidad.get('rentabilidad_dias'):
            mejor_dia = max(rentabilidad['rentabilidad_dias'].items(), 
                          key=lambda x: x[1]['utilidad'])
            peor_dia = min(rentabilidad['rentabilidad_dias'].items(), 
                          key=lambda x: x[1]['utilidad'])
            
            hallazgos.append(f"📈 Mejor día: {rentabilidad['dias_nombre'][mejor_dia[0]]} (${mejor_dia[1]['utilidad']:,.0f})")
            hallazgos.append(f"📉 Peor día: {rentabilidad['dias_nombre'][peor_dia[0]]} (${peor_dia[1]['utilidad']:,.0f})")
        
        # Análisis de rentabilidad
        if kpis['margen_bruto'] > 50:
            hallazgos.append("✅ Excelente rentabilidad (>50%)")
        elif kpis['margen_bruto'] > 30:
            hallazgos.append("👍 Buena rentabilidad (>30%)")
        else:
            hallazgos.append("⚠️ Rentabilidad mejorable (<30%)")
        
        # Análisis de tendencia
        if comparativa.get('variacion_mes_anterior', {}).get('pedidos', 0) > 10:
            hallazgos.append("🚀 Crecimiento fuerte en pedidos")
        elif comparativa.get('variacion_mes_anterior', {}).get('pedidos', 0) < -10:
            hallazgos.append("📉 Caída significativa en pedidos")
        
        return {
            'kpis': kpis,
            'hallazgos': hallazgos,
            'recomendaciones': self.generar_recomendaciones_dashboard(kpis, rentabilidad, comparativa)
        }
    
    def generar_recomendaciones_dashboard(self, kpis: Dict, rentabilidad: Dict, comparativa: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis del dashboard"""
        recomendaciones = []
        
        # Recomendaciones por rentabilidad
        if kpis['margen_bruto'] < 30:
            recomendaciones.append("💰 Optimizar costos operacionales")
            recomendaciones.append("📈 Revisar estrategia de precios")
        
        # Recomendaciones por tendencia
        if comparativa.get('tendencia') == 'decreciente':
            recomendaciones.append("🎯 Implementar estrategias de retención")
            recomendaciones.append("📱 Incrementar marketing digital")
        
        # Recomendaciones por días de la semana
        if rentabilidad.get('rentabilidad_dias'):
            mejor_dia = max(rentabilidad['rentabilidad_dias'].items(), 
                          key=lambda x: x[1]['utilidad'])
            peor_dia = min(rentabilidad['rentabilidad_dias'].items(), 
                          key=lambda x: x[1]['utilidad'])
            
            if mejor_dia[1]['utilidad'] > peor_dia[1]['utilidad'] * 2:
                recomendaciones.append(f"📅 Aprovechar alta demanda del {rentabilidad['dias_nombre'][mejor_dia[0]]}")
                recomendaciones.append(f"🎯 Implementar promociones para {rentabilidad['dias_nombre'][peor_dia[0]]}")
        
        return recomendaciones

def probar_dashboard():
    """Función para probar el dashboard avanzado"""
    print("📊 PROBANDO DASHBOARD AVANZADO")
    print("=" * 50)
    
    dashboard = DashboardAvanzado()
    
    # Generar dashboard completo
    resultado = dashboard.generar_dashboard_completo()
    
    if resultado:
        print("✅ Dashboard generado exitosamente")
        
        # Mostrar resumen ejecutivo
        resumen = resultado.get('resumen_ejecutivo', {})
        if resumen:
            print(f"\n📈 RESUMEN EJECUTIVO:")
            kpis = resumen.get('kpis', {})
            print(f"   • Total pedidos: {kpis.get('total_pedidos', 0):,}")
            print(f"   • Total ingresos: ${kpis.get('total_ingresos', 0):,.0f}")
            print(f"   • Promedio diario: {kpis.get('promedio_diario', 0):.1f} pedidos")
            print(f"   • Margen bruto: {kpis.get('margen_bruto', 0):.1f}%")
            print(f"   • Tendencia: {kpis.get('tendencia', 'N/A')}")
            
            print(f"\n🔍 HALLAZGOS:")
            for hallazgo in resumen.get('hallazgos', []):
                print(f"   • {hallazgo}")
            
            print(f"\n💡 RECOMENDACIONES:")
            for rec in resumen.get('recomendaciones', []):
                print(f"   • {rec}")
        
        print(f"\n📊 Gráficos generados: {len(resultado.get('graficos', {}))}")
        print(f"📈 Análisis de rentabilidad: {'✅' if resultado.get('analisis_rentabilidad') else '❌'}")
        print(f"📋 Comparativa histórica: {'✅' if resultado.get('comparativa_historica') else '❌'}")
    
    else:
        print("❌ Error generando dashboard")

if __name__ == "__main__":
    probar_dashboard() 