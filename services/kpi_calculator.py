"""
Calculadora de KPIs para Aguas Ancud
Separa la lógica de cálculo de KPIs del endpoint principal
Mantiene compatibilidad total con el frontend existente
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal, ROUND_HALF_UP
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KPICalculator:
    """Calculadora principal de KPIs para Aguas Ancud"""
    
    def __init__(self):
        # Constantes del negocio
        self.PRECIO_BIDON = Decimal('2000')
        self.CUOTA_CAMION = Decimal('260000')  # Costo fijo mensual
        self.COSTO_TAPA = Decimal('51')  # Costo por tapa sin IVA
        self.IVA_RATE = Decimal('0.19')  # 19% IVA
        self.LITROS_POR_BIDON = 20
        self.CAPACIDAD_TOTAL_LITROS = 30000
        
    def parse_fecha(self, fecha_str: str) -> Optional[datetime]:
        """Parsear fecha en formato DD-MM-YYYY"""
        try:
            if isinstance(fecha_str, str) and fecha_str:
                return datetime.strptime(fecha_str, '%d-%m-%Y')
            return None
        except:
            return None
    
    def calcular_bidones_desde_orden(self, orden_pedido: str) -> int:
        """Calcular cantidad de bidones desde el campo ordenpedido"""
        try:
            if isinstance(orden_pedido, str) and orden_pedido:
                # Extraer solo números del string
                import re
                numeros = re.findall(r'\d+', orden_pedido)
                if numeros:
                    return int(numeros[0])
            return 1  # Fallback: 1 bidón por pedido
        except:
            return 1
    
    def eliminar_duplicados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Eliminar duplicados basándose en usuario y fecha"""
        try:
            if 'usuario' in df.columns and 'fecha' in df.columns:
                # Convertir fechas para comparación
                df['fecha_parsed'] = df['fecha'].apply(self.parse_fecha)
                df = df.dropna(subset=['fecha_parsed'])
                
                # Eliminar duplicados manteniendo el último
                df_clean = df.drop_duplicates(subset=["usuario", "fecha"], keep="last")
                logger.info(f"Duplicados eliminados: {len(df) - len(df_clean)} registros")
                return df_clean
            return df
        except Exception as e:
            logger.error(f"Error eliminando duplicados: {e}")
            return df
    
    def calcular_cambios_porcentuales(self, actual: Decimal, anterior: Decimal) -> float:
        """Calcular cambio porcentual con manejo de división por cero"""
        try:
            if anterior > 0:
                cambio = ((actual - anterior) / anterior) * 100
                return round(float(cambio), 1)
            return 0.0
        except:
            return 0.0
    
    def calcular_costos_reales(self, total_bidones: int) -> Decimal:
        """Calcular costos reales del mes"""
        try:
            costo_tapa_con_iva = self.COSTO_TAPA * (Decimal('1') + self.IVA_RATE)
            costos_variables = costo_tapa_con_iva * Decimal(str(total_bidones))
            costos_reales = self.CUOTA_CAMION + costos_variables
            return costos_reales.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except Exception as e:
            logger.error(f"Error calculando costos: {e}")
            return Decimal('0')
    
    def calcular_iva_neto(self, ventas: Decimal, total_bidones: int) -> Decimal:
        """Calcular IVA neto a pagar"""
        try:
            iva_ventas = ventas * self.IVA_RATE
            iva_tapas = (self.COSTO_TAPA * Decimal(str(total_bidones))) * self.IVA_RATE
            iva_neto = iva_ventas - iva_tapas
            return iva_neto.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except Exception as e:
            logger.error(f"Error calculando IVA: {e}")
            return Decimal('0')
    
    def calcular_utilidad(self, ventas: Decimal, costos: Decimal) -> Decimal:
        """Calcular utilidad neta"""
        try:
            return (ventas - costos).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except Exception as e:
            logger.error(f"Error calculando utilidad: {e}")
            return Decimal('0')
    
    def calcular_punto_equilibrio(self) -> int:
        """Calcular punto de equilibrio en bidones"""
        try:
            costo_tapa_con_iva = self.COSTO_TAPA * (Decimal('1') + self.IVA_RATE)
            margen_por_bidon = self.PRECIO_BIDON - costo_tapa_con_iva
            
            if margen_por_bidon > 0:
                punto_equilibrio = self.CUOTA_CAMION / margen_por_bidon
                return int(round(punto_equilibrio))
            return 0
        except Exception as e:
            logger.error(f"Error calculando punto de equilibrio: {e}")
            return 0
    
    def calcular_capacidad_utilizada(self, total_bidones: int) -> float:
        """Calcular capacidad utilizada en porcentaje"""
        try:
            litros_vendidos = total_bidones * self.LITROS_POR_BIDON
            capacidad_utilizada = min(100, (litros_vendidos / self.CAPACIDAD_TOTAL_LITROS) * 100)
            return round(capacidad_utilizada, 1)
        except Exception as e:
            logger.error(f"Error calculando capacidad utilizada: {e}")
            return 0.0
    
    def calcular_clientes_activos(self, df_mes: pd.DataFrame, df_mes_pasado: pd.DataFrame) -> int:
        """Calcular clientes activos de los últimos 2 meses"""
        try:
            if 'usuario' in df_mes.columns and 'usuario' in df_mes_pasado.columns:
                clientes_mes = set(df_mes['usuario'].unique())
                clientes_mes_pasado = set(df_mes_pasado['usuario'].unique())
                clientes_activos = len(clientes_mes.union(clientes_mes_pasado))
                return clientes_activos
            return 0
        except Exception as e:
            logger.error(f"Error calculando clientes activos: {e}")
            return 0
    
    def calcular_ticket_promedio(self, ventas: Decimal, total_pedidos: int) -> Decimal:
        """Calcular ticket promedio"""
        try:
            if total_pedidos > 0:
                ticket = ventas / Decimal(str(total_pedidos))
                return ticket.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            return Decimal('0')
        except Exception as e:
            logger.error(f"Error calculando ticket promedio: {e}")
            return Decimal('0')
    
    def procesar_datos_pedidos(self, pedidos: List[Dict]) -> pd.DataFrame:
        """Procesar y limpiar datos de pedidos"""
        try:
            if not pedidos:
                return pd.DataFrame()
            
            df = pd.DataFrame(pedidos)
            logger.info(f"Datos originales: {len(df)} registros")
            
            # Filtrar solo Aguas Ancud si existe la columna
            if 'nombrelocal' in df.columns:
                df = df[df['nombrelocal'] == 'Aguas Ancud']
                logger.info(f"Después de filtrar Aguas Ancud: {len(df)} registros")
            
            if df.empty or 'fecha' not in df.columns:
                logger.warning("DataFrame vacío o sin columna fecha")
                return pd.DataFrame()
            
            # Convertir fechas
            df['fecha_parsed'] = df['fecha'].apply(self.parse_fecha)
            df = df.dropna(subset=['fecha_parsed'])
            logger.info(f"Con fechas válidas: {len(df)} registros")
            
            # Convertir precios a Decimal para precisión
            df['precio'] = pd.to_numeric(df['precio'], errors='coerce').fillna(0)
            df['precio'] = df['precio'].apply(lambda x: Decimal(str(x)))
            
            # Calcular bidones desde ordenpedido
            if 'ordenpedido' in df.columns:
                df['cantidad_bidones'] = df['ordenpedido'].apply(self.calcular_bidones_desde_orden)
            else:
                df['cantidad_bidones'] = 1  # Fallback
            
            # Eliminar duplicados
            df = self.eliminar_duplicados(df)
            logger.info(f"Después de eliminar duplicados: {len(df)} registros")
            
            return df
            
        except Exception as e:
            logger.error(f"Error procesando datos de pedidos: {e}")
            return pd.DataFrame()
    
    def calcular_kpis_mes(self, df_mes: pd.DataFrame, df_mes_pasado: pd.DataFrame) -> Dict[str, Any]:
        """Calcular todos los KPIs para un mes específico"""
        try:
            if df_mes.empty:
                return self.get_kpis_default()
            
            # Calcular métricas básicas
            ventas_mes = Decimal(str(df_mes['precio'].sum()))
            ventas_mes_pasado = Decimal(str(df_mes_pasado['precio'].sum())) if not df_mes_pasado.empty else Decimal('0')
            
            total_pedidos_mes = len(df_mes)
            total_pedidos_mes_pasado = len(df_mes_pasado)
            
            # Calcular bidones
            total_bidones_mes = int(df_mes['cantidad_bidones'].sum())
            total_bidones_mes_pasado = int(df_mes_pasado['cantidad_bidones'].sum()) if not df_mes_pasado.empty else 0
            
            # Calcular costos y utilidades
            costos_reales = self.calcular_costos_reales(total_bidones_mes)
            costos_mes_pasado = self.calcular_costos_reales(total_bidones_mes_pasado)
            
            iva = self.calcular_iva_neto(ventas_mes, total_bidones_mes)
            iva_mes_pasado = self.calcular_iva_neto(ventas_mes_pasado, total_bidones_mes_pasado)
            
            utilidad = self.calcular_utilidad(ventas_mes, costos_reales)
            utilidad_mes_pasado = self.calcular_utilidad(ventas_mes_pasado, costos_mes_pasado)
            
            # Calcular métricas adicionales
            punto_equilibrio = self.calcular_punto_equilibrio()
            capacidad_utilizada = self.calcular_capacidad_utilizada(total_bidones_mes)
            clientes_activos = self.calcular_clientes_activos(df_mes, df_mes_pasado)
            
            # Calcular cambios porcentuales
            cambio_ventas_porcentaje = self.calcular_cambios_porcentuales(ventas_mes, ventas_mes_pasado)
            cambio_bidones_porcentaje = self.calcular_cambios_porcentuales(
                Decimal(str(total_bidones_mes)), 
                Decimal(str(total_bidones_mes_pasado))
            )
            
            # Calcular métricas de clientes
            clientes_activos_mes_pasado = len(df_mes_pasado['usuario'].unique()) if not df_mes_pasado.empty and 'usuario' in df_mes_pasado.columns else 0
            clientes_inactivos_mes_pasado = max(0, round(clientes_activos_mes_pasado * 0.2))
            
            # Calcular ticket promedio
            ticket_promedio_mes_pasado = self.calcular_ticket_promedio(ventas_mes_pasado, total_pedidos_mes_pasado)
            
            # Construir resultado
            resultado = {
                "ventas_mes": int(ventas_mes),
                "ventas_mes_pasado": int(ventas_mes_pasado),
                "cambio_ventas_porcentaje": cambio_ventas_porcentaje,
                "total_pedidos_mes": total_pedidos_mes,
                "total_pedidos_mes_pasado": total_pedidos_mes_pasado,
                "total_litros_mes": int(total_bidones_mes * self.LITROS_POR_BIDON),
                "litros_vendidos_mes_pasado": int(total_bidones_mes_pasado * self.LITROS_POR_BIDON),
                "total_bidones_mes": total_bidones_mes,
                "total_bidones_mes_pasado": total_bidones_mes_pasado,
                "cambio_bidones_porcentaje": cambio_bidones_porcentaje,
                "costos_reales": int(costos_reales),
                "iva": int(iva),
                "iva_mes_pasado": int(iva_mes_pasado),
                "utilidad": int(utilidad),
                "utilidad_mes_pasado": int(utilidad_mes_pasado),
                "ticket_promedio_mes_pasado": int(ticket_promedio_mes_pasado),
                "clientes_activos_mes_pasado": clientes_activos_mes_pasado,
                "clientes_inactivos_mes_pasado": clientes_inactivos_mes_pasado,
                "punto_equilibrio": punto_equilibrio,
                "clientes_activos": clientes_activos,
                "capacidad_utilizada": capacidad_utilizada,
                "litros_vendidos": int(total_bidones_mes * self.LITROS_POR_BIDON),
                "capacidad_total": self.CAPACIDAD_TOTAL_LITROS,
            }
            
            logger.info("KPIs calculados exitosamente")
            return resultado
            
        except Exception as e:
            logger.error(f"Error calculando KPIs: {e}")
            return self.get_kpis_default()
    
    def get_kpis_default(self) -> Dict[str, Any]:
        """Valores por defecto para KPIs en caso de error"""
        return {
            "ventas_mes": 0,
            "ventas_mes_pasado": 0,
            "cambio_ventas_porcentaje": 0,
            "total_pedidos_mes": 0,
            "total_pedidos_mes_pasado": 0,
            "total_litros_mes": 0,
            "litros_vendidos_mes_pasado": 0,
            "total_bidones_mes": 0,
            "total_bidones_mes_pasado": 0,
            "cambio_bidones_porcentaje": 0,
            "costos_reales": 0,
            "iva": 0,
            "iva_mes_pasado": 0,
            "utilidad": 0,
            "utilidad_mes_pasado": 0,
            "ticket_promedio_mes_pasado": 0,
            "clientes_activos_mes_pasado": 0,
            "clientes_inactivos_mes_pasado": 0,
            "punto_equilibrio": 0,
            "clientes_activos": 0,
            "capacidad_utilizada": 0.0,
            "litros_vendidos": 0,
            "capacidad_total": self.CAPACIDAD_TOTAL_LITROS,
        }

# Instancia global del calculador
kpi_calculator = KPICalculator()

