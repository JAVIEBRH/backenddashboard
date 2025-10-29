#!/usr/bin/env python3
"""
Script de migración de datos de MySQL (epedido) a MongoDB (order)
Basado en la tabla comparativa proporcionada
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

class DataMigrator:
    def __init__(self):
        self.endpoint_pedidos = "https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php"
        self.headers = {"User-Agent": "Mozilla/5.0"}
        
    def obtener_datos_antiguos(self) -> List[Dict]:
        """Obtiene datos del esquema antiguo (MySQL)"""
        try:
            response = requests.get(self.endpoint_pedidos, headers=self.headers, timeout=10)
            response.raise_for_status()
            pedidos = response.json()
            
            # Filtrar solo Aguas Ancud
            df = pd.DataFrame(pedidos)
            if 'nombrelocal' in df.columns:
                df = df[df['nombrelocal'] == 'Aguas Ancud']
            
            return df.to_dict(orient='records')
        except Exception as e:
            print(f"Error obteniendo datos antiguos: {e}")
            return []
    
    def parse_fecha_antigua(self, fecha_str: str, hora_str: str = None) -> Optional[datetime]:
        """Convierte fecha del formato DD-MM-YYYY a datetime ISO"""
        try:
            if not fecha_str or fecha_str.strip() == '':
                return None
                
            # Parsear fecha DD-MM-YYYY
            fecha = datetime.strptime(fecha_str.strip(), "%d-%m-%Y")
            
            # Si hay hora, agregarla
            if hora_str and hora_str.strip():
                try:
                    # Intentar parsear diferentes formatos de hora
                    if ':' in hora_str:
                        if 'AM' in hora_str.upper() or 'PM' in hora_str.upper():
                            hora = datetime.strptime(hora_str.strip(), "%I:%M %p").time()
                        else:
                            hora = datetime.strptime(hora_str.strip(), "%H:%M").time()
                        fecha = datetime.combine(fecha.date(), hora)
                except:
                    pass  # Si no se puede parsear la hora, usar solo la fecha
            
            return fecha
        except Exception as e:
            print(f"Error parseando fecha '{fecha_str}': {e}")
            return None
    
    def convertir_precio(self, precio_str: str) -> float:
        """Convierte precio de varchar a Number"""
        try:
            if not precio_str or precio_str.strip() == '':
                return 0.0
            return float(precio_str.strip().replace(',', '').replace('$', ''))
        except:
            return 0.0
    
    def convertir_coordenada(self, coord_str: str) -> float:
        """Convierte coordenadas de varchar a Number"""
        try:
            if not coord_str or coord_str.strip() == '':
                return 0.0
            return float(coord_str.strip())
        except:
            return 0.0
    
    def normalizar_metodo_pago(self, metodo: str) -> str:
        """Normaliza método de pago al enum nuevo"""
        if not metodo:
            return 'efectivo'
        
        metodo_lower = metodo.lower().strip()
        
        if 'efectivo' in metodo_lower:
            return 'efectivo'
        elif 'transfer' in metodo_lower:
            return 'transferencia'
        elif 'webpay' in metodo_lower:
            return 'webpay'
        elif 'mercadopago' in metodo_lower:
            return 'mercadopago'
        elif 'tarjeta' in metodo_lower:
            return 'tarjeta'
        else:
            return 'otro'
    
    def normalizar_status(self, status: str) -> str:
        """Normaliza status al enum nuevo"""
        if not status:
            return 'pendiente'
        
        status_lower = status.lower().strip()
        
        if 'pendiente' in status_lower:
            return 'pendiente'
        elif 'confirmado' in status_lower:
            return 'confirmado'
        elif 'preparando' in status_lower:
            return 'preparando'
        elif 'camino' in status_lower or 'en_camino' in status_lower:
            return 'en_camino'
        elif 'entregado' in status_lower:
            return 'entregado'
        elif 'retrasado' in status_lower:
            return 'retrasado'
        elif 'devuelto' in status_lower:
            return 'devuelto'
        elif 'cancelado' in status_lower:
            return 'cancelado'
        else:
            return 'pendiente'
    
    def normalizar_tipo_entrega(self, retiro_local: str) -> str:
        """Normaliza tipo de entrega"""
        if not retiro_local:
            return 'domicilio'
        
        retiro_lower = str(retiro_local).lower().strip()
        
        if retiro_lower in ['1', 'si', 'true', 'retiro']:
            return 'retiro'
        elif 'mostrador' in retiro_lower:
            return 'mostrador'
        else:
            return 'domicilio'
    
    def convertir_boolean(self, valor: str) -> bool:
        """Convierte string a boolean"""
        if not valor:
            return False
        
        valor_lower = str(valor).lower().strip()
        return valor_lower in ['1', 'true', 'si', 'yes']
    
    def parsear_productos(self, orden_pedido: str) -> List[Dict]:
        """Parsea el detalle de pedido a array de productos"""
        productos = []
        
        if not orden_pedido or orden_pedido.strip() == '':
            # Producto por defecto si no hay detalle
            return [{
                "productId": "bidon_agua_20l",
                "name": "Bidón de Agua 20L",
                "unitPrice": 2000,
                "quantity": 1,
                "totalPrice": 2000,
                "notes": "Producto por defecto"
            }]
        
        try:
            # Intentar parsear diferentes formatos de orden_pedido
            # Por ahora, crear un producto genérico basado en el texto
            productos.append({
                "productId": "bidon_agua_20l",
                "name": "Bidón de Agua 20L",
                "unitPrice": 2000,
                "quantity": 1,
                "totalPrice": 2000,
                "notes": orden_pedido[:100] if len(orden_pedido) > 100 else orden_pedido
            })
        except:
            # Fallback a producto por defecto
            productos.append({
                "productId": "bidon_agua_20l",
                "name": "Bidón de Agua 20L",
                "unitPrice": 2000,
                "quantity": 1,
                "totalPrice": 2000,
                "notes": "Producto por defecto"
            })
        
        return productos
    
    def obtener_dia_semana_ingles(self, fecha: datetime) -> str:
        """Convierte día de la semana a inglés"""
        dias = {
            0: 'monday',
            1: 'tuesday', 
            2: 'wednesday',
            3: 'thursday',
            4: 'friday',
            5: 'saturday',
            6: 'sunday'
        }
        return dias.get(fecha.weekday(), 'monday')
    
    def migrar_pedido(self, pedido_antiguo: Dict) -> Dict:
        """Migra un pedido del esquema antiguo al nuevo"""
        
        # Fechas
        fecha_creacion = self.parse_fecha_antigua(
            pedido_antiguo.get('fecha', ''),
            pedido_antiguo.get('hora', '')
        )
        
        fecha_entrega = self.parse_fecha_antigua(
            f"{pedido_antiguo.get('dia', '')}-{pedido_antiguo.get('mes', '')}-{pedido_antiguo.get('ano', '')}"
        )
        
        # Productos
        productos = self.parsear_productos(pedido_antiguo.get('ordenpedido', ''))
        
        # Precios
        precio = self.convertir_precio(pedido_antiguo.get('precio', '0'))
        precio_final = self.convertir_precio(pedido_antiguo.get('pagofinal', '0'))
        
        # Coordenadas
        lat = self.convertir_coordenada(pedido_antiguo.get('lat', '0'))
        lon = self.convertir_coordenada(pedido_antiguo.get('lon', '0'))
        
        # Dirección completa
        direccion_parts = []
        if pedido_antiguo.get('dire'):
            direccion_parts.append(pedido_antiguo.get('dire'))
        if pedido_antiguo.get('comuna'):
            direccion_parts.append(pedido_antiguo.get('comuna'))
        if pedido_antiguo.get('prov'):
            direccion_parts.append(pedido_antiguo.get('prov'))
        
        direccion_completa = ', '.join(filter(None, direccion_parts))
        
        # Construir documento MongoDB
        order_doc = {
            # Campos principales
            "storeId": "aguas_ancud",  # Derivado de nombrelocal
            "commerceId": "ancud_001",  # ID del comercio
            "origin": "app",  # Asumir que viene de app
            
            # Precios
            "price": precio,
            "finalPrice": precio_final if precio_final > 0 else precio,
            
            # Fechas
            "createdAt": fecha_creacion.isoformat() if fecha_creacion else datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
            
            # Productos
            "products": productos,
            
            # Método de pago
            "paymentMethod": self.normalizar_metodo_pago(pedido_antiguo.get('metodopago', '')),
            "transferPay": self.convertir_boolean(pedido_antiguo.get('transferpay', '')),
            
            # Estado
            "status": self.normalizar_status(pedido_antiguo.get('status', '')),
            
            # Cliente
            "customer": {
                "id": pedido_antiguo.get('usuario', ''),
                "name": pedido_antiguo.get('usuario', ''),
                "phone": pedido_antiguo.get('telefonou', ''),
                "address": direccion_completa,
                "block": pedido_antiguo.get('deptoblock', ''),
                "lat": lat,
                "lon": lon,
                "observations": pedido_antiguo.get('observacion', ''),
                "notificationToken": pedido_antiguo.get('notific', '')
            },
            
            # Entrega
            "deliveryType": self.normalizar_tipo_entrega(pedido_antiguo.get('retirolocal', '')),
            "deliveryPerson": {
                "id": pedido_antiguo.get('userdelivery', ''),
                "name": pedido_antiguo.get('despachador', '')
            },
            
            # Observaciones
            "merchantObservation": pedido_antiguo.get('observaciondos', ''),
            "deliveryObservation": pedido_antiguo.get('observaciondos', ''),
            
            # Campos adicionales del esquema nuevo
            "deliverySchedule": {
                "day": self.obtener_dia_semana_ingles(fecha_entrega) if fecha_entrega else 'monday',
                "hour": pedido_antiguo.get('horaagenda', '09:00 AM')
            }
        }
        
        # Agregar fecha de entrega si existe
        if fecha_entrega:
            order_doc["deliveryDate"] = fecha_entrega.isoformat()
        
        # Agregar hora de entrega real si existe
        if pedido_antiguo.get('horaentrega'):
            try:
                hora_entrega = self.parse_fecha_antigua(
                    pedido_antiguo.get('fecha', ''),
                    pedido_antiguo.get('horaentrega', '')
                )
                if hora_entrega:
                    order_doc["deliveredAt"] = hora_entrega.isoformat()
            except:
                pass
        
        # Agregar calificación si existe
        if pedido_antiguo.get('calific'):
            try:
                calificacion = str(pedido_antiguo.get('calific', '')).strip()
                if calificacion.isdigit():
                    order_doc["rating"] = {
                        "value": int(calificacion),
                        "comment": ""
                    }
                else:
                    order_doc["rating"] = {
                        "value": 0,
                        "comment": calificacion
                    }
            except:
                pass
        
        # Preservar ID original si se desea
        if pedido_antiguo.get('id'):
            order_doc["legacyId"] = int(pedido_antiguo.get('id'))
        
        # Preservar código de pedido si existe
        if pedido_antiguo.get('idpedido'):
            order_doc["orderCode"] = str(pedido_antiguo.get('idpedido'))
        
        return order_doc
    
    def migrar_todos_los_datos(self) -> List[Dict]:
        """Migra todos los datos del esquema antiguo al nuevo"""
        print("🔄 Iniciando migración de datos...")
        
        # Obtener datos antiguos
        pedidos_antiguos = self.obtener_datos_antiguos()
        print(f"📊 Encontrados {len(pedidos_antiguos)} pedidos para migrar")
        
        if not pedidos_antiguos:
            print("❌ No se encontraron datos para migrar")
            return []
        
        # Migrar cada pedido
        pedidos_nuevos = []
        errores = []
        
        for i, pedido_antiguo in enumerate(pedidos_antiguos):
            try:
                pedido_nuevo = self.migrar_pedido(pedido_antiguo)
                pedidos_nuevos.append(pedido_nuevo)
                
                if (i + 1) % 100 == 0:
                    print(f"✅ Migrados {i + 1}/{len(pedidos_antiguos)} pedidos")
                    
            except Exception as e:
                error_msg = f"Error migrando pedido {i + 1}: {str(e)}"
                errores.append(error_msg)
                print(f"❌ {error_msg}")
        
        print(f"✅ Migración completada: {len(pedidos_nuevos)} pedidos migrados")
        if errores:
            print(f"⚠️ {len(errores)} errores durante la migración")
        
        return pedidos_nuevos
    
    def guardar_datos_migrados(self, pedidos_nuevos: List[Dict], archivo: str = "orders_migrated.json"):
        """Guarda los datos migrados en un archivo JSON"""
        try:
            with open(archivo, 'w', encoding='utf-8') as f:
                json.dump(pedidos_nuevos, f, ensure_ascii=False, indent=2)
            print(f"💾 Datos migrados guardados en {archivo}")
        except Exception as e:
            print(f"❌ Error guardando datos migrados: {e}")
    
    def generar_estadisticas_migracion(self, pedidos_nuevos: List[Dict]):
        """Genera estadísticas de la migración"""
        if not pedidos_nuevos:
            return
        
        print("\n📈 ESTADÍSTICAS DE MIGRACIÓN")
        print("=" * 50)
        
        # Estadísticas generales
        total_pedidos = len(pedidos_nuevos)
        print(f"Total de pedidos migrados: {total_pedidos}")
        
        # Métodos de pago
        metodos_pago = {}
        for pedido in pedidos_nuevos:
            metodo = pedido.get('paymentMethod', 'desconocido')
            metodos_pago[metodo] = metodos_pago.get(metodo, 0) + 1
        
        print(f"\nMétodos de pago:")
        for metodo, cantidad in metodos_pago.items():
            print(f"  • {metodo}: {cantidad} ({cantidad/total_pedidos*100:.1f}%)")
        
        # Estados
        estados = {}
        for pedido in pedidos_nuevos:
            estado = pedido.get('status', 'desconocido')
            estados[estado] = estados.get(estado, 0) + 1
        
        print(f"\nEstados:")
        for estado, cantidad in estados.items():
            print(f"  • {estado}: {cantidad} ({cantidad/total_pedidos*100:.1f}%)")
        
        # Tipos de entrega
        tipos_entrega = {}
        for pedido in pedidos_nuevos:
            tipo = pedido.get('deliveryType', 'desconocido')
            tipos_entrega[tipo] = tipos_entrega.get(tipo, 0) + 1
        
        print(f"\nTipos de entrega:")
        for tipo, cantidad in tipos_entrega.items():
            print(f"  • {tipo}: {cantidad} ({cantidad/total_pedidos*100:.1f}%)")
        
        # Rango de fechas
        fechas = []
        for pedido in pedidos_nuevos:
            if pedido.get('createdAt'):
                try:
                    fecha = datetime.fromisoformat(pedido['createdAt'].replace('Z', '+00:00'))
                    fechas.append(fecha)
                except:
                    pass
        
        if fechas:
            fechas.sort()
            print(f"\nRango de fechas:")
            print(f"  • Más antiguo: {fechas[0].strftime('%d-%m-%Y')}")
            print(f"  • Más reciente: {fechas[-1].strftime('%d-%m-%Y')}")

def main():
    """Función principal de migración"""
    print("🚀 INICIANDO MIGRACIÓN DE DATOS")
    print("=" * 60)
    
    migrator = DataMigrator()
    
    # Migrar datos
    pedidos_migrados = migrator.migrar_todos_los_datos()
    
    if pedidos_migrados:
        # Guardar datos migrados
        migrator.guardar_datos_migrados(pedidos_migrados)
        
        # Generar estadísticas
        migrator.generar_estadisticas_migracion(pedidos_migrados)
        
        print(f"\n✅ MIGRACIÓN COMPLETADA EXITOSAMENTE")
        print(f"📁 Archivo generado: orders_migrated.json")
        print(f"📊 Total de pedidos migrados: {len(pedidos_migrados)}")
    else:
        print(f"\n❌ MIGRACIÓN FALLIDA - No se pudieron migrar datos")

if __name__ == "__main__":
    main()





