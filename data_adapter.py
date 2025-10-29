"""
Capa de adaptación de datos para unificar formatos antiguos y nuevos
Mantiene compatibilidad total con el frontend existente
Optimizado con peticiones asíncronas y caché en memoria
"""

import requests
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import logging
# from services.async_data_fetcher import async_data_fetcher
# from utils.cache_manager import cache_manager

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URLs de los endpoints
ENDPOINT_PEDIDOS_ANTIGUO = "https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php"
ENDPOINT_CLIENTES_ANTIGUO = "https://fluvi.cl/fluviDos/GoApp/endpoints/clientes.php"
ENDPOINT_PEDIDOS_NUEVO = "https://gobackend-qomm.onrender.com/api/store/orders"
STORE_ID = "68697bf9c8e5172fd536738f"

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Modelos Pydantic para validación
class PedidoAntiguo(BaseModel):
    id: str
    idpedido: Optional[str] = None
    precio: str
    fecha: str
    dia: Optional[str] = None
    mes: Optional[str] = None
    ano: Optional[str] = None
    fechamostrar: Optional[str] = None
    hora: Optional[str] = None
    horaagenda: Optional[str] = None
    ordenpedido: Optional[str] = None
    metodopago: str
    usuario: str
    telefonou: Optional[str] = None
    lat: Optional[str] = None
    lon: Optional[str] = None
    dire: str
    comuna: Optional[str] = None
    deptoblock: Optional[str] = None
    observacion: Optional[str] = None
    notific: Optional[str] = None
    status: str
    retirolocal: Optional[str] = None
    userdelivery: Optional[str] = None
    tokendelivery: Optional[str] = None
    despachador: Optional[str] = None
    observaciondos: Optional[str] = None
    calific: Optional[str] = None
    nombrelocal: str
    logo: Optional[str] = None
    pagofinal: Optional[str] = None
    transferpay: Optional[str] = None
    prov: Optional[str] = None
    horaentrega: Optional[str] = None

class ClienteAntiguo(BaseModel):
    id: str
    idcliente: str
    nombre: str
    correo: str
    clave: Optional[str] = None
    direc: str
    comuna: Optional[str] = None
    deptoblock: Optional[str] = None
    lat: Optional[str] = None
    lon: Optional[str] = None
    telefono: str
    verificar: Optional[str] = None
    notifictoken: Optional[str] = None
    fecha: Optional[str] = None
    dia: Optional[str] = None
    mes: Optional[str] = None
    ano: Optional[str] = None
    localoficial: Optional[str] = None
    dispositivo: Optional[str] = None
    v: Optional[str] = None

class PedidoNuevo(BaseModel):
    _id: str
    storeId: str
    commerceId: Optional[str] = None
    orderCode: Optional[str] = None
    legacyId: Optional[int] = None
    price: float
    finalPrice: Optional[float] = None
    createdAt: str
    updatedAt: Optional[str] = None
    deliveryDate: Optional[str] = None
    deliveredAt: Optional[str] = None
    deliverySchedule: Optional[Dict] = None
    products: List[Dict] = []
    paymentMethod: str
    transferPay: Optional[bool] = None
    status: str
    deliveryType: str
    origin: Optional[str] = None
    customer: Dict
    deliveryPerson: Optional[Dict] = None
    merchantObservation: Optional[str] = None
    deliveryObservation: Optional[str] = None
    rating: Optional[Dict] = None

class DataAdapter:
    """Adaptador principal para unificar datos antiguos y nuevos con caché asíncrono"""
    
    def __init__(self):
        self.pedidos_antiguos_cache = None
        self.clientes_antiguos_cache = None
        self.pedidos_nuevos_cache = None
        self.cache_timestamp = None
        self.cache_duration = 1800  # 30 minutos
        self.use_async = True  # Habilitar modo asíncrono
    
    def _is_cache_valid(self) -> bool:
        """Verifica si el cache es válido"""
        if not self.cache_timestamp:
            return False
        return (datetime.now().timestamp() - self.cache_timestamp) < self.cache_duration
    
    def fetch_pedidos_antiguos(self) -> List[Dict]:
        """Obtiene pedidos del endpoint antiguo"""
        try:
            logger.info("Obteniendo pedidos del endpoint antiguo...")
            response = requests.get(ENDPOINT_PEDIDOS_ANTIGUO, headers=HEADERS, timeout=10)
            response.raise_for_status()
            pedidos = response.json()
            logger.info(f"Pedidos antiguos obtenidos: {len(pedidos)} registros")
            return pedidos
        except Exception as e:
            logger.error(f"Error obteniendo pedidos antiguos: {e}")
            return []
    
    def fetch_clientes_antiguos(self) -> List[Dict]:
        """Obtiene clientes del endpoint antiguo"""
        try:
            logger.info("Obteniendo clientes del endpoint antiguo...")
            response = requests.get(ENDPOINT_CLIENTES_ANTIGUO, headers=HEADERS, timeout=10)
            response.raise_for_status()
            clientes = response.json()
            logger.info(f"Clientes antiguos obtenidos: {len(clientes)} registros")
            return clientes
        except Exception as e:
            logger.error(f"Error obteniendo clientes antiguos: {e}")
            return []
    
    def fetch_pedidos_nuevos(self) -> List[Dict]:
        """Obtiene pedidos del endpoint nuevo (MongoDB)"""
        try:
            logger.info("Obteniendo pedidos del endpoint nuevo...")
            url = f"{ENDPOINT_PEDIDOS_NUEVO}?storeId={STORE_ID}&limit=1000"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            pedidos = data.get('data', {}).get('docs', []) if data.get('success') else []
            logger.info(f"Pedidos nuevos obtenidos: {len(pedidos)} registros")
            return pedidos
        except Exception as e:
            logger.error(f"Error obteniendo pedidos nuevos: {e}")
            return []
    
    def convertir_pedido_nuevo_a_antiguo(self, pedido_nuevo: Dict) -> Dict:
        """Convierte un pedido del formato nuevo al formato antiguo"""
        try:
            # Parsear fecha ISO
            fecha_iso = pedido_nuevo.get('createdAt', '')
            fecha_dt = None
            if fecha_iso:
                try:
                    fecha_dt = datetime.fromisoformat(fecha_iso.replace('Z', '+00:00'))
                except:
                    fecha_dt = datetime.now()
            
            # Extraer datos del customer
            customer = pedido_nuevo.get('customer', {})
            
            # Convertir deliverySchedule
            delivery_schedule = pedido_nuevo.get('deliverySchedule', {})
            hora_agenda = delivery_schedule.get('hour', '') if delivery_schedule else ''
            
            # Convertir products a ordenpedido (cantidad de bidones)
            products = pedido_nuevo.get('products', [])
            if products:
                # Sumar la cantidad de todos los productos
                total_cantidad = sum(product.get('quantity', 1) for product in products)
                orden_pedido = str(total_cantidad)
            else:
                # Si no hay productos, calcular basado en el precio real
                precio = pedido_nuevo.get('price', 0)
                if precio > 0:
                    # Precio real: $2000 por bidón para delivery
                    bidones_calculados = max(1, round(precio / 2000))
                    orden_pedido = str(bidones_calculados)
                else:
                    orden_pedido = '1'
            
            # Mapear deliveryType a retirolocal
            delivery_type = pedido_nuevo.get('deliveryType', 'domicilio')
            retiro_local = 'no' if delivery_type == 'domicilio' else 'si'
            
            # Mapear transferPay
            transfer_pay = pedido_nuevo.get('transferPay', False)
            transfer_pay_str = 'si' if transfer_pay else ''
            
            # Extraer deliveryPerson
            delivery_person = pedido_nuevo.get('deliveryPerson', {})
            
            # Extraer rating
            rating = pedido_nuevo.get('rating', {})
            
            # Construir pedido en formato antiguo
            pedido_antiguo = {
                'id': pedido_nuevo.get('_id', ''),
                'idpedido': pedido_nuevo.get('orderCode', ''),
                'precio': str(pedido_nuevo.get('price', 0)),
                'fecha': fecha_dt.strftime('%d-%m-%Y') if fecha_dt else '',
                'dia': fecha_dt.strftime('%d') if fecha_dt else '',
                'mes': fecha_dt.strftime('%m') if fecha_dt else '',
                'ano': fecha_dt.strftime('%Y') if fecha_dt else '',
                'fechamostrar': fecha_dt.strftime('%A, %d') if fecha_dt else '',
                'hora': fecha_dt.strftime('%H:%M:%S') if fecha_dt else '',
                'horaagenda': hora_agenda,
                'ordenpedido': orden_pedido,
                'metodopago': pedido_nuevo.get('paymentMethod', ''),
                'usuario': customer.get('email', ''),
                'telefonou': customer.get('phone', ''),
                'lat': str(customer.get('lat', '')),
                'lon': str(customer.get('lon', '')),
                'dire': customer.get('address', ''),
                'comuna': customer.get('address', '').split(',')[-1].strip() if customer.get('address') else '',
                'deptoblock': customer.get('block', ''),
                'observacion': customer.get('observations', ''),
                'notific': customer.get('notificationToken', ''),
                'status': pedido_nuevo.get('status', ''),
                'retirolocal': retiro_local,
                'userdelivery': delivery_person.get('id', ''),
                'tokendelivery': delivery_person.get('token', ''),
                'despachador': delivery_person.get('name', ''),
                'observaciondos': pedido_nuevo.get('merchantObservation', ''),
                'calific': str(rating.get('value', '')),
                'nombrelocal': 'Aguas Ancud',
                'logo': '',
                'pagofinal': pedido_nuevo.get('paymentMethod', ''),
                'transferpay': transfer_pay_str,
                'prov': pedido_nuevo.get('origin', ''),
                'horaentrega': pedido_nuevo.get('deliveredAt', '')
            }
            
            return pedido_antiguo
            
        except Exception as e:
            logger.error(f"Error convirtiendo pedido nuevo: {e}")
            return {}
    
    def convertir_cliente_nuevo_a_antiguo(self, cliente_nuevo: Dict) -> Dict:
        """Convierte un cliente del formato nuevo al formato antiguo"""
        # Por ahora, los clientes nuevos se extraen de los pedidos
        # Si hay un endpoint específico de clientes nuevos, se implementaría aquí
        return {}
    
    def combinar_pedidos(self, pedidos_antiguos: List[Dict], pedidos_nuevos: List[Dict]) -> List[Dict]:
        """Combina pedidos antiguos y nuevos, ordenados por fecha"""
        try:
            # Convertir pedidos nuevos al formato antiguo
            pedidos_nuevos_convertidos = []
            for pedido_nuevo in pedidos_nuevos:
                pedido_convertido = self.convertir_pedido_nuevo_a_antiguo(pedido_nuevo)
                if pedido_convertido:
                    pedidos_nuevos_convertidos.append(pedido_convertido)
            
            # Combinar ambos conjuntos
            todos_los_pedidos = pedidos_antiguos + pedidos_nuevos_convertidos
            
            # Ordenar por fecha (más recientes primero)
            def parse_fecha_para_ordenar(pedido):
                try:
                    fecha_str = pedido.get('fecha', '')
                    if fecha_str:
                        return datetime.strptime(fecha_str, '%d-%m-%Y')
                    return datetime.min
                except:
                    return datetime.min
            
            todos_los_pedidos.sort(key=parse_fecha_para_ordenar, reverse=True)
            
            logger.info(f"Pedidos combinados: {len(todos_los_pedidos)} total")
            logger.info(f"  - Antiguos: {len(pedidos_antiguos)}")
            logger.info(f"  - Nuevos convertidos: {len(pedidos_nuevos_convertidos)}")
            
            return todos_los_pedidos
            
        except Exception as e:
            logger.error(f"Error combinando pedidos: {e}")
            return pedidos_antiguos  # Fallback a solo antiguos
    
    def combinar_clientes(self, clientes_antiguos: List[Dict], clientes_nuevos: List[Dict]) -> List[Dict]:
        """Combina clientes antiguos y nuevos"""
        try:
            # Por ahora, solo devolvemos los clientes antiguos
            # Los clientes nuevos se pueden extraer de los pedidos si es necesario
            logger.info(f"Clientes combinados: {len(clientes_antiguos)} total")
            return clientes_antiguos
            
        except Exception as e:
            logger.error(f"Error combinando clientes: {e}")
            return clientes_antiguos
    
    async def obtener_pedidos_combinados_async(self) -> List[Dict]:
        """Obtiene pedidos combinados usando peticiones asíncronas con caché"""
        cache_key = "pedidos_combinados"
        
        # Verificar caché asíncrono
        cached_data = await cache_manager.get(cache_key, "pedidos")
        if cached_data:
            logger.info(f"Usando caché asíncrono de pedidos: {len(cached_data)} registros")
            return cached_data
        
        logger.info("Cache no válido o vacío, obteniendo datos frescos con peticiones asíncronas...")
        try:
            # Obtener todos los datos de forma concurrente
            results = await async_data_fetcher.fetch_all_data_concurrent()
            
            # Extraer datos de los resultados
            pedidos_antiguos = results['pedidos_antiguos']['data'] if results['pedidos_antiguos']['success'] else []
            pedidos_nuevos = results['pedidos_nuevos']['data'] if results['pedidos_nuevos']['success'] else []
            
            logger.info(f"Datos obtenidos asíncronamente - Antiguos: {len(pedidos_antiguos)}, Nuevos: {len(pedidos_nuevos)}")
            
            # Combinar y ordenar
            pedidos_combinados = self.combinar_pedidos(pedidos_antiguos, pedidos_nuevos)
            
            logger.info(f"Pedidos combinados: {len(pedidos_combinados)} registros")
            
            # Actualizar caché asíncrono
            await cache_manager.set(cache_key, pedidos_combinados, "pedidos", 1800)  # 30 minutos
            
            logger.info("Cache asíncrono actualizado exitosamente")
            return pedidos_combinados
            
        except Exception as e:
            logger.error(f"Error obteniendo pedidos combinados asíncronamente: {e}")
            # Fallback: usar caché síncrono si está disponible
            if self.pedidos_antiguos_cache is not None:
                logger.info("Usando cache síncrono como fallback debido a error")
                return self.pedidos_antiguos_cache
            return []
    
    def obtener_pedidos_combinados(self) -> List[Dict]:
        """Obtiene pedidos combinados (antiguos + nuevos) con cache - versión síncrona"""
        logger.info(f"Cache válido: {self._is_cache_valid()}, Cache disponible: {self.pedidos_antiguos_cache is not None}")
        
        if self._is_cache_valid() and self.pedidos_antiguos_cache is not None:
            logger.info(f"Usando cache de pedidos: {len(self.pedidos_antiguos_cache)} registros")
            return self.pedidos_antiguos_cache
        
        logger.info("Cache no válido o vacío, obteniendo datos frescos...")
        try:
            # Obtener datos de ambas fuentes
            pedidos_antiguos = self.fetch_pedidos_antiguos()
            pedidos_nuevos = self.fetch_pedidos_nuevos()
            
            logger.info(f"Datos obtenidos - Antiguos: {len(pedidos_antiguos)}, Nuevos: {len(pedidos_nuevos)}")
            
            # Combinar y ordenar
            pedidos_combinados = self.combinar_pedidos(pedidos_antiguos, pedidos_nuevos)
            
            logger.info(f"Pedidos combinados: {len(pedidos_combinados)} registros")
            
            # Actualizar cache
            self.pedidos_antiguos_cache = pedidos_combinados
            self.cache_timestamp = datetime.now().timestamp()
            
            logger.info("Cache actualizado exitosamente")
            return pedidos_combinados
            
        except Exception as e:
            logger.error(f"Error obteniendo pedidos combinados: {e}")
            # Fallback: usar cache si está disponible
            if self.pedidos_antiguos_cache is not None:
                logger.info("Usando cache como fallback debido a error")
                return self.pedidos_antiguos_cache
            return []
    
    def obtener_clientes_combinados(self) -> List[Dict]:
        """Obtiene clientes combinados (antiguos + nuevos) con cache"""
        if self._is_cache_valid() and self.clientes_antiguos_cache is not None:
            logger.info("Usando cache de clientes")
            return self.clientes_antiguos_cache
        
        try:
            # Obtener datos de ambas fuentes
            clientes_antiguos = self.fetch_clientes_antiguos()
            clientes_nuevos = []  # Por ahora no hay endpoint de clientes nuevos
            
            # Combinar
            clientes_combinados = self.combinar_clientes(clientes_antiguos, clientes_nuevos)
            
            # Actualizar cache
            self.clientes_antiguos_cache = clientes_combinados
            self.cache_timestamp = datetime.now().timestamp()
            
            return clientes_combinados
            
        except Exception as e:
            logger.error(f"Error obteniendo clientes combinados: {e}")
            # Fallback: usar cache si está disponible
            if self.clientes_antiguos_cache is not None:
                logger.info("Usando cache de clientes como fallback debido a error")
                return self.clientes_antiguos_cache
            return []

# Instancia global del adaptador
data_adapter = DataAdapter()
