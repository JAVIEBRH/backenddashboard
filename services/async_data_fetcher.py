"""
Servicio asíncrono para obtener datos de endpoints externos
Implementa peticiones concurrentes con reintentos automáticos y fallback
"""

import asyncio
import httpx
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncDataFetcher:
    """Cliente asíncrono para obtener datos de endpoints externos"""
    
    def __init__(self):
        self.timeout = 10.0
        self.max_retries = 3
        self.backoff_factor = 0.5
        self.headers = {"User-Agent": "Mozilla/5.0"}
        
        # URLs de los endpoints
        self.endpoints = {
            'pedidos_antiguos': "https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php",
            'clientes_antiguos': "https://fluvi.cl/fluviDos/GoApp/endpoints/clientes.php",
            'pedidos_nuevos': "https://gobackend-qomm.onrender.com/api/store/orders",
        }
        
        # Store ID para el endpoint nuevo
        self.store_id = "68697bf9c8e5172fd536738f"
    
    async def fetch_with_retry(self, client: httpx.AsyncClient, url: str, params: Dict = None) -> Tuple[bool, Any]:
        """Realizar petición con reintentos automáticos y backoff exponencial"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Intento {attempt + 1}/{self.max_retries} para {url}")
                start_time = datetime.now()
                
                response = await client.get(
                    url, 
                    params=params, 
                    headers=self.headers, 
                    timeout=self.timeout
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Petición exitosa a {url} en {duration:.2f}s")
                
                if response.status_code == 200:
                    data = response.json()
                    return True, data
                else:
                    logger.warning(f"Respuesta no exitosa: {response.status_code} para {url}")
                    
            except Exception as e:
                logger.warning(f"Error en intento {attempt + 1} para {url}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.info(f"Esperando {wait_time}s antes del siguiente intento")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Todos los intentos fallaron para {url}")
        
        return False, None
    
    async def fetch_pedidos_antiguos(self, client: httpx.AsyncClient) -> Tuple[bool, List[Dict]]:
        """Obtener pedidos del endpoint antiguo"""
        url = self.endpoints['pedidos_antiguos']
        success, data = await self.fetch_with_retry(client, url)
        
        if success and isinstance(data, list):
            logger.info(f"Pedidos antiguos obtenidos: {len(data)} registros")
            return True, data
        else:
            logger.error("Error obteniendo pedidos antiguos")
            return False, []
    
    async def fetch_clientes_antiguos(self, client: httpx.AsyncClient) -> Tuple[bool, List[Dict]]:
        """Obtener clientes del endpoint antiguo"""
        url = self.endpoints['clientes_antiguos']
        success, data = await self.fetch_with_retry(client, url)
        
        if success and isinstance(data, list):
            logger.info(f"Clientes antiguos obtenidos: {len(data)} registros")
            return True, data
        else:
            logger.error("Error obteniendo clientes antiguos")
            return False, []
    
    async def fetch_pedidos_nuevos(self, client: httpx.AsyncClient) -> Tuple[bool, List[Dict]]:
        """Obtener pedidos del endpoint nuevo (MongoDB)"""
        url = self.endpoints['pedidos_nuevos']
        params = {
            'storeId': self.store_id,
            'limit': 1000
        }
        
        success, data = await self.fetch_with_retry(client, url, params)
        
        if success and isinstance(data, dict):
            # Extraer pedidos de la respuesta estructurada
            pedidos = data.get('data', {}).get('docs', []) if data.get('success') else []
            logger.info(f"Pedidos nuevos obtenidos: {len(pedidos)} registros")
            return True, pedidos
        else:
            logger.error("Error obteniendo pedidos nuevos")
            return False, []
    
    async def fetch_all_data_concurrent(self) -> Dict[str, Any]:
        """Obtener todos los datos de forma concurrente"""
        logger.info("Iniciando peticiones concurrentes a endpoints externos")
        start_time = datetime.now()
        
        async with httpx.AsyncClient() as client:
            # Crear tareas concurrentes
            tasks = [
                self.fetch_pedidos_antiguos(client),
                self.fetch_clientes_antiguos(client),
                self.fetch_pedidos_nuevos(client)
            ]
            
            # Ejecutar todas las tareas en paralelo
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados
            pedidos_antiguos_success, pedidos_antiguos = results[0] if not isinstance(results[0], Exception) else (False, [])
            clientes_antiguos_success, clientes_antiguos = results[1] if not isinstance(results[1], Exception) else (False, [])
            pedidos_nuevos_success, pedidos_nuevos = results[2] if not isinstance(results[2], Exception) else (False, [])
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Peticiones concurrentes completadas en {duration:.2f}s")
            
            # Log de resultados
            logger.info(f"Resultados: Pedidos antiguos: {len(pedidos_antiguos)}, "
                       f"Clientes antiguos: {len(clientes_antiguos)}, "
                       f"Pedidos nuevos: {len(pedidos_nuevos)}")
            
            return {
                'pedidos_antiguos': {
                    'success': pedidos_antiguos_success,
                    'data': pedidos_antiguos
                },
                'clientes_antiguos': {
                    'success': clientes_antiguos_success,
                    'data': clientes_antiguos
                },
                'pedidos_nuevos': {
                    'success': pedidos_nuevos_success,
                    'data': pedidos_nuevos
                },
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
    
    async def fetch_with_fallback(self, endpoint_type: str, fallback_data: Any = None) -> Tuple[bool, Any]:
        """Obtener datos con fallback a datos históricos o caché"""
        try:
            async with httpx.AsyncClient() as client:
                if endpoint_type == 'pedidos_antiguos':
                    return await self.fetch_pedidos_antiguos(client)
                elif endpoint_type == 'clientes_antiguos':
                    return await self.fetch_clientes_antiguos(client)
                elif endpoint_type == 'pedidos_nuevos':
                    return await self.fetch_pedidos_nuevos(client)
                else:
                    logger.error(f"Tipo de endpoint no reconocido: {endpoint_type}")
                    return False, fallback_data
        except Exception as e:
            logger.error(f"Error en fetch_with_fallback para {endpoint_type}: {e}")
            return False, fallback_data

# Instancia global del fetcher
async_data_fetcher = AsyncDataFetcher()

