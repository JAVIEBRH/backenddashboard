"""
Gestor de caché asíncrono en memoria para optimizar rendimiento
Maneja caché independiente por endpoint con TTL de 30 minutos
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncCacheManager:
    """Gestor de caché asíncrono con TTL independiente por endpoint"""
    
    def __init__(self, default_ttl: int = 1800):  # 30 minutos por defecto
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.lock = asyncio.Lock()
    
    async def get(self, key: str, endpoint: str) -> Optional[Any]:
        """Obtener valor del caché si es válido"""
        async with self.lock:
            if endpoint not in self.cache:
                return None
            
            cache_data = self.cache[endpoint].get(key)
            if not cache_data:
                return None
            
            data, timestamp, ttl = cache_data
            
            # Verificar si el caché ha expirado
            if time.time() - timestamp > ttl:
                logger.info(f"Cache expirado para {endpoint}:{key}")
                del self.cache[endpoint][key]
                return None
            
            logger.info(f"Cache hit para {endpoint}:{key}")
            return data
    
    async def set(self, key: str, value: Any, endpoint: str, ttl: Optional[int] = None) -> None:
        """Establecer valor en el caché"""
        async with self.lock:
            if endpoint not in self.cache:
                self.cache[endpoint] = {}
            
            ttl = ttl or self.default_ttl
            self.cache[endpoint][key] = (value, time.time(), ttl)
            logger.info(f"Cache set para {endpoint}:{key} con TTL {ttl}s")
    
    async def invalidate(self, endpoint: str, key: Optional[str] = None) -> None:
        """Invalidar caché de un endpoint específico o una clave específica"""
        async with self.lock:
            if endpoint not in self.cache:
                return
            
            if key:
                if key in self.cache[endpoint]:
                    del self.cache[endpoint][key]
                    logger.info(f"Cache invalidado para {endpoint}:{key}")
            else:
                del self.cache[endpoint]
                logger.info(f"Cache invalidado completamente para {endpoint}")
    
    async def clear_all(self) -> None:
        """Limpiar todo el caché"""
        async with self.lock:
            self.cache.clear()
            logger.info("Cache completamente limpiado")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché"""
        async with self.lock:
            stats = {
                "total_endpoints": len(self.cache),
                "endpoints": {}
            }
            
            for endpoint, data in self.cache.items():
                valid_entries = 0
                expired_entries = 0
                
                for key, (_, timestamp, ttl) in data.items():
                    if time.time() - timestamp > ttl:
                        expired_entries += 1
                    else:
                        valid_entries += 1
                
                stats["endpoints"][endpoint] = {
                    "valid_entries": valid_entries,
                    "expired_entries": expired_entries,
                    "total_entries": valid_entries + expired_entries
                }
            
            return stats
    
    async def cleanup_expired(self) -> None:
        """Limpiar entradas expiradas del caché"""
        async with self.lock:
            current_time = time.time()
            for endpoint in list(self.cache.keys()):
                for key in list(self.cache[endpoint].keys()):
                    _, timestamp, ttl = self.cache[endpoint][key]
                    if current_time - timestamp > ttl:
                        del self.cache[endpoint][key]
                        logger.info(f"Cache expirado limpiado: {endpoint}:{key}")
                
                # Eliminar endpoint vacío
                if not self.cache[endpoint]:
                    del self.cache[endpoint]

# Instancia global del gestor de caché
cache_manager = AsyncCacheManager()

# Funciones de conveniencia para uso síncrono desde endpoints FastAPI
def get_cache_key(endpoint: str, params: Dict[str, Any] = None) -> str:
    """Generar clave de caché basada en endpoint y parámetros"""
    if params:
        param_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
        return f"{endpoint}_{param_str}"
    return endpoint

def sync_get_cache(key: str, endpoint: str) -> Optional[Any]:
    """Versión síncrona para obtener del caché"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Si estamos en un loop ya corriendo, crear una tarea
            task = asyncio.create_task(cache_manager.get(key, endpoint))
            # Esto es una aproximación - en producción sería mejor usar un enfoque diferente
            return None  # Por ahora, deshabilitar caché síncrono
        else:
            return loop.run_until_complete(cache_manager.get(key, endpoint))
    except:
        return None

def sync_set_cache(key: str, value: Any, endpoint: str, ttl: Optional[int] = None) -> None:
    """Versión síncrona para establecer en el caché"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Si estamos en un loop ya corriendo, crear una tarea
            asyncio.create_task(cache_manager.set(key, value, endpoint, ttl))
        else:
            loop.run_until_complete(cache_manager.set(key, value, endpoint, ttl))
    except:
        pass  # Fallback silencioso si no se puede usar caché

def sync_invalidate_cache(endpoint: str, key: Optional[str] = None) -> None:
    """Versión síncrona para invalidar caché"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(cache_manager.invalidate(endpoint, key))
        else:
            loop.run_until_complete(cache_manager.invalidate(endpoint, key))
    except:
        pass

