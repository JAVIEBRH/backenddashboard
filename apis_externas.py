import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

class APIsExternas:
    def __init__(self):
        self.api_keys = self.cargar_api_keys()
        self.cache = {}
        self.cache_timeout = 3600  # 1 hora
        
    def cargar_api_keys(self) -> Dict:
        """Carga las API keys desde archivo de configuración"""
        try:
            with open('api_keys.json', 'r') as f:
                return json.load(f)
        except:
            # Configuración por defecto (sin API keys)
            return {
                'openweather': None,
                'google_events': None,
                'traffic_api': None
            }
    
    def obtener_clima_real(self, lat: float = -41.8697, lon: float = -73.8203) -> Dict:
        """Obtiene datos del clima real usando OpenWeatherMap API"""
        try:
            # Verificar cache
            cache_key = f"clima_{lat}_{lon}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    return cache_data
            
            # API Key de OpenWeatherMap (gratuita)
            api_key = self.api_keys.get('openweather')
            
            if api_key:
                # API real
                url = f"http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': api_key,
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                clima_data = {
                    'temperatura': data['main']['temp'],
                    'humedad': data['main']['humidity'],
                    'descripcion': data['weather'][0]['description'],
                    'icono': data['weather'][0]['icon'],
                    'viento': data['wind']['speed'],
                    'presion': data['main']['pressure']
                }
            else:
                # FALLBACK: Simulación basada en época del año (cuando API real no está disponible)
                clima_data = self.simular_clima()
            
            # Guardar en cache
            self.cache[cache_key] = (time.time(), clima_data)
            
            return clima_data
            
        except Exception as e:
            print(f"❌ Error obteniendo clima: {str(e)}")
            # FALLBACK: Usar simulación cuando hay errores de conexión
            return self.simular_clima()
    
    def simular_clima(self) -> Dict:
        """FALLBACK: Simula datos del clima cuando la API real no está disponible"""
        mes = datetime.now().month
        
        if mes in [12, 1, 2]:  # Verano
            return {
                'temperatura': 25 + (mes - 12) * 2,
                'humedad': 60,
                'descripcion': 'Soleado',
                'icono': '01d',
                'viento': 15,
                'presion': 1013
            }
        elif mes in [6, 7, 8]:  # Invierno
            return {
                'temperatura': 8 + (mes - 6) * 2,
                'humedad': 80,
                'descripcion': 'Lluvioso',
                'icono': '10d',
                'viento': 25,
                'presion': 1005
            }
        else:  # Primavera/Otoño
            return {
                'temperatura': 15 + (mes - 3) * 2,
                'humedad': 70,
                'descripcion': 'Parcialmente nublado',
                'icono': '02d',
                'viento': 20,
                'presion': 1010
            }
    
    def calcular_factor_clima(self, clima_data: Dict) -> float:
        """Calcula factor de demanda basado en el clima"""
        temp = clima_data['temperatura']
        humedad = clima_data['humedad']
        descripcion = clima_data['description'].lower()
        
        # Factor por temperatura
        if temp > 25:
            factor_temp = 1.2  # Calor = más agua
        elif temp > 20:
            factor_temp = 1.1
        elif temp > 15:
            factor_temp = 1.0
        elif temp > 10:
            factor_temp = 0.95
        else:
            factor_temp = 0.9  # Frío = menos agua
        
        # Factor por humedad
        if humedad < 50:
            factor_humedad = 1.1  # Seco = más agua
        elif humedad > 80:
            factor_humedad = 0.95  # Húmedo = menos agua
        else:
            factor_humedad = 1.0
        
        # Factor por descripción
        if 'lluvia' in descripcion or 'lluvioso' in descripcion:
            factor_desc = 0.9  # Lluvia = menos demanda
        elif 'soleado' in descripcion or 'despejado' in descripcion:
            factor_desc = 1.1  # Sol = más demanda
        else:
            factor_desc = 1.0
        
        return round(factor_temp * factor_humedad * factor_desc, 3)
    
    def obtener_eventos_locales(self, fecha: datetime) -> List[Dict]:
        """Obtiene eventos locales que pueden afectar la demanda"""
        try:
            # Verificar cache
            cache_key = f"eventos_{fecha.strftime('%Y-%m')}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    return cache_data
            
            # FALLBACK: Eventos locales de Ancud (simulados cuando API real no está disponible)
            eventos_ancud = [
                {
                    'nombre': 'Fiesta de la Cerveza',
                    'fecha': '2025-01-15',
                    'tipo': 'festival',
                    'impacto': 1.3,
                    'descripcion': 'Festival gastronómico local'
                },
                {
                    'nombre': 'Feria Costumbrista',
                    'fecha': '2025-02-20',
                    'tipo': 'feria',
                    'impacto': 1.2,
                    'descripcion': 'Feria de productos locales'
                },
                {
                    'nombre': 'Carnaval de Invierno',
                    'fecha': '2025-07-10',
                    'tipo': 'carnaval',
                    'impacto': 1.1,
                    'descripcion': 'Evento cultural de invierno'
                },
                {
                    'nombre': 'Festival del Mar',
                    'fecha': '2025-09-18',
                    'tipo': 'festival',
                    'impacto': 1.4,
                    'descripcion': 'Festival marítimo'
                },
                {
                    'nombre': 'Fiesta de la Vendimia',
                    'fecha': '2025-03-25',
                    'tipo': 'fiesta',
                    'impacto': 1.15,
                    'descripcion': 'Celebración de la vendimia'
                }
            ]
            
            # Filtrar eventos para la fecha específica
            fecha_str = fecha.strftime('%Y-%m-%d')
            eventos_fecha = [
                evento for evento in eventos_ancud 
                if evento['fecha'] == fecha_str
            ]
            
            # Guardar en cache
            self.cache[cache_key] = (time.time(), eventos_fecha)
            
            return eventos_fecha
            
        except Exception as e:
            print(f"❌ Error obteniendo eventos: {str(e)}")
            return []
    
    def calcular_factor_eventos(self, eventos: List[Dict]) -> float:
        """Calcula factor de demanda basado en eventos"""
        if not eventos:
            return 1.0
        
        # Sumar impactos de todos los eventos
        factor_total = 1.0
        for evento in eventos:
            factor_total *= evento['impacto']
        
        return round(factor_total, 3)
    
    def obtener_trafico(self, fecha: datetime) -> Dict:
        """FALLBACK: Obtiene datos de tráfico (simulado cuando API real no está disponible)"""
        try:
            # Verificar cache
            cache_key = f"trafico_{fecha.strftime('%Y-%m-%d')}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                if time.time() - cache_time < self.cache_timeout:
                    return cache_data
            
            # FALLBACK: Simulación de tráfico basada en día de la semana
            dia_semana = fecha.weekday()
            
            if dia_semana < 5:  # Lunes a Viernes
                trafico_data = {
                    'nivel': 'alto',
                    'factor': 1.1,
                    'descripcion': 'Tráfico laboral normal'
                }
            elif dia_semana == 5:  # Sábado
                trafico_data = {
                    'nivel': 'medio',
                    'factor': 0.9,
                    'descripcion': 'Tráfico de fin de semana'
                }
            else:  # Domingo
                trafico_data = {
                    'nivel': 'bajo',
                    'factor': 0.8,
                    'descripcion': 'Tráfico mínimo dominical'
                }
            
            # Guardar en cache
            self.cache[cache_key] = (time.time(), trafico_data)
            
            return trafico_data
            
        except Exception as e:
            print(f"❌ Error obteniendo tráfico: {str(e)}")
            return {
                'nivel': 'normal',
                'factor': 1.0,
                'descripcion': 'Datos no disponibles'
            }
    
    def obtener_datos_completos(self, fecha: datetime) -> Dict:
        """Obtiene todos los datos externos para una fecha"""
        try:
            # Clima
            clima = self.obtener_clima_real()
            factor_clima = self.calcular_factor_clima(clima)
            
            # Eventos
            eventos = self.obtener_eventos_locales(fecha)
            factor_eventos = self.calcular_factor_eventos(eventos)
            
            # Tráfico
            trafico = self.obtener_trafico(fecha)
            
            return {
                'fecha': fecha.strftime('%d-%m-%Y'),
                'clima': {
                    'datos': clima,
                    'factor': factor_clima
                },
                'eventos': {
                    'lista': eventos,
                    'factor': factor_eventos
                },
                'trafico': {
                    'datos': trafico,
                    'factor': trafico['factor']
                },
                'factor_total_externo': round(factor_clima * factor_eventos * trafico['factor'], 3)
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo datos externos: {str(e)}")
            return {
                'fecha': fecha.strftime('%d-%m-%Y'),
                'clima': {'datos': {}, 'factor': 1.0},
                'eventos': {'lista': [], 'factor': 1.0},
                'trafico': {'datos': {}, 'factor': 1.0},
                'factor_total_externo': 1.0
            }

def probar_apis_externas():
    """Función para probar las APIs externas"""
    print("🌐 PROBANDO APIS EXTERNAS")
    print("=" * 50)
    
    apis = APIsExternas()
    fecha_prueba = datetime.now() + timedelta(days=1)
    
    datos = apis.obtener_datos_completos(fecha_prueba)
    
    print(f"📅 Fecha: {datos['fecha']}")
    
    print(f"\n🌡️ CLIMA:")
    clima = datos['clima']
    print(f"   • Temperatura: {clima['datos'].get('temperatura', 'N/A')}°C")
    print(f"   • Humedad: {clima['datos'].get('humedad', 'N/A')}%")
    print(f"   • Descripción: {clima['datos'].get('descripcion', 'N/A')}")
    print(f"   • Factor: {clima['factor']}")
    
    print(f"\n🎉 EVENTOS:")
    eventos = datos['eventos']
    if eventos['lista']:
        for evento in eventos['lista']:
            print(f"   • {evento['nombre']}: {evento['descripcion']}")
    else:
        print("   • No hay eventos programados")
    print(f"   • Factor: {eventos['factor']}")
    
    print(f"\n🚗 TRÁFICO:")
    trafico = datos['trafico']
    print(f"   • Nivel: {trafico['datos'].get('nivel', 'N/A')}")
    print(f"   • Descripción: {trafico['datos'].get('descripcion', 'N/A')}")
    print(f"   • Factor: {trafico['factor']}")
    
    print(f"\n📊 FACTOR TOTAL EXTERNO: {datos['factor_total_externo']}")

if __name__ == "__main__":
    probar_apis_externas() 