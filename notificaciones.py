import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class NotificacionesInteligentes:
    def __init__(self):
        self.config = self.cargar_configuracion()
        self.historial_alertas = []
        self.umbrales = {
            'alta_demanda': 8,
            'baja_demanda': 3,
            'inventario_bajo': 50,
            'inventario_critico': 20
        }
    
    def cargar_configuracion(self) -> Dict:
        """Carga configuración de notificaciones"""
        try:
            with open('config_notificaciones.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            # Configuración por defecto
            return {
                'email': {
                    'habilitado': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'email_origen': '',
                    'password': '',
                    'email_destino': ''
                },
                'alertas': {
                    'alta_demanda': True,
                    'baja_demanda': True,
                    'inventario': True,
                    'promociones': True
                },
                'frecuencia': {
                    'alertas_diarias': True,
                    'resumen_semanal': True,
                    'resumen_mensual': True
                }
            }
    
    def guardar_configuracion(self):
        """Guarda la configuración actual"""
        try:
            with open('config_notificaciones.json', 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Error guardando configuración: {str(e)}")
    
    def verificar_alta_demanda(self, prediccion: float, fecha: datetime) -> Optional[Dict]:
        """Verifica si hay alta demanda y genera alerta"""
        if prediccion > self.umbrales['alta_demanda']:
            alerta = {
                'tipo': 'alta_demanda',
                'fecha': fecha.strftime('%d-%m-%Y'),
                'valor': prediccion,
                'umbral': self.umbrales['alta_demanda'],
                'mensaje': f"🔥 ALTA DEMANDA ESPERADA: {prediccion} pedidos",
                'recomendaciones': [
                    "📦 Preparar inventario extra",
                    "👥 Asegurar personal suficiente",
                    "🚚 Verificar capacidad de entrega",
                    "💰 Considerar precios dinámicos"
                ],
                'prioridad': 'alta',
                'timestamp': datetime.now().isoformat()
            }
            
            self.historial_alertas.append(alerta)
            return alerta
        
        return None
    
    def verificar_baja_demanda(self, prediccion: float, fecha: datetime) -> Optional[Dict]:
        """Verifica si hay baja demanda y genera alerta"""
        if prediccion < self.umbrales['baja_demanda']:
            alerta = {
                'tipo': 'baja_demanda',
                'fecha': fecha.strftime('%d-%m-%Y'),
                'valor': prediccion,
                'umbral': self.umbrales['baja_demanda'],
                'mensaje': f"📉 BAJA DEMANDA ESPERADA: {prediccion} pedidos",
                'recomendaciones': [
                    "🎯 Lanzar promociones especiales",
                    "📱 Incrementar marketing digital",
                    "🤝 Contactar clientes inactivos",
                    "📊 Analizar causas de la baja demanda"
                ],
                'prioridad': 'media',
                'timestamp': datetime.now().isoformat()
            }
            
            self.historial_alertas.append(alerta)
            return alerta
        
        return None
    
    def verificar_inventario(self, inventario_actual: int) -> List[Dict]:
        """Verifica el estado del inventario y genera alertas"""
        alertas = []
        
        if inventario_actual <= self.umbrales['inventario_critico']:
            alerta = {
                'tipo': 'inventario_critico',
                'valor': inventario_actual,
                'umbral': self.umbrales['inventario_critico'],
                'mensaje': f"🚨 INVENTARIO CRÍTICO: {inventario_actual} bidones",
                'recomendaciones': [
                    "🚚 Hacer pedido urgente a proveedor",
                    "📞 Contactar proveedores alternativos",
                    "⚡ Priorizar clientes VIP",
                    "📊 Revisar proyecciones de demanda"
                ],
                'prioridad': 'critica',
                'timestamp': datetime.now().isoformat()
            }
            alertas.append(alerta)
            
        elif inventario_actual <= self.umbrales['inventario_bajo']:
            alerta = {
                'tipo': 'inventario_bajo',
                'valor': inventario_actual,
                'umbral': self.umbrales['inventario_bajo'],
                'mensaje': f"⚠️ INVENTARIO BAJO: {inventario_actual} bidones",
                'recomendaciones': [
                    "📋 Planificar próximo pedido",
                    "📊 Revisar tendencias de venta",
                    "🔄 Optimizar rotación de inventario",
                    "📈 Ajustar predicciones de demanda"
                ],
                'prioridad': 'alta',
                'timestamp': datetime.now().isoformat()
            }
            alertas.append(alerta)
        
        self.historial_alertas.extend(alertas)
        return alertas
    
    def generar_sugerencias_promociones(self, prediccion: float, fecha: datetime) -> List[Dict]:
        """Genera sugerencias de promociones basadas en predicciones"""
        sugerencias = []
        
        # Análisis por día de la semana
        dia_semana = fecha.weekday()
        dias_nombre = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        if dia_semana == 5:  # Sábado
            sugerencias.append({
                'tipo': 'promocion_sabado',
                'titulo': 'Promoción de Fin de Semana',
                'descripcion': 'Incrementar ventas en día de baja demanda',
                'acciones': [
                    "🎉 Descuento 10% en pedidos de fin de semana",
                    "📱 Campaña en redes sociales",
                    "🎯 Envío de SMS a clientes inactivos",
                    "🏷️ Etiquetas especiales para productos"
                ],
                'efectividad_esperada': '15-25% incremento',
                'timestamp': datetime.now().isoformat()
            })
        
        # Análisis por época del año
        mes = fecha.month
        if mes in [12, 1, 2]:  # Verano
            sugerencias.append({
                'tipo': 'promocion_verano',
                'titulo': 'Promoción de Verano',
                'descripcion': 'Aprovechar alta demanda estacional',
                'acciones': [
                    "☀️ Paquetes familiares para verano",
                    "🏖️ Promoción 'Más agua, más vida'",
                    "🎁 Regalo adicional en pedidos grandes",
                    "📦 Descuentos por volumen"
                ],
                'efectividad_esperada': '20-30% incremento',
                'timestamp': datetime.now().isoformat()
            })
        
        # Análisis por nivel de demanda
        if prediccion < 4:
            sugerencias.append({
                'tipo': 'promocion_estimulo',
                'titulo': 'Promoción de Estímulo',
                'descripcion': 'Incrementar demanda en período bajo',
                'acciones': [
                    "💥 Oferta flash por 24 horas",
                    "🎯 Descuentos progresivos por cantidad",
                    "📞 Llamadas a clientes fieles",
                    "📧 Email marketing personalizado"
                ],
                'efectividad_esperada': '25-40% incremento',
                'timestamp': datetime.now().isoformat()
            })
        
        return sugerencias
    
    def generar_resumen_diario(self, predicciones: List[Dict], inventario: int) -> Dict:
        """Genera resumen diario de alertas y recomendaciones"""
        fecha_hoy = datetime.now().strftime('%d-%m-%Y')
        
        # Filtrar alertas de hoy
        alertas_hoy = [
            alerta for alerta in self.historial_alertas 
            if alerta.get('fecha') == fecha_hoy
        ]
        
        # Calcular estadísticas
        total_alertas = len(alertas_hoy)
        alertas_criticas = len([a for a in alertas_hoy if a['prioridad'] == 'critica'])
        alertas_altas = len([a for a in alertas_hoy if a['prioridad'] == 'alta'])
        
        # Generar resumen
        resumen = {
            'fecha': fecha_hoy,
            'inventario_actual': inventario,
            'estado_inventario': self.clasificar_estado_inventario(inventario),
            'alertas': {
                'total': total_alertas,
                'criticas': alertas_criticas,
                'altas': alertas_altas,
                'lista': alertas_hoy
            },
            'predicciones': predicciones,
            'recomendaciones_principales': self.extraer_recomendaciones_principales(alertas_hoy),
            'timestamp': datetime.now().isoformat()
        }
        
        return resumen
    
    def clasificar_estado_inventario(self, inventario: int) -> str:
        """Clasifica el estado del inventario"""
        if inventario <= self.umbrales['inventario_critico']:
            return 'CRÍTICO'
        elif inventario <= self.umbrales['inventario_bajo']:
            return 'BAJO'
        elif inventario <= 100:
            return 'NORMAL'
        else:
            return 'ÓPTIMO'
    
    def extraer_recomendaciones_principales(self, alertas: List[Dict]) -> List[str]:
        """Extrae las recomendaciones más importantes de las alertas"""
        todas_recomendaciones = []
        
        for alerta in alertas:
            if 'recomendaciones' in alerta:
                todas_recomendaciones.extend(alerta['recomendaciones'])
        
        # Eliminar duplicados y tomar las primeras 5
        recomendaciones_unicas = list(set(todas_recomendaciones))
        return recomendaciones_unicas[:5]
    
    def enviar_email(self, asunto: str, contenido: str, destinatario: str = None) -> bool:
        """Envía email de notificación"""
        if not self.config['email']['habilitado']:
            print("📧 Email no habilitado en configuración")
            return False
        
        try:
            # Configuración SMTP
            smtp_server = self.config['email']['smtp_server']
            smtp_port = self.config['email']['smtp_port']
            email_origen = self.config['email']['email_origen']
            password = self.config['email']['password']
            email_destino = destinatario or self.config['email']['email_destino']
            
            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = email_origen
            msg['To'] = email_destino
            msg['Subject'] = asunto
            
            # Agregar contenido
            msg.attach(MIMEText(contenido, 'html'))
            
            # Enviar email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_origen, password)
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email enviado a {email_destino}")
            return True
            
        except Exception as e:
            print(f"❌ Error enviando email: {str(e)}")
            return False
    
    def generar_contenido_email(self, resumen: Dict) -> str:
        """Genera contenido HTML para email"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 15px; text-align: center; }}
                .alert {{ background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; }}
                .warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; padding: 10px; margin: 10px 0; }}
                .info {{ background-color: #e3f2fd; border-left: 4px solid #2196F3; padding: 10px; margin: 10px 0; }}
                .success {{ background-color: #e8f5e8; border-left: 4px solid #4CAF50; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Resumen Diario - Aguas Ancud</h1>
                <p>Fecha: {resumen['fecha']}</p>
            </div>
            
            <h2>📦 Estado del Inventario</h2>
            <div class="{'alert' if resumen['estado_inventario'] == 'CRÍTICO' else 'warning' if resumen['estado_inventario'] == 'BAJO' else 'success'}">
                <strong>Inventario Actual:</strong> {resumen['inventario_actual']} bidones
                <br><strong>Estado:</strong> {resumen['estado_inventario']}
            </div>
            
            <h2>🚨 Alertas ({resumen['alertas']['total']})</h2>
            {self.generar_html_alertas(resumen['alertas']['lista'])}
            
            <h2>💡 Recomendaciones Principales</h2>
            <ul>
                {''.join([f'<li>{rec}</li>' for rec in resumen['recomendaciones_principales']])}
            </ul>
            
            <hr>
            <p><em>Este es un reporte automático generado por el sistema de notificaciones inteligentes.</em></p>
        </body>
        </html>
        """
        
        return html
    
    def generar_html_alertas(self, alertas: List[Dict]) -> str:
        """Genera HTML para las alertas"""
        if not alertas:
            return '<div class="info">No hay alertas para hoy.</div>'
        
        html = ""
        for alerta in alertas:
            clase = 'alert' if alerta['prioridad'] == 'critica' else 'warning' if alerta['prioridad'] == 'alta' else 'info'
            html += f"""
            <div class="{clase}">
                <strong>{alerta['mensaje']}</strong><br>
                <strong>Recomendaciones:</strong>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in alerta.get('recomendaciones', [])])}
                </ul>
            </div>
            """
        
        return html

def probar_notificaciones():
    """Función para probar el sistema de notificaciones"""
    print("🔔 PROBANDO SISTEMA DE NOTIFICACIONES")
    print("=" * 50)
    
    notif = NotificacionesInteligentes()
    
    # Simular predicciones
    predicciones = [
        {'fecha': '19-07-2025', 'pedidos': 10, 'bidones': 15},
        {'fecha': '20-07-2025', 'pedidos': 2, 'bidones': 3},
        {'fecha': '21-07-2025', 'pedidos': 7, 'bidones': 10}
    ]
    
    # Verificar alertas
    fecha_prueba = datetime.now() + timedelta(days=1)
    
    print("\n🚨 VERIFICANDO ALERTAS:")
    
    # Alta demanda
    alerta_alta = notif.verificar_alta_demanda(10, fecha_prueba)
    if alerta_alta:
        print(f"   • {alerta_alta['mensaje']}")
        for rec in alerta_alta['recomendaciones']:
            print(f"     - {rec}")
    
    # Baja demanda
    alerta_baja = notif.verificar_baja_demanda(2, fecha_prueba)
    if alerta_baja:
        print(f"   • {alerta_baja['mensaje']}")
        for rec in alerta_baja['recomendaciones']:
            print(f"     - {rec}")
    
    # Inventario
    alertas_inv = notif.verificar_inventario(15)
    for alerta in alertas_inv:
        print(f"   • {alerta['mensaje']}")
        for rec in alerta['recomendaciones']:
            print(f"     - {rec}")
    
    print("\n🎯 SUGERENCIAS DE PROMOCIONES:")
    sugerencias = notif.generar_sugerencias_promociones(2, fecha_prueba)
    for sug in sugerencias:
        print(f"   • {sug['titulo']}: {sug['descripcion']}")
        for accion in sug['acciones']:
            print(f"     - {accion}")
    
    print("\n📊 RESUMEN DIARIO:")
    resumen = notif.generar_resumen_diario(predicciones, 15)
    print(f"   • Inventario: {resumen['inventario_actual']} ({resumen['estado_inventario']})")
    print(f"   • Alertas: {resumen['alertas']['total']} (Críticas: {resumen['alertas']['criticas']})")
    print(f"   • Recomendaciones principales: {len(resumen['recomendaciones_principales'])}")

if __name__ == "__main__":
    probar_notificaciones() 