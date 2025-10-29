from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

app = FastAPI(title="API Aguas Ancud", version="1.0")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API Aguas Ancud funcionando"}

@app.get("/test")
def test():
    return {"message": "Server is working", "status": "OK"}

@app.get("/kpis")
def get_kpis():
    return {
        "ventas_mes": 150000,
        "ventas_mes_pasado": 120000,
        "total_pedidos_mes": 75,
        "total_pedidos_mes_pasado": 60,
        "total_litros_mes": 1500,
        "litros_vendidos_mes_pasado": 1200,
        "costos_reales": 80000,
        "iva": 15000,
        "punto_equilibrio": 40,
        "clientes_activos": 45,
        "utilidad": 70000,
        "utilidad_mes_pasado": 50000,
        "ticket_promedio_mes_pasado": 2000,
        "clientes_activos_mes_pasado": 40,
        "clientes_inactivos_mes_pasado": 8,
        "iva_mes_pasado": 12000,
        "capacidad_utilizada": 75.5,
        "litros_vendidos": 1500,
        "capacidad_total": 30000
    }

@app.get("/pedidos")
def get_pedidos():
    return []

@app.get("/clientes")
def get_clientes():
    return []

@app.get("/ventas-diarias")
def get_ventas_diarias():
    return {
        "ventas_hoy": 22000,
        "ventas_mismo_dia_mes_anterior": 18000,
        "porcentaje_cambio": 22.2,
        "es_positivo": True,
        "fecha_comparacion": "05-07-2024",
        "tendencia_7_dias": [
            {"fecha": "30-07", "ventas": 15000, "dia_semana": "Mar"},
            {"fecha": "31-07", "ventas": 18000, "dia_semana": "Mie"},
            {"fecha": "01-08", "ventas": 20000, "dia_semana": "Jue"},
            {"fecha": "02-08", "ventas": 19000, "dia_semana": "Vie"},
            {"fecha": "03-08", "ventas": 21000, "dia_semana": "Sab"},
            {"fecha": "04-08", "ventas": 19500, "dia_semana": "Dom"},
            {"fecha": "05-08", "ventas": 22000, "dia_semana": "Lun"}
        ],
        "tipo_comparacion": "mensual"
    }

@app.get("/ventas-semanales")
def get_ventas_semanales():
    return {
        "ventas_semana_actual": 120000,
        "ventas_semana_pasada": 100000,
        "pedidos_semana_actual": 60,
        "pedidos_semana_pasada": 50,
        "porcentaje_cambio": 20.0,
        "es_positivo": True
    }

@app.get("/pedidos-por-horario")
def get_pedidos_por_horario():
    return {
        "pedidos_manana": 25,
        "pedidos_tarde": 35,
        "total": 60,
        "porcentaje_manana": 42,
        "porcentaje_tarde": 58
    }

@app.get("/inventario/estado")
def get_estado_inventario():
    return {
        "stock_actual": 120,
        "stock_minimo": 50,
        "stock_maximo": 200,
        "demanda_diaria_promedio": 8.5,
        "dias_restantes": 14.1,
        "estado": "normal",
        "alertas": [],
        "recomendaciones": []
    }

@app.get("/ventas-historicas")
def get_ventas_historicas():
    return [
        {"name": "Ene", "ventas": 120000},
        {"name": "Feb", "ventas": 135000},
        {"name": "Mar", "ventas": 145000},
        {"name": "Abr", "ventas": 155000},
        {"name": "May", "ventas": 165000},
        {"name": "Jun", "ventas": 175000},
        {"name": "Jul", "ventas": 185000},
        {"name": "Ago", "ventas": 195000}
    ]

@app.get("/ventas-totales-historicas")
def get_ventas_totales_historicas():
    return {
        "ventas_totales": 1275000,
        "total_pedidos": 6375
    }

if __name__ == "__main__":
    print("🚀 INICIANDO SERVIDOR FINAL EN PUERTO 8001...")
    print("✅ Servidor funcionando en http://localhost:8001")
    print("✅ NO SE VA A CERRAR")
    print("✅ Endpoints disponibles:")
    print("   - GET /test")
    print("   - GET /kpis")
    print("   - GET /pedidos")
    print("   - GET /clientes")
    print("   - GET /ventas-diarias")
    print("   - GET /ventas-semanales")
    print("   - GET /pedidos-por-horario")
    print("   - GET /inventario/estado")
    print("   - GET /ventas-historicas")
    print("   - GET /ventas-totales-historicas")
    
    # Configurar para que NO SE CIERRE
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001, 
        log_level="info",
        reload=False,
        access_log=True
    ) 