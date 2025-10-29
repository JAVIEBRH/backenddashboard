from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="API Aguas Ancud Ultra Simple", version="1.0")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
def test_endpoint():
    return {"message": "Server is working", "ventas_hoy": 22000}

@app.get("/kpis")
def get_kpis():
    """KPIs simulados para testing"""
    return {
        "ventas_mes": 2500000,
        "ventas_mes_pasado": 2200000,
        "total_pedidos_mes": 1250,
        "total_pedidos_mes_pasado": 1100,
        "total_litros_mes": 25000,
        "litros_vendidos_mes_pasado": 22000,
        "costos_reales": 265000,
        "iva": 475000,
        "iva_mes_pasado": 418000,
        "utilidad": 2235000,
        "utilidad_mes_pasado": 1935000,
        "ticket_promedio_mes_pasado": 2000,
        "clientes_activos_mes_pasado": 150,
        "clientes_inactivos_mes_pasado": 30,
        "punto_equilibrio": 130,
        "clientes_activos": 180,
        "capacidad_utilizada": 83.3,
        "litros_vendidos": 25000,
        "capacidad_total": 30000,
    }

if __name__ == "__main__":
    print("🚀 Iniciando servidor en puerto 8002...")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info") 