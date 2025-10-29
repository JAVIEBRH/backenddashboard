from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="API Aguas Ancud - Test CORS", version="1.0")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:5174", 
        "http://localhost:5175", 
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API Aguas Ancud funcionando - Test CORS"}

@app.get("/test")
def test():
    return {"message": "Server is working", "status": "OK"}

@app.get("/kpis")
def get_kpis():
    """KPIs de prueba para verificar CORS"""
    return {
        "ventas_mes": 150000,
        "ventas_mes_pasado": 120000,
        "total_pedidos_mes": 45,
        "total_pedidos_mes_pasado": 38,
        "total_litros_mes": 1500,
        "litros_vendidos_mes_pasado": 1200,
        "costos_reales": 45000,
        "iva": 28500,
        "punto_equilibrio": 75000,
        "clientes_activos": 25,
        "test": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
