#!/usr/bin/env python3
"""
Script de prueba simple para verificar que el servidor funcione
"""

from fastapi import FastAPI
import uvicorn

# Crear una app simple de prueba
app = FastAPI(title="Test Server")

@app.get("/")
def read_root():
    return {"message": "Servidor funcionando correctamente"}

@app.get("/test")
def test_endpoint():
    return {"status": "OK", "message": "Backend funcionando"}

if __name__ == "__main__":
    print("🚀 Iniciando servidor de prueba...")
    print("📍 URL: http://localhost:8001")
    print("🧪 Test: http://localhost:8001/test")
    uvicorn.run(app, host="127.0.0.1", port=8001)