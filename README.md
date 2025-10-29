# Backend - Aguas Ancud API (FastAPI)

Este backend expone endpoints REST para el dashboard de Aguas Ancud.

## Endpoints principales
- `/pedidos` : Devuelve todos los pedidos de Aguas Ancud
- `/clientes` : Devuelve todos los clientes
- `/kpis` : Devuelve KPIs principales del mes

## Cómo correr

1. Instala dependencias:
   ```
   pip install fastapi uvicorn pandas requests
   ```
2. Ejecuta el servidor:
   ```
   uvicorn main:app --reload
   ```

## Estructura
- `main.py` : Código principal de la API
- `requirements.txt` : Dependencias
- `README.md` : Esta guía 