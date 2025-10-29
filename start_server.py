import subprocess
import time
import sys
import os

def start_server():
    """Inicia el servidor de forma robusta y lo mantiene vivo"""
    print("🚀 INICIANDO SERVIDOR BACKEND - AGUAS ANCUD")
    print("=" * 50)
    
    while True:
        try:
            print(f"\n⏰ {time.strftime('%H:%M:%S')} - Iniciando servidor...")
            
            # Comando para iniciar uvicorn
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "main:app", 
                "--host", "0.0.0.0", 
                "--port", "8001",
                "--reload"
            ]
            
            # Ejecutar el servidor
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            print("✅ Servidor iniciado correctamente en http://localhost:8001")
            print("📊 Endpoints disponibles:")
            print("   - /kpis - KPIs principales")
            print("   - /pedidos - Lista de pedidos")
            print("   - /clientes - Lista de clientes")
            print("   - /test - Test de conectividad")
            print("\n🔄 El servidor se reiniciará automáticamente si se cae")
            print("⏹️  Presiona Ctrl+C para detener")
            
            # Esperar a que el proceso termine
            process.wait()
            
        except KeyboardInterrupt:
            print("\n🛑 Servidor detenido por el usuario")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("🔄 Reiniciando en 5 segundos...")
            time.sleep(5)
            continue

if __name__ == "__main__":
    start_server()
