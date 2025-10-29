import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self):
        self.factores_entrenados = self.cargar_factores()
        self.datos_historicos = self.obtener_datos_historicos()
        self.modelos = {}
        self.scaler = StandardScaler()
        self.clusters_clientes = None
        
    def cargar_factores(self):
        """Carga los factores entrenados"""
        try:
            with open('factores_entrenados.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def obtener_datos_historicos(self):
        """Obtiene y procesa datos históricos para ML"""
        try:
            response = requests.get("https://fluvi.cl/fluviDos/GoApp/endpoints/pedidos.php", 
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            pedidos = response.json()
            
            # Filtrar Aguas Ancud
            pedidos_ancud = []
            for pedido in pedidos:
                if pedido.get('nombrelocal') == 'Aguas Ancud':
                    try:
                        fecha_str = pedido.get('fecha', '')
                        if fecha_str:
                            fecha = datetime.strptime(fecha_str, "%d-%m-%Y")
                            pedido['fecha_parsed'] = fecha
                            pedidos_ancud.append(pedido)
                    except:
                        continue
            
            return pedidos_ancud
        except:
            return []
    
    def preparar_datos_ml(self):
        """Prepara datos para Machine Learning"""
        if not self.datos_historicos:
            return None, None
        
        # Crear DataFrame
        df = pd.DataFrame(self.datos_historicos)
        
        # Agrupar por fecha
        df['fecha_parsed'] = pd.to_datetime(df['fecha_parsed'])
        df['fecha'] = df['fecha_parsed'].dt.date
        
        # Contar pedidos por día
        pedidos_por_dia = df.groupby('fecha').size().reset_index(name='pedidos')
        pedidos_por_dia['fecha'] = pd.to_datetime(pedidos_por_dia['fecha'])
        
        # Crear características
        X = []
        y = []
        
        for _, row in pedidos_por_dia.iterrows():
            fecha = row['fecha']
            pedidos = row['pedidos']
            
            # Características temporales
            features = [
                fecha.weekday(),  # Día de la semana (0-6)
                fecha.month,      # Mes (1-12)
                fecha.day,        # Día del mes (1-31)
                fecha.isocalendar()[1],  # Semana del año
                (fecha.day - 1) // 7 + 1,  # Semana del mes
                fecha.quarter,    # Trimestre
                int(fecha.isocalendar()[1] > 26),  # Fin de año
                int(fecha.month in [12, 1, 2]),  # Verano
                int(fecha.month in [6, 7, 8]),  # Invierno
                int(fecha.month in [3, 4, 5]),  # Otoño
                int(fecha.month in [9, 10, 11]),  # Primavera
            ]
            
            # Características de tendencia (últimos 7 días)
            if len(pedidos_por_dia) >= 7:
                fecha_7_dias = fecha - timedelta(days=7)
                pedidos_7_dias = pedidos_por_dia[pedidos_por_dia['fecha'] == fecha_7_dias]['pedidos'].values
                if len(pedidos_7_dias) > 0:
                    features.append(pedidos_7_dias[0])
                else:
                    features.append(0)
            else:
                features.append(0)
            
            # Características de tendencia (últimos 14 días)
            if len(pedidos_por_dia) >= 14:
                fecha_14_dias = fecha - timedelta(days=14)
                pedidos_14_dias = pedidos_por_dia[pedidos_por_dia['fecha'] == fecha_14_dias]['pedidos'].values
                if len(pedidos_14_dias) > 0:
                    features.append(pedidos_14_dias[0])
                else:
                    features.append(0)
            else:
                features.append(0)
            
            X.append(features)
            y.append(pedidos)
        
        return np.array(X), np.array(y)
    
    def entrenar_modelos_ml(self):
        """Entrena múltiples modelos de Machine Learning"""
        X, y = self.preparar_datos_ml()
        
        if X is None or len(X) < 10:
            print("❌ Datos insuficientes para entrenar modelos ML")
            return False
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modelos
        modelos = {
            'regresion_lineal': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'red_neuronal': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
        }
        
        resultados = {}
        
        for nombre, modelo in modelos.items():
            try:
                # Entrenar modelo
                if nombre == 'red_neuronal':
                    modelo.fit(X_train_scaled, y_train)
                    y_pred = modelo.predict(X_test_scaled)
                else:
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)
                
                # Evaluar modelo
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                resultados[nombre] = {
                    'modelo': modelo,
                    'mse': mse,
                    'r2': r2,
                    'precision': max(0, r2 * 100)
                }
                
                print(f"✅ {nombre}: R² = {r2:.3f}, Precisión = {r2*100:.1f}%")
                
            except Exception as e:
                print(f"❌ Error entrenando {nombre}: {str(e)}")
        
        self.modelos = resultados
        return len(resultados) > 0
    
    def clustering_clientes(self):
        """Realiza clustering de clientes"""
        if not self.datos_historicos:
            return None
        
        # Preparar datos de clientes
        df = pd.DataFrame(self.datos_historicos)
        
        # Agrupar por cliente
        clientes_data = []
        for cliente in df['nombrecliente'].unique():
            pedidos_cliente = df[df['nombrecliente'] == cliente]
            
            # Características del cliente
            total_pedidos = len(pedidos_cliente)
            if total_pedidos == 0:
                continue
                
            # Calcular características
            fechas = pd.to_datetime(pedidos_cliente['fecha_parsed'])
            dias_activo = (fechas.max() - fechas.min()).days + 1
            frecuencia = total_pedidos / max(1, dias_activo)
            
            # Promedio de pedidos por día
            pedidos_por_dia = pedidos_cliente.groupby(pedidos_cliente['fecha_parsed'].dt.date).size()
            promedio_pedidos = pedidos_por_dia.mean()
            
            # Variabilidad
            variabilidad = pedidos_por_dia.std() if len(pedidos_por_dia) > 1 else 0
            
            clientes_data.append([
                total_pedidos,
                dias_activo,
                frecuencia,
                promedio_pedidos,
                variabilidad
            ])
        
        if len(clientes_data) < 3:
            return None
        
        # Clustering
        try:
            kmeans = KMeans(n_clusters=min(4, len(clientes_data)), random_state=42)
            clusters = kmeans.fit_predict(clientes_data)
            
            # Analizar clusters
            clientes_df = pd.DataFrame(clientes_data, columns=[
                'total_pedidos', 'dias_activo', 'frecuencia', 'promedio_pedidos', 'variabilidad'
            ])
            clientes_df['cluster'] = clusters
            
            # Caracterizar clusters
            caracterizacion = {}
            for cluster in range(kmeans.n_clusters_):
                cluster_data = clientes_df[clientes_df['cluster'] == cluster]
                caracterizacion[cluster] = {
                    'tamaño': len(cluster_data),
                    'promedio_pedidos': cluster_data['total_pedidos'].mean(),
                    'frecuencia_promedio': cluster_data['frecuencia'].mean(),
                    'tipo': self.caracterizar_cluster(cluster_data)
                }
            
            self.clusters_clientes = {
                'modelo': kmeans,
                'caracterizacion': caracterizacion,
                'datos': clientes_df
            }
            
            print(f"✅ Clustering completado: {kmeans.n_clusters_} clusters identificados")
            return True
            
        except Exception as e:
            print(f"❌ Error en clustering: {str(e)}")
            return False
    
    def caracterizar_cluster(self, cluster_data):
        """Caracteriza un cluster de clientes"""
        avg_pedidos = cluster_data['total_pedidos'].mean()
        avg_frecuencia = cluster_data['frecuencia'].mean()
        
        if avg_pedidos > 20 and avg_frecuencia > 0.1:
            return "VIP"
        elif avg_pedidos > 10 and avg_frecuencia > 0.05:
            return "Recurrente"
        elif avg_pedidos > 5:
            return "Regular"
        else:
            return "Ocasional"
    
    def predecir_ml(self, fecha: datetime, modelo_preferido: str = 'mejor'):
        """Predicción usando Machine Learning"""
        if not self.modelos:
            print("❌ No hay modelos entrenados")
            return None
        
        # Preparar características para la fecha
        features = [
            fecha.weekday(),
            fecha.month,
            fecha.day,
            fecha.isocalendar()[1],
            (fecha.day - 1) // 7 + 1,
            fecha.quarter,
            int(fecha.isocalendar()[1] > 26),
            int(fecha.month in [12, 1, 2]),
            int(fecha.month in [6, 7, 8]),
            int(fecha.month in [3, 4, 5]),
            int(fecha.month in [9, 10, 11]),
            0,  # Valor por defecto para tendencia 7 días
            0   # Valor por defecto para tendencia 14 días
        ]
        
        # Seleccionar modelo
        if modelo_preferido == 'mejor':
            mejor_modelo = max(self.modelos.keys(), key=lambda x: self.modelos[x]['r2'])
            modelo = self.modelos[mejor_modelo]['modelo']
            nombre_modelo = mejor_modelo
        elif modelo_preferido in self.modelos:
            modelo = self.modelos[modelo_preferido]['modelo']
            nombre_modelo = modelo_preferido
        else:
            print(f"❌ Modelo '{modelo_preferido}' no encontrado")
            return None
        
        # Hacer predicción
        try:
            if nombre_modelo == 'red_neuronal':
                features_scaled = self.scaler.transform([features])
                prediccion = modelo.predict(features_scaled)[0]
            else:
                prediccion = modelo.predict([features])[0]
            
            # Asegurar predicción positiva
            prediccion = max(0, prediccion)
            
            return {
                'fecha': fecha.strftime("%d-%m-%Y"),
                'prediccion': round(prediccion, 1),
                'modelo_usado': nombre_modelo,
                'precision_modelo': self.modelos[nombre_modelo]['precision'],
                'bidones_estimados': round(prediccion * 1.5, 1),
                'ingresos_estimados': round(prediccion * 1.5 * 2000, 0)
            }
            
        except Exception as e:
            print(f"❌ Error en predicción ML: {str(e)}")
            return None
    
    def guardar_modelos(self):
        """Guarda los modelos entrenados"""
        try:
            modelos_guardados = {}
            for nombre, info in self.modelos.items():
                modelos_guardados[nombre] = {
                    'precision': info['precision'],
                    'r2': info['r2'],
                    'mse': info['mse']
                }
            
            with open('modelos_ml.json', 'w', encoding='utf-8') as f:
                json.dump(modelos_guardados, f, ensure_ascii=False, indent=2)
            
            # Guardar modelos con pickle
            with open('modelos_ml.pkl', 'wb') as f:
                pickle.dump(self.modelos, f)
            
            print("✅ Modelos ML guardados")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando modelos: {str(e)}")
            return False
    
    def cargar_modelos(self):
        """Carga modelos guardados"""
        try:
            with open('modelos_ml.pkl', 'rb') as f:
                self.modelos = pickle.load(f)
            print("✅ Modelos ML cargados")
            return True
        except:
            print("❌ No se encontraron modelos guardados")
            return False

def entrenar_ml_completo():
    """Entrena todos los modelos de Machine Learning"""
    print("🤖 INICIANDO ENTRENAMIENTO DE MACHINE LEARNING")
    print("=" * 60)
    
    ml_predictor = MLPredictor()
    
    # Entrenar modelos
    print("\n📊 Entrenando modelos de predicción...")
    if ml_predictor.entrenar_modelos_ml():
        print("✅ Modelos entrenados exitosamente")
    else:
        print("❌ Error entrenando modelos")
        return
    
    # Clustering de clientes
    print("\n👥 Realizando clustering de clientes...")
    if ml_predictor.clustering_clientes():
        print("✅ Clustering completado")
    else:
        print("⚠️ Clustering no disponible")
    
    # Guardar modelos
    print("\n💾 Guardando modelos...")
    ml_predictor.guardar_modelos()
    
    # Probar predicción
    print("\n🧪 Probando predicción ML...")
    fecha_prueba = datetime.now() + timedelta(days=1)
    prediccion = ml_predictor.predecir_ml(fecha_prueba)
    
    if prediccion:
        print(f"📅 Fecha: {prediccion['fecha']}")
        print(f"📦 Predicción: {prediccion['prediccion']} pedidos")
        print(f"🛢️ Bidones: {prediccion['bidones_estimados']}")
        print(f"💰 Ingresos: ${prediccion['ingresos_estimados']:,}")
        print(f"🤖 Modelo: {prediccion['modelo_usado']}")
        print(f"🎯 Precisión: {prediccion['precision_modelo']:.1f}%")
    
    print("\n✅ Entrenamiento ML completado")

if __name__ == "__main__":
    entrenar_ml_completo() 