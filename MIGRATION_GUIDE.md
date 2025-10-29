# Guía de Migración: MySQL (epedido) → MongoDB (order)

## Resumen de la Migración

Este documento describe la migración del esquema de datos de la tabla `epedido` en MySQL al nuevo esquema `order` en MongoDB, basado en la tabla comparativa proporcionada.

## Archivos Creados/Modificados

### 1. Script de Migración
- **Archivo**: `migration_script.py`
- **Propósito**: Script completo para migrar datos del esquema antiguo al nuevo
- **Uso**: `python migration_script.py`

### 2. Modelos de Datos
- **Archivo**: `models.py`
- **Propósito**: Define los nuevos modelos Pydantic para el esquema MongoDB
- **Incluye**: Enums, validaciones, funciones de conversión

### 3. Backend Actualizado
- **Archivo**: `main.py`
- **Cambios**: 
  - Nuevo endpoint `/pedidos-v2` con esquema MongoDB
  - Mantiene compatibilidad con endpoint legacy `/pedidos`
  - Importa nuevos modelos

### 4. Frontend Actualizado
- **Archivo**: `frontend/src/services/api.js`
- **Cambios**: Nueva función `getPedidosV2()` para el nuevo esquema
- **Archivo**: `frontend/src/pages/Pedidos.jsx`
- **Cambios**: Compatible con ambos esquemas (legacy y nuevo)

### 5. Datos de Ejemplo
- **Archivo**: `orders_migrated.json`
- **Propósito**: Ejemplos de pedidos en el nuevo formato MongoDB

## Mapeo de Campos

### Campos Principales
| Campo Antiguo (MySQL) | Campo Nuevo (MongoDB) | Transformación |
|----------------------|----------------------|----------------|
| `id` | `_id` | ObjectId automático + `legacyId` |
| `idpedido` | `orderCode` | String preservado |
| `precio` | `price` | varchar → Number |
| `pagofinal` | `finalPrice` | varchar → Number |
| `fecha + hora` | `createdAt` | Unión a Date ISO |
| `dia/mes/ano` | `deliveryDate` | Combinación a Date |
| `ordenpedido` | `products[]` | Parseo a array de productos |

### Campos de Cliente
| Campo Antiguo | Campo Nuevo | Transformación |
|---------------|-------------|----------------|
| `usuario` | `customer.name` | String preservado |
| `telefonou` | `customer.phone` | String preservado |
| `dire` | `customer.address` | String preservado |
| `lat/lon` | `customer.lat/lon` | varchar → Number |
| `observacion` | `customer.observations` | String preservado |

### Campos de Entrega
| Campo Antiguo | Campo Nuevo | Transformación |
|---------------|-------------|----------------|
| `retirolocal` | `deliveryType` | Mapeo a enum |
| `userdelivery` | `deliveryPerson.id` | String preservado |
| `despachador` | `deliveryPerson.name` | String preservado |
| `horaagenda` | `deliverySchedule.hour` | String preservado |

### Campos de Pago y Estado
| Campo Antiguo | Campo Nuevo | Transformación |
|---------------|-------------|----------------|
| `metodopago` | `paymentMethod` | Normalización a enum |
| `transferpay` | `transferPay` | String → Boolean |
| `status` | `status` | Normalización a enum |

## Nuevos Campos en MongoDB

### Campos Agregados
- `storeId`: Identificador de la tienda
- `commerceId`: Identificador del comercio
- `origin`: Origen del pedido (app, web, admin, pos)
- `products[].isPack`: Indica si es un pack
- `products[].items[]`: Items dentro de un pack
- `rating.comment`: Comentario de calificación
- `deliverySchedule.day`: Día de la semana en inglés
- `deliveredAt`: Fecha real de entrega
- `updatedAt`: Timestamp de actualización

### Enums Definidos
```python
class PaymentMethod(str, Enum):
    EFECTIVO = "efectivo"
    TRANSFERENCIA = "transferencia"
    WEBPAY = "webpay"
    MERCADOPAGO = "mercadopago"
    TARJETA = "tarjeta"
    OTRO = "otro"

class OrderStatus(str, Enum):
    PENDIENTE = "pendiente"
    CONFIRMADO = "confirmado"
    PREPARANDO = "preparando"
    EN_CAMINO = "en_camino"
    ENTREGADO = "entregado"
    RETRASADO = "retrasado"
    DEVUELTO = "devuelto"
    CANCELADO = "cancelado"

class DeliveryType(str, Enum):
    DOMICILIO = "domicilio"
    RETIRO = "retiro"
    MOSTRADOR = "mostrador"
```

## Campos Eliminados

Los siguientes campos del esquema antiguo no tienen equivalente directo en el nuevo esquema:

- `fechamostrar`: Campo de presentación (se formatea al vuelo)
- `tokendelivery`: No incluido en el esquema actual
- `nombrelocal/logo`: Deben ir en colección de tiendas
- `prov`: Integrado en `customer.address`
- `comuna`: Integrado en `customer.address`

## Instrucciones de Uso

### 1. Ejecutar Migración
```bash
cd theycallmebitch/backend
python migration_script.py
```

### 2. Verificar Datos Migrados
El script generará:
- `orders_migrated.json`: Datos migrados
- Estadísticas de migración en consola

### 3. Usar Nuevo Endpoint
```javascript
// Frontend
import { getPedidosV2 } from '../services/api';

const pedidos = await getPedidosV2();
```

### 4. Compatibilidad
El sistema mantiene compatibilidad con el esquema legacy:
- Endpoint `/pedidos` sigue funcionando
- Frontend detecta automáticamente el esquema disponible
- Fallback automático si el nuevo esquema no está disponible

## Validaciones y Transformaciones

### Fechas
- Formato DD-MM-YYYY → ISO 8601
- Unión de fecha + hora cuando esté disponible
- Manejo de fechas inválidas

### Precios
- Conversión de varchar a Number
- Manejo de valores nulos o inválidos
- Preservación de decimales

### Coordenadas
- Conversión de varchar a Number (float)
- Validación de rangos válidos

### Productos
- Parseo de `ordenpedido` a array estructurado
- Detección de packs vs productos individuales
- Cálculo de precios totales

## Consideraciones de Rendimiento

### Optimizaciones Implementadas
- Procesamiento por lotes
- Manejo de errores individuales
- Logging detallado
- Estadísticas de migración

### Recomendaciones
- Ejecutar migración en horarios de bajo tráfico
- Hacer backup de datos antes de migrar
- Validar datos migrados antes de activar en producción
- Monitorear rendimiento del nuevo esquema

## Próximos Pasos

1. **Validación**: Revisar datos migrados
2. **Testing**: Probar endpoints nuevos
3. **Despliegue**: Activar nuevo esquema gradualmente
4. **Monitoreo**: Supervisar rendimiento
5. **Limpieza**: Remover código legacy cuando sea seguro

## Soporte

Para dudas o problemas con la migración:
1. Revisar logs del script de migración
2. Verificar formato de datos de entrada
3. Consultar este documento
4. Revisar ejemplos en `orders_migrated.json`





