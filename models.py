"""
Modelos de datos para el nuevo esquema MongoDB
Basado en la migración de MySQL (epedido) a MongoDB (order)
"""

from datetime import datetime
from typing import List, Dict, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field

class PaymentMethod(str, Enum):
    """Métodos de pago disponibles"""
    EFECTIVO = "efectivo"
    TRANSFERENCIA = "transferencia"
    WEBPAY = "webpay"
    MERCADOPAGO = "mercadopago"
    TARJETA = "tarjeta"
    OTRO = "otro"

class OrderStatus(str, Enum):
    """Estados de pedido disponibles"""
    PENDIENTE = "pendiente"
    CONFIRMADO = "confirmado"
    PREPARANDO = "preparando"
    EN_CAMINO = "en_camino"
    ENTREGADO = "entregado"
    RETRASADO = "retrasado"
    DEVUELTO = "devuelto"
    CANCELADO = "cancelado"

class DeliveryType(str, Enum):
    """Tipos de entrega disponibles"""
    DOMICILIO = "domicilio"
    RETIRO = "retiro"
    MOSTRADOR = "mostrador"

class Origin(str, Enum):
    """Orígenes del pedido"""
    APP = "app"
    WEB = "web"
    ADMIN = "admin"
    POS = "pos"

class DayOfWeek(str, Enum):
    """Días de la semana en inglés"""
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"

class ProductItem(BaseModel):
    """Item dentro de un pack"""
    productId: str
    name: str
    quantity: int
    unitPrice: float

class Product(BaseModel):
    """Producto en el pedido"""
    productId: str
    name: str
    unitPrice: float
    quantity: int
    totalPrice: float
    notes: Optional[str] = None
    isPack: bool = False
    items: Optional[List[ProductItem]] = None

class Customer(BaseModel):
    """Información del cliente"""
    id: Optional[str] = None
    name: str
    phone: Optional[str] = None
    address: Optional[str] = None
    block: Optional[str] = None  # depto, block, torre
    lat: Optional[float] = None
    lon: Optional[float] = None
    observations: Optional[str] = None
    notificationToken: Optional[str] = None

class DeliveryPerson(BaseModel):
    """Información del repartidor"""
    id: Optional[str] = None
    name: Optional[str] = None
    token: Optional[str] = None

class DeliverySchedule(BaseModel):
    """Horario de entrega"""
    day: Optional[DayOfWeek] = None
    hour: Optional[str] = None  # Formato: "09:00 AM"

class Rating(BaseModel):
    """Calificación del pedido"""
    value: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = None

class Order(BaseModel):
    """Modelo principal de pedido (order)"""
    # Identificadores
    id: Optional[str] = Field(None, alias="_id")
    storeId: str
    commerceId: str
    orderCode: Optional[str] = None  # Código de pedido legible
    legacyId: Optional[int] = None  # ID del sistema anterior
    
    # Precios
    price: float
    finalPrice: float
    
    # Fechas
    createdAt: datetime
    updatedAt: datetime
    deliveryDate: Optional[datetime] = None
    deliveredAt: Optional[datetime] = None
    
    # Productos
    products: List[Product]
    
    # Pago
    paymentMethod: PaymentMethod
    transferPay: bool = False
    
    # Estado
    status: OrderStatus
    
    # Cliente
    customer: Customer
    
    # Entrega
    deliveryType: DeliveryType
    deliveryPerson: Optional[DeliveryPerson] = None
    deliverySchedule: Optional[DeliverySchedule] = None
    
    # Observaciones
    merchantObservation: Optional[str] = None
    deliveryObservation: Optional[str] = None
    
    # Calificación
    rating: Optional[Rating] = None
    
    # Origen
    origin: Origin = Origin.APP
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class OrderCreate(BaseModel):
    """Modelo para crear un nuevo pedido"""
    storeId: str
    commerceId: str
    price: float
    finalPrice: Optional[float] = None
    products: List[Product]
    paymentMethod: PaymentMethod
    transferPay: bool = False
    status: OrderStatus = OrderStatus.PENDIENTE
    customer: Customer
    deliveryType: DeliveryType
    deliveryPerson: Optional[DeliveryPerson] = None
    deliverySchedule: Optional[DeliverySchedule] = None
    merchantObservation: Optional[str] = None
    deliveryObservation: Optional[str] = None
    origin: Origin = Origin.APP
    deliveryDate: Optional[datetime] = None

class OrderUpdate(BaseModel):
    """Modelo para actualizar un pedido existente"""
    price: Optional[float] = None
    finalPrice: Optional[float] = None
    products: Optional[List[Product]] = None
    paymentMethod: Optional[PaymentMethod] = None
    transferPay: Optional[bool] = None
    status: Optional[OrderStatus] = None
    customer: Optional[Customer] = None
    deliveryType: Optional[DeliveryType] = None
    deliveryPerson: Optional[DeliveryPerson] = None
    deliverySchedule: Optional[DeliverySchedule] = None
    merchantObservation: Optional[str] = None
    deliveryObservation: Optional[str] = None
    deliveryDate: Optional[datetime] = None
    deliveredAt: Optional[datetime] = None
    rating: Optional[Rating] = None

class OrderResponse(BaseModel):
    """Modelo de respuesta para pedidos"""
    _id: str
    storeId: str
    commerceId: str
    orderCode: Optional[str] = None
    legacyId: Optional[int] = None
    price: float
    finalPrice: float
    createdAt: str
    updatedAt: str
    deliveryDate: Optional[str] = None
    deliveredAt: Optional[str] = None
    products: List[Product]
    paymentMethod: str
    transferPay: bool
    status: str
    customer: Customer
    deliveryType: str
    deliveryPerson: Optional[DeliveryPerson] = None
    deliverySchedule: Optional[DeliverySchedule] = None
    merchantObservation: Optional[str] = None
    deliveryObservation: Optional[str] = None
    rating: Optional[Rating] = None
    origin: str

class OrderStats(BaseModel):
    """Estadísticas de pedidos"""
    total_orders: int
    total_revenue: float
    orders_by_status: Dict[str, int]
    orders_by_payment_method: Dict[str, int]
    orders_by_delivery_type: Dict[str, int]
    average_order_value: float
    orders_today: int
    orders_this_month: int

class CustomerStats(BaseModel):
    """Estadísticas de clientes"""
    total_customers: int
    active_customers: int
    inactive_customers: int
    customers_by_zone: Dict[str, int]
    average_orders_per_customer: float
    top_customers: List[Dict[str, Union[str, int, float]]]

# Funciones de utilidad para conversión

def convert_legacy_order(legacy_data: Dict) -> Order:
    """Convierte datos del esquema antiguo al nuevo"""
    
    # Parsear fechas
    created_at = datetime.now()
    if legacy_data.get('fecha'):
        try:
            created_at = datetime.strptime(legacy_data['fecha'], "%d-%m-%Y")
        except:
            pass
    
    delivery_date = None
    if legacy_data.get('dia') and legacy_data.get('mes') and legacy_data.get('ano'):
        try:
            delivery_date = datetime(
                int(legacy_data['ano']),
                int(legacy_data['mes']),
                int(legacy_data['dia'])
            )
        except:
            pass
    
    # Convertir precios
    price = float(legacy_data.get('precio', 0))
    final_price = float(legacy_data.get('pagofinal', 0)) or price
    
    # Productos
    products = []
    if legacy_data.get('ordenpedido'):
        products.append(Product(
            productId="bidon_agua_20l",
            name="Bidón de Agua 20L",
            unitPrice=2000,
            quantity=1,
            totalPrice=price,
            notes=legacy_data.get('ordenpedido', '')[:100]
        ))
    else:
        products.append(Product(
            productId="bidon_agua_20l",
            name="Bidón de Agua 20L",
            unitPrice=2000,
            quantity=1,
            totalPrice=price
        ))
    
    # Cliente
    customer = Customer(
        id=legacy_data.get('usuario', ''),
        name=legacy_data.get('usuario', ''),
        phone=legacy_data.get('telefonou', ''),
        address=legacy_data.get('dire', ''),
        block=legacy_data.get('deptoblock', ''),
        lat=float(legacy_data.get('lat', 0)) if legacy_data.get('lat') else None,
        lon=float(legacy_data.get('lon', 0)) if legacy_data.get('lon') else None,
        observations=legacy_data.get('observacion', ''),
        notificationToken=legacy_data.get('notific', '')
    )
    
    # Repartidor
    delivery_person = None
    if legacy_data.get('userdelivery') or legacy_data.get('despachador'):
        delivery_person = DeliveryPerson(
            id=legacy_data.get('userdelivery', ''),
            name=legacy_data.get('despachador', '')
        )
    
    # Horario de entrega
    delivery_schedule = None
    if delivery_date or legacy_data.get('horaagenda'):
        delivery_schedule = DeliverySchedule(
            day=DayOfWeek(delivery_date.strftime('%A').lower()) if delivery_date else None,
            hour=legacy_data.get('horaagenda', '09:00 AM')
        )
    
    # Calificación
    rating = None
    if legacy_data.get('calific'):
        try:
            calificacion = str(legacy_data['calific']).strip()
            if calificacion.isdigit():
                rating = Rating(value=int(calificacion))
            else:
                rating = Rating(comment=calificacion)
        except:
            pass
    
    return Order(
        storeId="aguas_ancud",
        commerceId="ancud_001",
        orderCode=legacy_data.get('idpedido', ''),
        legacyId=int(legacy_data.get('id', 0)) if legacy_data.get('id') else None,
        price=price,
        finalPrice=final_price,
        createdAt=created_at,
        updatedAt=datetime.now(),
        deliveryDate=delivery_date,
        products=products,
        paymentMethod=PaymentMethod(legacy_data.get('metodopago', 'efectivo').lower()),
        transferPay=bool(legacy_data.get('transferpay', False)),
        status=OrderStatus(legacy_data.get('status', 'pendiente').lower()),
        customer=customer,
        deliveryType=DeliveryType('retiro' if legacy_data.get('retirolocal') == '1' else 'domicilio'),
        deliveryPerson=delivery_person,
        deliverySchedule=delivery_schedule,
        merchantObservation=legacy_data.get('observaciondos', ''),
        deliveryObservation=legacy_data.get('observaciondos', ''),
        rating=rating,
        origin=Origin.APP
    )

def order_to_response(order: Order) -> OrderResponse:
    """Convierte un Order a OrderResponse para la API"""
    return OrderResponse(
        _id=str(order._id) if order._id else "",
        storeId=order.storeId,
        commerceId=order.commerceId,
        orderCode=order.orderCode,
        legacyId=order.legacyId,
        price=order.price,
        finalPrice=order.finalPrice,
        createdAt=order.createdAt.isoformat(),
        updatedAt=order.updatedAt.isoformat(),
        deliveryDate=order.deliveryDate.isoformat() if order.deliveryDate else None,
        deliveredAt=order.deliveredAt.isoformat() if order.deliveredAt else None,
        products=order.products,
        paymentMethod=order.paymentMethod.value,
        transferPay=order.transferPay,
        status=order.status.value,
        customer=order.customer,
        deliveryType=order.deliveryType.value,
        deliveryPerson=order.deliveryPerson,
        deliverySchedule=order.deliverySchedule,
        merchantObservation=order.merchantObservation,
        deliveryObservation=order.deliveryObservation,
        rating=order.rating,
        origin=order.origin.value
    )
