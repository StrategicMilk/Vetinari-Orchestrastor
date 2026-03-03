import fastapi
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from typing import List

from .db import get_db
from .models import OrderModel, UserModel

class Order(BaseModel):
    id: int = Field(..., description="Order ID")
    customer_id: int = Field(..., description="Customer ID")
    status: str = Field(..., description="Status")

class User(BaseModel):
    id: int = Field(..., description="User ID")
    name: str = Field(..., description="Name")
    email: str = Field(..., description="Email")

@fastapi.post("/orders", response_model=Order)
async def create_order(order_data: OrderData):
    """Creates a new order"""
    db_engine = get_db()
    order = create_order_in_database(db_engine, order_data)
    return {"status": "OK"}

@fastapi.get("/orders/{order_id}", response_model=Order)
async def read_order(order_id: int):
    """Reads an existing order"""
    db_engine = get_db()
    order = OrderModel.query.get(order_id)
    if order:
        return {"status": "OK", "data": order.dict()}
    return {"status": "NOT_FOUND"}

@fastapi.post("/orders/{order_id}/update", response_model=Order)
async def update_order(order_id: int, order_data: UpdateOrder):
    """Updates an existing order"""
    db_engine = get_db()
    order = OrderModel.query.get(order_id)
    if order:
        update_order_in_database(order, order_data)
        return {"status": "OK"}
    return {"status": "NOT_FOUND"}

@fastapi.post("/orders/{order_id}/delete", response_model=Order)
async def delete_order(order_id: int):
    """Deletes an existing order"""
    db_engine = get_db()
    OrderModel.query.get(order_id).delete()
    db_engine.commit()
    return {"status": "OK"}