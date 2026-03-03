# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import click

app = FastAPI()

class User(BaseModel):
    name: str
    email: str

@app.post("/users", response_model=User)
def create_user(user: User):
    return user

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"name": "John Doe", "email": "john.doe@example.com"}

# TODO: Integrate with main application using API routes.