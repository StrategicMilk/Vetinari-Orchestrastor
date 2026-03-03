# implementation.py

def add_numbers(a, b):
    """Adds two numbers together."""
    return a + b

def divide_numbers(a, b):
    """Divides a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class Calculator:
    def __init__(self):
        self.history = []

    def calculate(self, operation, x, y):
        result = None
        if operation == "add":
            result = add_numbers(x, y)
        elif operation == "divide":
            result = divide_numbers(x, y)
        
        self.history.append(f"{x} {operation} {y} = {result}")
        return result

    def get_history(self):
        return self.history