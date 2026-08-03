from pydantic import BaseModel,EmailStr, Field



class User(BaseModel):
    name: str
    age: int
    email: EmailStr
    cgpa: float = Field(ge=0.0, le=4.0, default=2.0, description="CGPA must be between 0.0 and 4.0")




new_user = {
    "name": "John Doe",
    "age": 25,
    "email": "john@example.com",
    "cgpa": 3.8
}

try:
    user = User(**new_user)
    print(user)
except Exception as e:
    print(f"Error: {e}")



# if and email should be optional, define defaults:
# 
# 

from pydantic import BaseModel


class User(BaseModel):
    name: str | None = None
    age: int | None = None
    email: str | None = None
    cgpa: float | None = None


new_user = {'name': 'John Doe', 'age': 25, 'email': 'khan@example.com', 'cgpa': 3.8}

user = User(**new_user)
print(user)

user_dict = user.model_dump() # Convert the model instance to a dictionary
print(user_dict)

user_json = user.model_dump_json() # Convert the model instance to a JSON string
print(user_json)