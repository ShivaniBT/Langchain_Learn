from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Abhay"
    age: Optional[int] = None
    email: Optional[EmailStr] = None
    cgpa: float = Field(gt=0, lt=10, default=7.5, description= "A decimal numerber from 0-10 , showing academic performance of student") #Feild function to add validations

# new_student={}

#coerce - will convert the types if possible
new_student={
    "name": "Shivani",
    "age": "21",  # string instead of int
    "email":"shivani@m.com",
    "cgpa": 9.1
}



student = Student(**new_student)

print(student)

# student_dict = dict(student)
student_dict = student.model_dump()
print(student_dict)

student_json= student.model_dump_json()
print(student_json)