from pydantic import BaseModel
from typing import List


class InputText(BaseModel):
    input_text: str


class StatusInput(BaseModel):
    input_text: str
    employee_details: dict


class VacationInput(BaseModel):
    input_text: str
    employee_details: dict


class CorrectSentence(BaseModel):
    input_text: str
    context: str


class EmployeeData(BaseModel):
    employee_details: List[dict]
    project_details: List[dict]


class Item(BaseModel):
    input_text: str
    employee_details: List[dict]
    project_details: List[dict]


class UserIn(BaseModel):
    username: str
    email: str
    designation: str


class User(BaseModel):
    id: int
    username: str
    email: str
    designation: str
