from pydantic import BaseModel, Field
from typing import Literal


class Inputdata(BaseModel):
    age:float=Field(...,ge=18,description='Age of the person')
    gender:Literal['male','female']=Field(...,description='Gender of the person')
    bmi:float=Field(...,ge=1,description='BMI of the person (an person weight is good or not according to their heigh)')
    bloodpressure:int=Field()
    diabetic:Literal['Yes','No']=Field(...,description='If ther person has diabetic or not')
    children:int=Field(...,ge=0,description='Number of children of the person')
    smoker:Literal['Yes','No']=Field(...,description='person is smokre ot not')
    region:Literal['southeast','southwest','northeast','northwest']=Field(...,description='region of the person')

class response(BaseModel):
    claim:float