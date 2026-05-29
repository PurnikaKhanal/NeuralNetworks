import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from inference import predict_base64_image

templates = Jinja2Templates(directory="templates")
class CanvasData(BaseModel):
    image: str

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/")
def home(request: Request):
    print("Received request for home page",request, flush=True)
    print(type(request), flush=True)
    return templates.TemplateResponse(request=request, context={"request": request}, name="index.html")

@app.post("/predict")
async def predict(data: CanvasData):

    prediction = predict_base64_image(data.image)

    return {
        "prediction": prediction
    }