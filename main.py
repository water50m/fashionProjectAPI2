from fastapi import FastAPI
from app.controllers.data_controller import router as data_router
from app.controllers.prediction_controller import router as predict_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="My API with FastAPI & Classes") # type: ignore
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # หรือใช้ ["*"] ชั่วคราวเพื่อทดสอบ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

# include routers
app.include_router(data_router, prefix="/data", tags=["data"])  # type: ignore
# include routers
app.include_router(predict_router, prefix="/predict", tags=["predict"])  # type: ignore
