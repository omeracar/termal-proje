from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.upload import router as upload_router
from app.api.analyze import router as analyze_router
from app.api.history import router as history_router
from app.database import init_database

app = FastAPI(
    title="Termal Kamera - Kayıp Personel Tespiti",
    description="Klasik termal görüntü analizi, analiz geçmişi ve PDF raporlama prototipi",
    version="0.2.0",
)


@app.on_event("startup")
def startup_event():
    init_database()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(analyze_router)
app.include_router(history_router)


@app.get("/")
def root():
    return {"message": "Termal Kamera Analiz API'si çalışıyor", "version": "0.2.0", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
