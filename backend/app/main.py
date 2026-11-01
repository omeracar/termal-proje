from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.upload import router as upload_router
from app.api.analyze import router as analyze_router

app = FastAPI(
    title="Termal Kamera - Kayıp Personel Tespiti",
    description="Termal görüntü analizi, insan tespiti ve yaşam belirtisi değerlendirme sistemi",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(analyze_router)

@app.get("/")
def root():
    return {
        "message": "Termal Kamera Analiz API'si çalışıyor",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
