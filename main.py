from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ----- CORS FIX -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://olive-prediction-frontend.vercel.app"
    ],  # ❗ PAS DE / À LA FIN
    allow_credentials=True,
    allow_methods=["*"],   # obligatoire pour OPTIONS
    allow_headers=["*"],   # obligatoire pour OPTIONS
)

# ----- OPTIONS FIX -----
@app.options("/predict")
async def options_handler():
    return {}

# Charger modèle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

labels_map = {
    1: "Qualité faible",
    3: "Bonne qualité",
    7: "Meilleure qualité"
}

class OliveFeatures(BaseModel):
    sterols: float
    triglycerides: float
    phenols: float
    acidite: float
    alcools_triterpeniques: float
    derives_tocopherol: float
    acides_gras: float
    densite_huile: float
    ph: float
    vitamine_e: float
    polyphenols: float


@app.get("/")
def root():
    return {"message": "API FastAPI fonctionne !"}


@app.post("/predict")
def predict(data: List[OliveFeatures]):
    results = []

    for d in data:
        features = np.array([[
            d.sterols,
            d.triglycerides,
            d.phenols,
            d.acidite,
            d.alcools_triterpeniques,
            d.derives_tocopherol,
            d.acides_gras,
            d.densite_huile,
            d.ph,
            d.vitamine_e,
            d.polyphenols
        ]])

        prediction = model.predict(features)[0]
        qualite_str = labels_map.get(int(prediction), "Qualité inconnue")

        results.append({
            "qualite_predite": qualite_str,
            "code": int(prediction)
        })

    return {"predictions": results}
