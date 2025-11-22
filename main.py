from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Autoriser CORS pour le front
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Dictionnaire pour décoder la classe prédite
labels_map = {
    0: "Mauvaise qualité",
    1: "Bonne qualité",
    2: "Excellente qualité"
}

# Schema Pydantic
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

# Route test
@app.get("/")
def root():
    return {"message": "API FastAPI fonctionne !"}

# Endpoint pour prédiction multiple
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
