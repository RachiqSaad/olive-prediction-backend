from fastapi import FastAPI
from schemas import OliveFeatures
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Charger le modèle .pkl
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
# 🔹 Dictionnaire pour décoder la classe numérique du modèle
labels_map = {
    0: "Mauvaise qualité",
    1: "Bonne qualité",
    2: "Excellente qualité"
}
# 🔹 Endpoint /predict
@app.post("/predict")
def predict(data: OliveFeatures):
    # Encoder la variable catégorielle "type"
    type_map = {"Vert": 0, "Brun": -1}
    type_encoded = type_map.get(data.type, 1)

    # Créer le vecteur de features
    features = np.array([[
        type_encoded,
        data.sterols,
        data.triglycerides,
        data.phenols,
        data.acidite,
        data.alcools_triterpeniques,
        data.derives_tocopherol,
        data.acides_gras,
        data.densite_huile,
        data.ph,
        data.vitamine_e,
        data.polyphenols
    ]])

    # Prédiction
    prediction = model.predict(features)[0]

    # Conversion en texte
    qualite_str = labels_map.get(int(prediction), "Qualité inconnue")

    return {
        "qualite_predite": qualite_str,
        "code": int(prediction)
    }
