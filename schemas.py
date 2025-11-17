from pydantic import BaseModel

# 🔹 Schéma JSON d’entrée
class OliveFeatures(BaseModel):
    type: str
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


