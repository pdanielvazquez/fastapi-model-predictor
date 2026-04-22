from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

class Wine(BaseModel):
    """Modelo de datos para vino (calidad basada en propiedades físico-químicas)"""
    
    fixed_acidity: float = Field(alias="fixed acidity")
    volatile_acidity: float = Field(alias="volatile acidity")
    citric_acid: float = Field(alias="citric acid")
    residual_sugar: float = Field(alias="residual sugar")
    chlorides: float = Field(alias="chlorides")
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide")
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict/")
def predict(item: Wine):
    # Carga el modelo entrenado
    model = joblib.load('random_forest_wine_quality_model.joblib')

    try:
        # USAR los datos del item recibido
        X_new = pd.DataFrame([[
            item.fixed_acidity,
            item.volatile_acidity,
            item.citric_acid,
            item.residual_sugar,
            item.chlorides,
            item.free_sulfur_dioxide,
            item.total_sulfur_dioxide,
            item.density,
            item.pH,
            item.sulphates,
            item.alcohol
        ]], columns=[
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ])
        
        prediction = model.predict(X_new)
        
        # CONVERTIR a tipo nativo Python
        prediction_value = int(prediction[0])  # numpy.int64 -> int
        # O también: prediction_value = prediction[0].item()
        
        return {"prediction": prediction_value}
        
    except Exception as e:
        # Mejor usar HTTPException para errores
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
