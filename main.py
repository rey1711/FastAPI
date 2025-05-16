from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Load model dan scaler
model = joblib.load('model_regresi_rf_tuned.pkl')
scaler = joblib.load('scaler_regresi.pkl')

# Inisialisasi FastAPI
app = FastAPI(title="API Prediksi Penyusutan Aset Masa Depan")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Nama fitur numerik untuk scaler (sesuai nama SAAT training)
numerical_features = [
    'MASA PEROLEHAN', 'NILAI PEROLEHAN', 'Selisih_Semester', 'Umur_Aset'
]

# Model input (sesuai format API, bukan model)
class AsetInput(BaseModel):
    MASA_PEROLEHAN: float
    NILAI_PEROLEHAN: float
    Aset_Baru: int
    Perolehan_Mahal: int
    Akum_Jan_Tinggi: int
    Akum_Des_Tinggi: int
    S1_DiAtasRata: int
    S2_DiAtasRata: int
    Selisih_Semester: float
    Umur_Aset: float

# Endpoint prediksi
@app.post("/predict")
def predict_penyusutan(data: AsetInput):
    try:
        # Convert ke dataframe
        input_data = pd.DataFrame([data.dict()])

        # Rename agar cocok dengan model yang dilatih
        input_data.rename(columns={
            'MASA_PEROLEHAN': 'MASA PEROLEHAN',
            'NILAI_PEROLEHAN': 'NILAI PEROLEHAN'
        }, inplace=True)

        # Scaling
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Prediksi
        hasil = model.predict(input_data)[0]
        return {
            "Prediksi_Penyusutan_Semester_1": round(hasil[0], 2),
            "Prediksi_Penyusutan_Semester_2": round(hasil[1], 2)
        }

    except Exception as e:
        return {"error": str(e)}
