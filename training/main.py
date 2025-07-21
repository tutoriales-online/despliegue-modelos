# Este es un script para entrenar un modelo de machine learning con datos de cancelaciones de hoteles.
# El objetivo del modelo es predecir si una reserva de hotel será cancelada.

from pathlib import Path

import bentoml
import joblib
import pandas as pd
from pipeline import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Configuración de rutas para los modelos
MODELS_PATH = "models"
MODEL_NAME = "hotel_cancellations_model"


def load_data(data_path: str) -> pd.DataFrame:
    """
    Carga y prepara los datos de reservas de hoteles.

    Args:
        data_path (str): Ruta al archivo CSV con los datos de reservas

    Returns:
        pd.DataFrame: DataFrame limpio y preparado para el entrenamiento

    Notas:
        - Elimina información personal de los clientes (nombre, email, teléfono, tarjeta de crédito)
        - Evita data leakage eliminando columnas relacionadas con el estado de la reserva
        - Convierte columnas de tipo objeto a string para compatibilidad
    """

    # Leer los datos
    hotel_bookings = pd.read_csv(data_path)

    # Eliminar información personal de los clientes
    hotel_bookings = hotel_bookings.drop(["name", "email", "phone-number", "credit_card"], axis=1)

    # Evitar data leakage - no usar información que no estaría disponible al momento de la reserva
    hotel_bookings = hotel_bookings.drop(["reservation_status", "reservation_status_date"], axis=1)

    # Convertir objetos a strings para compatibilidad
    object_columns = hotel_bookings.select_dtypes("object").columns
    hotel_bookings[object_columns] = hotel_bookings[object_columns].astype(str)

    return hotel_bookings


def split_data(data: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Args:
        data (pd.DataFrame): DataFrame completo con features y target
        test_size (float): Proporción de datos para el conjunto de prueba (por defecto 0.2)

    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Features y targets divididos
    """

    # Separar features (características) del target (variable objetivo)
    features = data.drop("is_canceled", axis=1)  # Features son todas las columnas excepto la columna objetivo
    target = data["is_canceled"]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Entrena el modelo de machine learning usando el pipeline definido.

    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento

    Returns:
        Pipeline: Modelo entrenado listo para hacer predicciones
    """

    # Construir el pipeline de procesamiento y modelo
    pipeline = build_pipeline()

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    return pipeline


def save_model(model: Pipeline, models_path: str, model_name: str) -> Path:
    """
    Guarda el modelo entrenado en el sistema de archivos.

    Args:
        model (Pipeline): Modelo entrenado a guardar
        models_path (str): Directorio donde guardar el modelo
        model_name (str): Nombre del archivo del modelo

    Returns:
        Path: Ruta completa donde se guardó el modelo
    """

    model_path = Path(models_path)
    model_path.mkdir(parents=True, exist_ok=True)

    final_model_path = model_path / model_name

    # Guardar el modelo
    with open(final_model_path, "wb") as f:
        joblib.dump(model, f)

    return final_model_path

def main():
    """
    Función principal que ejecuta todo el flujo de entrenamiento del modelo.

    Pasos:
    1. Cargar y preparar los datos
    2. Dividir en conjuntos de entrenamiento y prueba
    3. Entrenar el modelo
    4. Guardar el modelo entrenado

    Returns:
        Path: Ruta donde se guardó el modelo
    """

    # Cargar los datos
    print("Cargando datos...")
    data = load_data("data/original.csv")

    # Dividir los datos
    print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = split_data(data)

    # Entrenar el modelo
    print("Entrenando modelo de machine learning...")
    model = train_model(X_train, y_train)

    # Guardar el modelo
    print("Guardando modelo entrenado...")
    model_path = save_model(model, MODELS_PATH, MODEL_NAME)

    return model_path


if __name__ == "__main__":
    main()
