from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Binarizer, OneHotEncoder, RobustScaler


def build_pipeline() -> Pipeline:
    """
    Construye un pipeline completo de machine learning para predecir cancelaciones de reservas de hoteles.

    Este pipeline incluye:
    1. Codificación one-hot para variables categóricas
    2. Binarización de variables numéricas específicas
    3. Escalado robusto para variables numéricas continuas
    4. Passthrough para variables que no necesitan transformación
    5. Modelo Random Forest para clasificación

    Returns:
        Pipeline: Pipeline completo listo para entrenamiento y predicción

    Notas sobre el procesamiento:
        - One-hot encoding: Convierte variables categóricas en variables binarias
        - Binarización: Convierte variables numéricas en binarias (0/1)
        - RobustScaler: Escala variables numéricas de forma robusta a outliers
        - Passthrough: Mantiene variables sin transformación
    """

    # Codificador one-hot para variables categóricas
    internal_one_hot_encoding = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    columns_to_encode = [
        "hotel",  # Nombre del hotel
        "meal",  # Tipo de comida incluida
        "distribution_channel",  # Canal de distribución
        "reserved_room_type",  # Tipo de habitación reservada
        "assigned_room_type",  # Tipo de habitación asignada
        "customer_type",  # Tipo de cliente
    ]

    one_hot_encoding = ColumnTransformer([("one_hot_encode", internal_one_hot_encoding, columns_to_encode)])

    # Binarizador para variables numéricas específicas
    internal_binarizer = Binarizer()
    columns_to_binarize = [
        "total_of_special_requests",  # Total de solicitudes especiales
        "required_car_parking_spaces",  # Espacios de estacionamiento requeridos
        "booking_changes",  # Cambios en la reserva
        "previous_bookings_not_canceled",  # Reservas previas no canceladas
        "previous_cancellations",  # Cancelaciones previas
    ]
    internal_encoder_binarizer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    binarizer = ColumnTransformer([("binarizer", internal_binarizer, columns_to_binarize)])

    # Pipeline combinado para binarización y codificación one-hot
    one_hot_binarized = Pipeline(
        [
            ("binarizer", binarizer),  # Primero binarizar
            ("one_hot_encoder", internal_encoder_binarizer),  # Luego codificar one-hot
        ]
    )

    # Escalador robusto para variables numéricas continuas
    internal_scaler = RobustScaler()
    columns_to_scale = ["adr"]  # Average Daily Rate (tarifa diaria promedio)

    scaler = ColumnTransformer([("scaler", internal_scaler, columns_to_scale)])

    # Columnas que pasan sin transformación
    pass_columns = [
        "stays_in_week_nights",  # Noches de estadía entre semana
        "stays_in_weekend_nights",  # Noches de estadía en fin de semana
    ]

    passthrough = ColumnTransformer([("pass_columns", "passthrough", pass_columns)])

    # Pipeline completo de ingeniería de características
    feature_engineering_pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("categories", one_hot_encoding),  # Variables categóricas codificadas
                        ("binaries", one_hot_binarized),  # Variables binarizadas
                        ("scaled", scaler),  # Variables escaladas
                        ("passthrough", passthrough),  # Variables sin transformación
                    ]
                ),
            )
        ]
    )

    # Modelo de machine learning - Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100)

    # Pipeline final completo
    final_pipeline = Pipeline([("feature_engineering", feature_engineering_pipeline), ("model", model)])

    return final_pipeline
