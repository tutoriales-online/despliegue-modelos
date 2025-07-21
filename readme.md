# Prevención de Cancelaciones de Reservas de Hoteles

## Descripción del Proyecto

Este proyecto implementa un sistema de machine learning para predecir cancelaciones de reservas de hoteles. El objetivo es ayudar a los hoteles a identificar reservas con alta probabilidad de cancelación, permitiendo una mejor gestión de inventario y optimización de ingresos.

## Estructura del Proyecto

```
production-ml/
├── data/                    # Datos de entrenamiento y prueba
│   ├── original.csv         # Dataset original con reservas
│   └── test_data.csv        # Datos de prueba
├── models/                  # Modelos entrenados guardados
├── training/                # Código de entrenamiento
│   ├── main.py             # Script principal de entrenamiento
│   ├── pipeline.py         # Definición del pipeline de ML
│   └── __init__.py
├── requirements.txt         # Dependencias del proyecto
└── readme.md               # Este archivo
```

## Uso del Proyecto

### 1. Entrenar el Modelo

Para entrenar un nuevo modelo con los datos disponibles:

```bash
python training/main.py
```

### 2. Identificar los modelos en BentoML

```bash
bentoml models list
```

### 3. Servir el Modelo

```bash
bentoml serve
```

### 4. Construir un Bento

```bash
bentoml build
```

### 5. Construir una imagen Docker

```bash
bentoml containerize hotel_cancellations_service:[TAG]
```


## Apéndice

Para colocar `breakpoint` en el servicio y ver qué sucede:

```bash
bentoml serve --development --working-dir deployment
```
