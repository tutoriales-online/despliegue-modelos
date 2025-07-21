import logging
from uuid import uuid4

import bentoml
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


python_image = bentoml.images.Image(python_version="3.11", lock_python_packages=False)


@bentoml.service(image=python_image)
class HotelCancellationsService:

    bento_model = bentoml.models.BentoModel("hotel_cancellations_model")

    def __init__(self):
        self.model = bentoml.sklearn.load_model(self.bento_model)
        logger.info(f"Loaded model {self.bento_model.tag}")

    @bentoml.api
    def predict_v1(self, data: pd.DataFrame) -> np.ndarray:
        logger.info(f"Predicting {data.shape[0]} rows")
        predictions = self.model.predict(data)
        return predictions

    @bentoml.api
    def predict_v2(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Predicting {data.shape[0]} rows")
        prediction_ids = [uuid4() for _ in range(data.shape[0])]
        reservation_ids = data["reservation_id"].values

        probabilities = self.model.predict_proba(data)
        predictions = self.model.predict(data)

        return pd.DataFrame(
            {
                "prediction_id": prediction_ids,
                "reservation_id": reservation_ids,
                "prediction": predictions,
                "probability": probabilities[:, 1],
            }
        )
