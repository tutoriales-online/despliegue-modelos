from datetime import datetime, timedelta

import requests
import streamlit as st


def calculate_weekdays_weekends(start_date, end_date):
    """
    Calculate the number of weekdays (Monday-Friday) and weekends (Saturday-Sunday)
    between two dates, inclusive.

    Args:
        start_date: datetime.date object for the start date
        end_date: datetime.date object for the end date

    Returns:
        tuple: (weekdays, weekends)
    """
    weekdays = 0
    weekends = 0

    current_date = start_date
    while current_date <= end_date:
        # Monday is 0, Sunday is 6 in weekday()
        if current_date.weekday() < 5:  # Monday (0) to Friday (4)
            weekdays += 1
        else:  # Saturday (5) and Sunday (6)
            weekends += 1
        current_date += timedelta(days=1)

    return weekdays, weekends


st.title("Sistema de reservas de hotel")

ENDPOINT = "http://localhost:3000"
with st.sidebar:
    st.title("Configuración")
    st.write("Configuración del sistema")

    st.subheader("Endpoint")
    ENDPOINT = st.text_input("Endpoint", ENDPOINT)


left, centre, right = st.columns(3)

with left:
    hotel = st.selectbox("Hotel", ["Resort Hotel", "City Hotel"])

with centre:
    starting_date = st.date_input("Fecha de inicio", value=datetime.now())

with right:
    ending_date = st.date_input("Fecha de fin", min_value=starting_date)


left, centre, right = st.columns(3)

with left:
    meal = st.selectbox(
        "Comida",
        [
            "BB",
            "HB",
            "FB",
        ],
    )

with centre:
    reserved_room_type = st.selectbox(
        "Tipo de habitación",
        [
            "A",
            "D",
            "E",
            "F",
            "G",
            "B",
            "C",
            "H",
            "P",
            "L",
        ],
    )

with right:
    customer_type = st.selectbox("Tipo de cliente", ["Transient", "Transient-Party", "Contract", "Group"])


left, centre, right = st.columns(3)

with left:
    required_car_parking_spaces = st.number_input(
        "Espacios de estacionamiento requeridos", min_value=0, max_value=10, value=0
    )

with centre:
    adr = st.slider("Tarifa diaria promedio", min_value=0, max_value=1000, value=0)

with right:
    container = st.container()

st.divider()


def predict_cancellation():
    url = f"{ENDPOINT}/predict_v2"
    weekdays, weekends = calculate_weekdays_weekends(starting_date, ending_date)
    response = requests.post(
        url,
        json={
            "data": [
                {
                    "reservation_id": 1,
                    "hotel": hotel,
                    "reserved_room_type": reserved_room_type,
                    "customer_type": customer_type,
                    "required_car_parking_spaces": required_car_parking_spaces,
                    "adr": adr,
                    "meal": meal,
                    "distribution_channel": "Direct",
                    "stays_in_week_nights": weekdays,
                    "stays_in_weekend_nights": weekends,
                }
            ]
        },
    )
    return response.json()


# clicked = st.button("Predecir cancelación", on_click=predict_cancellation)

prediction_result = predict_cancellation()

with container:
    prediction = prediction_result[0]["probability"]
    st.metric("Probabilidad de cancelación", f"{prediction:.1%}")
