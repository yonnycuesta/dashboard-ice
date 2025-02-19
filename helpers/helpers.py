from datetime import timedelta

import pandas as pd
import pytz


# TODO:: Establecer un rando de fechas
def get_week_range(date):
    start_of_week = date - timedelta(days=date.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week.strftime("%d %B"), end_of_week.strftime("%d %B")


"""
    Filtra el DataFrame por rango de fechas, considerando el día completo.
    Args:
        df: DataFrame con los datos
        fecha_inicio: Timestamp con la fecha inicial
        fecha_fin: Timestamp con la fecha final
    """
def filtrar_df_por_fecha(df, fecha_inicio, fecha_fin):

    if fecha_inicio is None or fecha_fin is None:
        return df

    # Asegurarse de que las fechas del DataFrame tengan zona horaria
    if df["fecha"].dt.tz is None:
        df["fecha"] = df["fecha"].dt.tz_localize(pytz.UTC)

    # Agregar zona horaria a las fechas de filtro si no la tienen
    if fecha_inicio.tz is None:
        fecha_inicio = fecha_inicio.tz_localize(pytz.UTC)
    if fecha_fin.tz is None:
        fecha_fin = fecha_fin.tz_localize(pytz.UTC)

    # Si fecha_inicio y fecha_fin son el mismo día, ajustar fecha_fin al final del día
    if fecha_inicio.date() == fecha_fin.date():
        fecha_fin = fecha_fin.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Filtrar el DataFrame
    df_filtrado = df[
        (df["fecha"] >= fecha_inicio)
        & (
            df["fecha"] <= fecha_fin
        )  # Cambiado a <= para incluir el último segundo del día
    ]

    return df_filtrado


#TODO:: Calcula los técnicos activos basado en tareas completadas o bien devueltas durante el período filtrado.
def calcular_tecnicos_activos_periodo(df):
    # Convertir a datetime si no lo es
    if not pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        df["fecha"] = pd.to_datetime(df["fecha"])

    # Asegurarse de que las fechas en el DataFrame tengan zona horaria
    if df["fecha"].dt.tz is None:
        df["fecha"] = df["fecha"].dt.tz_localize(pytz.UTC)

    # Filtrar técnicos que han tenido actividad en el período
    tecnicos_activos = df[
        (df["status"] == "Completada")
        | ((df["status"] == "Devuelta") & (df["returned_well"] == 1))
    ]["staff"].unique()

    return tecnicos_activos