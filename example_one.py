from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pytz

from helpers.helpers import get_week_range


# API_URL = "http://localhost:8000/api"
API_URL = "https://ice-productividad-production.up.railway.app/api"



current_staffs = (
    "ESTEBAN CALVO ELIZONDO",
    "GERARDO MORA GRANADOS",
    "CHRISTOPHER CEDEÑO ORTEGA",
    "JONATHAN ANDREY FERNANDEZ PICADO",
    "MARCELINO SANCHEZ ZUÑIGA",
    "LEYNER BARROZO VARGAS",
    "FABIAN HERNANDEZ ZAMORA",
    "RANDALL CAMACHO ROJAS",
    "PABLO MENDEZ MASIS",
)


# Function to get data from API
def get_updated_tasks():
    try:
        response = requests.get(f"{API_URL}/tasks/")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener datos de la API: {e}")
        return []


# Calcula los técnicos activos basado en tareas completadas o bien devueltas durante el período filtrado.
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


def crear_graficos_cumplimiento(df, fecha_inicio, fecha_fin):

    META_DIARIA_TECNICO = 250  # Meta diaria por técnico en $
    TOTAL_CUADRILLAS_ACTIVAS = len(calcular_tecnicos_activos_periodo(df))
    # Calcular días laborables en el período
    dias_laborables = calcular_dias_laborables(fecha_inicio, fecha_fin)
    META_TOTAL = TOTAL_CUADRILLAS_ACTIVAS * META_DIARIA_TECNICO * dias_laborables

    # Verificar si hay datos
    if df.empty:
        return (
            None,
            None,
            {
                "tecnicos_activos": 0,
                "meta_total": META_TOTAL,
                "total_alcanzado": 0,
                "porcentaje_cumplimiento": 0,
                "dias_laborables": dias_laborables,
                "tecnicos_inactivos": list(current_staffs),
            },
        )

    # Filtrar tareas completadas o bien devueltas
    df_valido = df[
        (df["status"] == "Completada")
        | ((df["status"] == "Devuelta") & (df["returned_well"] == 1))
    ]

    tecnicos_activos = df_valido["staff"].unique()

    # Calcular técnicos inactivos
    tecnicos_inactivos = [
        tech for tech in current_staffs if tech not in tecnicos_activos
    ]

    # Si no hay técnicos activos, retornar valores por defecto
    if len(tecnicos_activos) == 0:
        return (
            None,
            None,
            {
                "tecnicos_activos": 0,
                "meta_total": META_TOTAL,
                "total_alcanzado": 0,
                "porcentaje_cumplimiento": 0,
                "dias_laborables": dias_laborables,
                "tecnicos_inactivos": tecnicos_inactivos,
            },
        )

    # Calcular métricas por técnico
    metricas_tecnicos = []
    for tecnico in tecnicos_activos:
        tareas_tecnico = df_valido[df_valido["staff"] == tecnico]
        ingresos_tecnico = tareas_tecnico["total"].sum()
        metricas_tecnicos.append(
            {
                "tecnico": tecnico,
                "ingresos": ingresos_tecnico,
                "num_tareas": len(tareas_tecnico),
                "porcentaje_meta": (ingresos_tecnico / META_TOTAL * 100).round(1),
                "porcentaje_meta_individual": (
                    ingresos_tecnico / META_DIARIA_TECNICO * 100
                ).round(1),
            }
        )

    # Crear gráfico de pie de contribución por técnico
    datos_pie = []

    for metrica in metricas_tecnicos:
        datos_pie.append(
            f"{metrica['tecnico']}<br>"
            + f"${metrica['ingresos']:,.2f}<br>"
            # + f"{metrica['num_tareas']} tareas<br>"
            # + f"({metrica['porcentaje_meta']}% de meta)<br>"
            + f"({metrica['porcentaje_meta_individual']}% de meta)"
        )

    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=datos_pie,
                values=[m["ingresos"] for m in metricas_tecnicos],
                hole=0.3,
                textinfo="label",
                hovertemplate="<b>%{label}</b><extra></extra>",
            )
        ]
    )

    # datos_pie = []
    # for metrica in metricas_tecnicos:
    #     datos_pie.append(
    #         f"{metrica['tecnico']}<br>"
    #         + f"${metrica['ingresos']:,.2f}<br>"
    #         + f"{metrica['num_tareas']} tareas<br>"
    #         + f"({metrica['porcentaje_meta']}% de meta)<br>"
    #         + f"({metrica['porcentaje_meta_individual']}% de meta individual)"
    #     )

    # fig_pie = go.Figure(
    #     data=[
    #         go.Pie(
    #             labels=[
    #                 m["tecnico"] for m in metricas_tecnicos
    #             ],  # Solo el nombre del técnico
    #             values=[m["ingresos"] for m in metricas_tecnicos],
    #             hole=0.3,
    #             textinfo="label",
    #             hovertemplate=(
    #                 "<b>%{label}</b><br>"
    #                 "Ingresos: $%{value:,.2f}<br>"
    #                 "Tareas: %{customdata[0]}<br>"
    #                 "% de Meta: %{customdata[1]:.1f}%<br>"
    #                 "% de Meta Individual: %{customdata[2]:.1f}%"
    #                 "<extra></extra>"
    #             ),
    #             customdata=[
    #                 [
    #                     m["num_tareas"],
    #                     float(m["porcentaje_meta"]),
    #                     float(m["porcentaje_meta_individual"]),
    #                 ]
    #                 for m in metricas_tecnicos
    #             ],
    #         )
    #     ]
    # )

    fig_pie.update_layout(
        title={
            "text": f"Contribución por Técnico ({dias_laborables} días laborables)",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    # fig_pie.update_layout(
    #     title={
    #         "text": f"Contribución por Cuadrilla ({dias_laborables} días laborables)",
    #         "y": 0.95,
    #         "x": 0.5,
    #         "xanchor": "center",
    #         "yanchor": "top",
    #     },
    #     showlegend=True,
    #     legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    # )

    # TODO:: Crear gráfico de barras para comparación de meta
    total_alcanzado = sum(m["ingresos"] for m in metricas_tecnicos)
    porcentaje_cumplimiento = (total_alcanzado / META_TOTAL * 100).round(1)

    fig_barras = go.Figure()

    fig_barras.add_trace(
        go.Bar(
            x=["Meta"],
            y=[META_TOTAL],
            name="Meta del Período",
            marker_color="lightgray",
            text=[f"${META_TOTAL:,.2f}"],
            textposition="auto",
        )
    )

    fig_barras.add_trace(
        go.Bar(
            x=["Alcanzado"],
            y=[total_alcanzado],
            name=f"Alcanzado ({porcentaje_cumplimiento}%)",
            marker_color="rgb(0, 123, 255)",
            text=[f"${total_alcanzado:,.2f}"],
            textposition="auto",
        )
    )

    fig_barras.update_layout(
        title=f"Cumplimiento de Meta del Período ({dias_laborables} días laborables)",
        barmode="group",
        yaxis_title="Ingresos ($)",
        showlegend=True,
    )

    return (
        fig_pie,
        fig_barras,
        {
            "tecnicos_activos": len(tecnicos_activos),
            "meta_total": META_TOTAL,
            "total_alcanzado": total_alcanzado,
            "porcentaje_cumplimiento": porcentaje_cumplimiento,
            "dias_laborables": dias_laborables,
            "tecnicos_inactivos": tecnicos_inactivos,
        },
    )


# """
# 1. Genera las semanas del mes, considerando lunes a sábado.
# """
def get_semanas_del_mes(year, month):

    primer_dia = pd.Timestamp(year=year, month=month, day=1)

    # Si es el mes actual, usar el día actual como límite
    if month == pd.Timestamp.now().month:
        ultimo_dia = pd.Timestamp.now()
    else:
        ultimo_dia = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(
            days=1
        )

    semanas = []
    fecha_actual = primer_dia
    num_semana = 1

    while fecha_actual <= ultimo_dia:
        # Encontrar el primer día de la semana (lunes)
        inicio_semana = fecha_actual
        if fecha_actual.dayofweek > 0:  # Si no es lunes
            dias_al_lunes = fecha_actual.dayofweek
            inicio_semana = fecha_actual - pd.Timedelta(days=dias_al_lunes)

        # El fin de la semana es el sábado
        fin_semana = inicio_semana + pd.Timedelta(days=5)

        # Ajustar si el fin de semana excede el mes actual o el día actual
        if fin_semana > ultimo_dia:
            fin_semana = ultimo_dia

        # Ajustar si el inicio de semana es antes del mes actual
        if inicio_semana.month != month:
            inicio_semana = primer_dia

        semanas.append((num_semana, inicio_semana, fin_semana))

        fecha_actual = inicio_semana + pd.Timedelta(days=7)
        num_semana += 1

    return semanas


# """
# Calcula el número de días laborables (lunes a sábado) entre dos fechas
# """
def calcular_dias_laborables(fecha_inicio, fecha_fin):

    dias_laborables = 0
    fecha_actual = fecha_inicio

    while fecha_actual <= fecha_fin:
        if fecha_actual.dayofweek < 6:  # 0-5 son lunes a sábado
            dias_laborables += 1
        fecha_actual += pd.Timedelta(days=1)

    return dias_laborables


def crear_filtros_fecha():
    """
    Crea los filtros de fecha para el dashboard
    """
    st.write("Filtros de Fecha")

    # Obtener año y mes actual
    hoy = pd.Timestamp.now()

    # Crear lista de meses disponibles (desde enero hasta el mes actual)
    meses_disponibles = {
        i: mes
        for i, mes in enumerate(
            [
                "Enero",
                "Febrero",
                "Marzo",
                "Abril",
                "Mayo",
                "Junio",
                "Julio",
                "Agosto",
                "Septiembre",
                "Octubre",
                "Noviembre",
                "Diciembre",
            ],
            1,
        )
        if i <= hoy.month
    }

    col1, col2 = st.columns(2)

    with col1:
        mes_seleccionado = st.selectbox(
            "Seleccionar Mes:",
            options=list(meses_disponibles.keys()),
            format_func=lambda x: meses_disponibles[x],
            index=len(meses_disponibles) - 1,
        )

    # Generar lista de semanas para el mes seleccionado
    semanas = get_semanas_del_mes(hoy.year, mes_seleccionado)
    opciones_semanas = [
        f"Semana {sem[0]} ({sem[1].strftime('%d/%m')} - {sem[2].strftime('%d/%m')})"
        for sem in semanas
    ]
    opciones_semanas.insert(0, "Todas las semanas")

    with col2:
        semana_seleccionada = st.selectbox(
            "Seleccionar Semana:", options=opciones_semanas, index=0
        )

    # Crear lista de días disponibles
    primer_dia = pd.Timestamp(year=hoy.year, month=mes_seleccionado, day=1)
    if mes_seleccionado == hoy.month:
        ultimo_dia = hoy.day
    else:
        ultimo_dia = (
            pd.Timestamp(year=hoy.year, month=mes_seleccionado + 1, day=1)
            - pd.Timedelta(days=1)
        ).day

    dias_disponibles = list(range(1, ultimo_dia + 1))
    dias_opciones = ["Todos los días"] + [f"{dia:02d}" for dia in dias_disponibles]

    col1, col2 = st.columns(2)

    with col1:
        # Deshabilitar selección de día si hay una semana seleccionada
        dia_seleccionado = st.selectbox(
            "Seleccionar Día:",
            options=dias_opciones,
            index=0,
            disabled=semana_seleccionada != "Todas las semanas",
        )

    # with col2:
    #     # Input para cantidad de técnicos
    #     total_tecnicos = st.number_input(
    #         "Cantidad de cuadrillas:", min_value=1, max_value=50, value=9
    #     )  # Valor por defecto: 9

    # Crear las fechas de filtro
    year = hoy.year
    fecha_inicio = None
    fecha_fin = None

    if dia_seleccionado != "Todos los días":
        # Filtro por día específico
        # dia = int(dia_seleccionado)
        # fecha_inicio = pd.Timestamp(year=year, month=mes_seleccionado, day=dia)
        # fecha_fin = fecha_inicio
        dia = int(dia_seleccionado)
        fecha_inicio = pd.Timestamp(
            year=year, month=mes_seleccionado, day=dia, hour=0, minute=0, second=0
        )
        fecha_fin = pd.Timestamp(
            year=year, month=mes_seleccionado, day=dia, hour=23, minute=59, second=59
        )
    elif semana_seleccionada != "Todas las semanas":
        # Filtro por semana
        semana_idx = opciones_semanas.index(semana_seleccionada) - 1
        fecha_inicio = semanas[semana_idx][1]
        fecha_fin = semanas[semana_idx][2]
    else:
        # Filtro por mes completo
        fecha_inicio = pd.Timestamp(year=year, month=mes_seleccionado, day=1)
        if mes_seleccionado == hoy.month:
            fecha_fin = hoy
        else:
            fecha_fin = pd.Timestamp(
                year=year, month=mes_seleccionado + 1, day=1
            ) - pd.Timedelta(days=1)

    return fecha_inicio, fecha_fin


def filtrar_df_por_fecha(df, fecha_inicio, fecha_fin):
    """
    Filtra el DataFrame por rango de fechas, considerando el día completo.

    Args:
        df: DataFrame con los datos
        fecha_inicio: Timestamp con la fecha inicial
        fecha_fin: Timestamp con la fecha final
    """
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


# TODO:: Dashboard de Productividad Técnicos
def main():
    # st.title("Dashboard de Productividad Técnicos")
    tasks = get_updated_tasks()
    if not tasks:
        st.warning("No se pudieron obtener datos de la API")
        return

    df = pd.DataFrame(tasks)

    # df["fecha"] = pd.to_datetime(df["datedelivery_time"])
    df["fecha"] = pd.NaT

    # Asignar "completed_time" si el estado es "Completada"
    df.loc[df["status"] == "Completada", "fecha"] = pd.to_datetime(
        df.loc[df["status"] == "Completada", "completed_time"]
    )

    # Asignar "returnedwell_time" si el estado es "Devuelta" y returned_well es 1
    df.loc[(df["status"] == "Devuelta") & (df["returned_well"] == 1), "fecha"] = (
        pd.to_datetime(
            df.loc[
                (df["status"] == "Devuelta") & (df["returned_well"] == 1),
                "returnedwell_time",
            ]
        )
    )

    df["returned_well"] = pd.to_numeric(df["returned_well"], errors="coerce").fillna(0)

    years = sorted(df["fecha"].dt.year.unique().tolist(), reverse=True)
    selected_year = st.selectbox("Seleccionar Año:", years)

    df = df[df["fecha"].dt.year == selected_year]

    # Filtros en la parte superior
    col1, col2, col3 = st.columns(3)
    with col1:
        sites = ["Todos"] + sorted(df["site"].unique().tolist())
        selected_site = st.selectbox("Filtrar por Sitio:", sites)

    with col2:
        events = ["Todos"] + sorted(df["event"].unique().tolist())
        selected_event = st.selectbox("Filtrar por Evento:", events)

    with col3:
        estados = ["Todos", "Completadas", "Bien Devueltas"]
        selected_estado = st.selectbox("Filtrar por Estado:", estados)

    # Aplicar filtros
    filtered_df = df.copy()
    if selected_site != "Todos":
        filtered_df = filtered_df[filtered_df["site"] == selected_site]
    if selected_event != "Todos":
        filtered_df = filtered_df[filtered_df["event"] == selected_event]
    if selected_estado != "Todos":
        if selected_estado == "Completadas":
            filtered_df = filtered_df[filtered_df["status"] == "Completada"]
        elif selected_estado == "Bien Devueltas":
            filtered_df = filtered_df[
                (filtered_df["status"] == "Devuelta")
                & (filtered_df["returned_well"] == 1)
            ]

    # DataFrame para cálculos (solo Completadas y Bien Devueltas)
    calculo_df = filtered_df[
        (filtered_df["status"] == "Completada")
        | ((filtered_df["status"] == "Devuelta") & (filtered_df["returned_well"] == 1))
    ]

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_pedidos = len(calculo_df)
        num_tecnicos = calculo_df["staff"].nunique()
        productividad = total_pedidos / num_tecnicos if num_tecnicos > 0 else 0
        st.metric("Productividad", f"{productividad:.1f}")

    with col2:
        ingresos = calculo_df["total"].sum()
        st.metric("Ingresos Totales", f"${ingresos:,.2f}")

    with col3:
        st.metric("Cuadrillas", num_tecnicos)

    with col4:
        ingreso_por_tecnico = ingresos / num_tecnicos if num_tecnicos > 0 else 0
        st.metric("Ingreso por Cuadrilla", f"${ingreso_por_tecnico:,.2f}")

    # Filtros de tiempo
    st.write("---")
    tiempo_options = ["Diario", "Última Semana", "Mes Actual", "Mes Anterior"]
    tiempo_filter = st.radio("Filtrar por período:", tiempo_options, horizontal=True)

    # Aplicar filtro de tiempo
    now = datetime.now()
    if tiempo_filter == "Diario":
        calculo_df = calculo_df[calculo_df["fecha"].dt.date == now.date()]
    elif tiempo_filter == "Última Semana":
        week_ago = now - timedelta(days=7)
        calculo_df = calculo_df[calculo_df["fecha"] >= week_ago]
        start_date, end_date = get_week_range(week_ago)
        st.info(f"Período: {start_date} al {end_date}")
    elif tiempo_filter == "Mes Actual":
        calculo_df = calculo_df[calculo_df["fecha"].dt.month == now.month]
    elif tiempo_filter == "Mes Anterior":
        calculo_df = calculo_df[
            calculo_df["fecha"].dt.month == (now - timedelta(days=30)).month
        ]

    # Gráfico de ingresos por Cuadrilla
    if len(calculo_df) > 0:
        ingresos_por_tecnico = calculo_df.groupby("staff")["total"].sum().reset_index()
        ingresos_por_tecnico.columns = ["Cuadrilla", "Ingresos"]

        fig = go.Figure(
            go.Bar(
                x=ingresos_por_tecnico["Ingresos"],
                y=ingresos_por_tecnico["Cuadrilla"],
                orientation="h",
                text=[f"${x:,.2f}" for x in ingresos_por_tecnico["Ingresos"]],
                textposition="auto",
            )
        )

        fig.update_layout(
            title="Ingresos por Cuadrilla",
            xaxis_title="Ingresos ($)",
            yaxis_title="Cuadrilla",
            height=400 + (len(ingresos_por_tecnico) * 30),
            margin=dict(l=200),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos para mostrar con los filtros seleccionados")

    # TODO:: -------------------------------------------- GRÁFICO DE CUMPLIMIENTO DE METAS ------------------------------
    st.write("---")
    st.subheader("Cumplimiento de Metas")

    # Agregar filtros de fecha
    fecha_inicio, fecha_fin = crear_filtros_fecha()
    # Filtrar DataFrame
    df_filtrado = filtrar_df_por_fecha(filtered_df, fecha_inicio, fecha_fin)

    # Mostrar período seleccionado
    if fecha_inicio and fecha_fin:
        st.info(
            f"Mostrando datos del período: {fecha_inicio.strftime('%d/%m/%Y')} al {fecha_fin.strftime('%d/%m/%Y')}"
        )

    # Crear los gráficos con datos filtrados
    fig_pie, fig_barras, metricas = crear_graficos_cumplimiento(
        df_filtrado, fecha_inicio, fecha_fin
    )

    # Mostrar métricas
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Meta del Período", f"${metricas['meta_total']:,.2f}")
    with col2:
        st.metric("Total Alcanzado", f"${metricas['total_alcanzado']:,.2f}")
    with col3:
        st.metric("% Cumplimiento", f"{metricas['porcentaje_cumplimiento']}%")
    with col4:
        st.metric("Cuadrillas Activas", metricas["tecnicos_activos"])
    with col5:
        st.metric("Días Laborables", metricas["dias_laborables"])

    # Mostrar gráficos solo si hay datos
    if fig_pie and fig_barras:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_barras, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)
            if metricas["tecnicos_inactivos"]:
                st.write("##### Técnicos sin actividad en el período:")
                # if metricas["tecnicos_inactivos"]:
                # st.write("##### Técnicos sin actividad en el período")
                # Crear DataFrame con una sola columna
                df_inactivos = pd.DataFrame(
                    metricas["tecnicos_inactivos"], columns=["Nombre"]
                )
                # Mostrar la tabla sin índice y con estilo personalizado
                st.dataframe(
                    df_inactivos,
                    hide_index=True,
                    use_container_width=True,
                    height=(
                        len(metricas["tecnicos_inactivos"]) * 35 + 38
                    ),  # Ajuste de altura dinámico
                )
    else:
        st.warning("No se encontraron datos para el período seleccionado.")


main()
