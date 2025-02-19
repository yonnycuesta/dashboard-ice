from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


from helpers.helpers import (
    filtrar_df_por_fecha,
    calcular_tecnicos_activos_periodo,
)


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


def calcular_dias_laborables_mes(year, month):
    """
    Calcula los días laborables (lunes a sábado) para un mes específico
    """
    primer_dia = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        siguiente_mes = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        siguiente_mes = pd.Timestamp(year=year, month=month + 1, day=1)
    ultimo_dia = siguiente_mes - pd.Timedelta(days=1)

    dias_laborables = 0
    fecha_actual = primer_dia

    while fecha_actual <= ultimo_dia:
        if fecha_actual.dayofweek < 6:  # 0-5 son lunes a sábado
            dias_laborables += 1
        fecha_actual += pd.Timedelta(days=1)

    return dias_laborables


def crear_graficos_cumplimiento(df, fecha_inicio, fecha_fin):

    META_DIARIA_TECNICO = 250  # Meta diaria por técnico en $
    TOTAL_CUADRILLAS_ACTIVAS = len(calcular_tecnicos_activos_periodo(df))
    # Calcular días laborables en el período
    dias_laborables = calcular_dias_laborables(fecha_inicio, fecha_fin)
    META_TOTAL = TOTAL_CUADRILLAS_ACTIVAS * META_DIARIA_TECNICO * dias_laborables

    # Calcular días laborables totales del mes (lunes a sábado)
    dias_laborables_mes = calcular_dias_laborables_mes(
        fecha_inicio.year, fecha_inicio.month
    )
    META_TOTAL_MES = (
        TOTAL_CUADRILLAS_ACTIVAS * META_DIARIA_TECNICO * dias_laborables_mes
    )
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

    # Crear gráfico de proyección
    df_valido["fecha"] = pd.to_datetime(df_valido["fecha"])
    df_diario = (
        df_valido.groupby(df_valido["fecha"].dt.date)["total"].sum().reset_index()
    )
    df_diario["acumulado"] = df_diario["total"].cumsum()

    # Calcular días transcurridos y total actual
    dias_transcurridos = len(df_diario)
    total_actual = df_diario["acumulado"].iloc[-1] if not df_diario.empty else 0

    # Crear gráfico de proyección
    fig_projection = go.Figure()

    # Línea de meta mensual con etiqueta
    x_mes = [fecha_inicio, fecha_fin]
    y_mes = [0, META_TOTAL_MES]
    fig_projection.add_trace(
        go.Scatter(
            x=x_mes,
            y=y_mes,
            name=f"Meta Mensual ({dias_laborables_mes} días)",
            line=dict(color="red", dash="dash"),
            text=["", f"${META_TOTAL_MES:,.0f}"],
            mode="lines+text",
            textposition="top right",
            textfont=dict(color="red"),
        )
    )

    # Línea de meta del período seleccionado
    # Línea de meta del período con etiqueta
    fig_projection.add_trace(
        go.Scatter(
            x=x_mes,
            y=[0, META_TOTAL],
            name=f"Meta Período ({dias_laborables} días)",
            line=dict(color="gray", dash="dash"),
            text=["", f"${META_TOTAL:,.0f}"],
            mode="lines+text",
            textposition="bottom right",
            textfont=dict(color="gray"),
        )
    )

    # Línea de progreso actual
    if not df_diario.empty:
        # Crear lista de textos vacíos excepto el último valor
        text_labels = [""] * (len(df_diario) - 1) + [f"${total_actual:,.0f}"]

        fig_projection.add_trace(
            go.Scatter(
                x=df_diario["fecha"],
                y=df_diario["acumulado"],
                name=f"Progreso Actual ({dias_transcurridos} días)",
                line=dict(color="blue"),
                text=text_labels,
                mode="lines+text",
                textposition="middle right",
                textfont=dict(color="blue"),
            )
        )

    fig_projection.update_layout(
        title=f"Progreso vs. Metas ({dias_laborables_mes} días laborables en el mes)",
        xaxis_title="Fecha",
        yaxis_title="Monto ($)",
        showlegend=True,
    )

    # TODO:: Calcular métricas de proyección mensual
    # Calcular métricas de proyección mensual
    promedio_diario_actual = (
        total_actual / dias_transcurridos if dias_transcurridos > 0 else 0
    )
    dias_restantes_mes = dias_laborables_mes - dias_transcurridos
    proyeccion_final = total_actual + (promedio_diario_actual * dias_restantes_mes)

    mensaje_proyeccion = f"""
📅 Análisis de Proyección Mensual ({dias_laborables_mes} días laborables)

Progreso:
• Monto actual: ${total_actual:,.2f}
• Porcentaje completado: {(total_actual/META_TOTAL_MES*100):.1f}% de la meta mensual
• Meta mensual total: ${META_TOTAL_MES:,.2f}
• Días laborados: {dias_transcurridos} de {dias_laborables_mes}
• Promedio diario actual: ${promedio_diario_actual:,.2f}

🎯 Proyección:
• Al ritmo actual se alcanzaría: ${proyeccion_final:,.2f} al final del mes
• Meta diaria necesaria para objetivo mensual: ${(META_TOTAL_MES - total_actual) / dias_restantes_mes if dias_restantes_mes > 0 else 0:,.2f}
"""

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
        # Calcular días trabajados (días únicos con tareas completadas o bien devueltas)
        dias_trabajados = tareas_tecnico["fecha"].dt.date.nunique()
        meta_individual = META_DIARIA_TECNICO * dias_trabajados

        metricas_tecnicos.append(
            {
                "tecnico": tecnico,
                "ingresos": ingresos_tecnico,
                "num_tareas": len(tareas_tecnico),
                "dias_trabajados": dias_trabajados,
                "meta_individual": meta_individual,
                "porcentaje_meta": (ingresos_tecnico / META_TOTAL * 100).round(1),
                "porcentaje_meta_individual": (
                    (ingresos_tecnico / meta_individual * 100).round(1)
                    if meta_individual > 0
                    else 0
                ),
            }
        )

    # Crear gráfico de pie de contribución por técnico
    datos_pie = []

    for metrica in metricas_tecnicos:
        datos_pie.append(
            f"{metrica['tecnico']}<br>"
            + f"Alcanzado: ${metrica['ingresos']:,.2f}<br>"
            + f"Tareas: {metrica['num_tareas']}<br>"
            + f"Días laborados: {metrica['dias_trabajados']}<br>"
            + f"Meta: ${metrica['meta_individual']:,.2f}<br>"
            + f"Porcentaje: {metrica['porcentaje_meta_individual']}%"
        )
    # TODO:: Crear gráfico de barras para comparación de meta
    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=datos_pie,
                values=[m["ingresos"] for m in metricas_tecnicos],
                hole=0.3,
                textposition="inside",
                textinfo="none",
                hovertemplate="<b>%{label}</b><extra></extra>",
            )
        ]
    )

    fig_pie.update_layout(
        title={
            "text": f"Contribución por Cuadrilla ({dias_laborables} días laborables)",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.9, xanchor="center", x=0.5),
    )
    # Crear el gráfico de torta

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
        fig_projection,
        {
            "tecnicos_activos": len(tecnicos_activos),
            "meta_total": META_TOTAL,
            "total_alcanzado": total_alcanzado,
            "porcentaje_cumplimiento": porcentaje_cumplimiento,
            "dias_laborables": dias_laborables,
            "tecnicos_inactivos": tecnicos_inactivos,
            "mensaje_proyeccion": mensaje_proyeccion,
            "meta_total_mes": META_TOTAL_MES,
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


# TODO:: Crea los filtros de fecha para el dashboard
def crear_filtros_fecha():

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
    # primer_dia = pd.Timestamp(year=hoy.year, month=mes_seleccionado, day=1)
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

    # Crear las fechas de filtro
    year = hoy.year
    fecha_inicio = None
    fecha_fin = None

    if dia_seleccionado != "Todos los días":
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


# TODO:: Dashboard de Productividad Técnicos
def main():
    # st.title("Dashboard de Productividad Técnicos")
    tasks = get_updated_tasks()
    if not tasks:
        st.warning("No se pudieron obtener datos de la API")
        return

    df = pd.DataFrame(tasks)

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

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        years = sorted(df["fecha"].dt.year.unique().tolist(), reverse=True)
        selected_year = st.selectbox("Seleccionar Año:", years)

        df = df[df["fecha"].dt.year == selected_year]

    with col6:
        sites = ["Todos"] + sorted(df["site"].unique().tolist())
        selected_site = st.selectbox("Filtrar por Sitio:", sites)

    with col7:
        events = ["Todos"] + sorted(df["event"].unique().tolist())
        selected_event = st.selectbox("Filtrar por Evento:", events)

    with col8:
        estados = ["Todos", "Completadas", "Bien Devueltas"]
        selected_estado = st.selectbox("Filtrar por Estado:", estados)

    # TODO:: -------------------------------------------- GRÁFICO DE CUMPLIMIENTO DE METAS ------------------------------
    st.write("---")
    st.subheader("Cumplimiento de Metas")

    # Agregar filtros de fecha
    fecha_inicio, fecha_fin = crear_filtros_fecha()
    filtered_df = df.copy()

    # Filtrar DataFrame
    df_filtrado = filtrar_df_por_fecha(filtered_df, fecha_inicio, fecha_fin)

    # Mostrar período seleccionado
    if fecha_inicio and fecha_fin:
        st.info(
            f"Mostrando datos del período: {fecha_inicio.strftime('%d/%m/%Y')} al {fecha_fin.strftime('%d/%m/%Y')}"
        )

    # Crear los gráficos con datos filtrados
    fig_pie, fig_barras, fig_projection, metricas = crear_graficos_cumplimiento(
        df_filtrado, fecha_inicio, fecha_fin
    )
    # Mostrar mensaje de proyección
    st.write("### Análisis de Proyección")
    st.write(metricas["mensaje_proyeccion"])

    # Mostrar gráfico de proyección
    st.plotly_chart(fig_projection, use_container_width=True)

    # Mostrar métricas
    col1, col2, col3, col4, col5 = st.columns(5)

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
                df_inactivos = pd.DataFrame(
                    metricas["tecnicos_inactivos"], columns=["Nombre"]
                )
                # Mostrar la tabla sin índice y con estilo personalizado
                st.dataframe(
                    df_inactivos,
                    hide_index=True,
                    use_container_width=True,
                    height=(len(metricas["tecnicos_inactivos"]) * 35 + 38),
                )
    else:
        st.warning("No se encontraron datos para el período seleccionado.")


main()
