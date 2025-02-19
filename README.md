fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=nombres_tecnicos,  # Usar solo los nombres para las etiquetas en el gráfico
                values=valores,
                hole=0.3,
                textposition="inside",
                textinfo="label+percent",  # Mostrar nombre y porcentaje dentro del gráfico
                hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<extra></extra>",
                customdata=leyenda_detallada,  # Usar para la leyenda detallada
            )
        ]
    )