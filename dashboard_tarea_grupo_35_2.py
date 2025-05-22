import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

# Definimos la paleta global de tonos azules
TONOS_AZUL = [
    '#0d3d6b',
    '#2774b0',
    '#4690d0',
    '#74c0e0',
    '#a2d4f4',
    '#d0eaff'  # muy claro
]

# ============================
# 0. Configuración Inicial
# ============================
st.set_page_config(page_title="Dashboard Ventas - Tiendas de Conveniencia", layout="wide")
st.title("Análisis de Ventas de una Cadena de Tiendas de Conveniencia")
st.markdown("""
Este dashboard permite explorar el desempeño comercial de una **cadena de tiendas de conveniencia**.
Analiza datos clave por tienda, línea de producto y preferencias de clientes.
""")


# ============================
# 1. Cargar Datos
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", parse_dates=["Date"])
    # Convertir columnas a categórico
    for col in ['Gender', 'Product line', 'Payment', 'Customer type', 'Branch', 'City']:
        df[col] = df[col].astype('category')
    df['Month'] = df['Date'].dt.to_period('M')
    return df

df = load_data()

# ============================
# 2. Filtros en Sidebar
# ============================
st.sidebar.header("Filtros")
branches = st.sidebar.multiselect("Sucursal", df["Branch"].unique(), default=list(df["Branch"].unique()))
date_range = st.sidebar.date_input("Rango de Fechas", [df["Date"].min(), df["Date"].max()])

selected_lines = st.sidebar.multiselect("Líneas de Producto", options=df["Product line"].unique(), default=list(df["Product line"].unique()))
selected_genders = st.sidebar.multiselect("Género", options=df["Gender"].unique().tolist(), default=df["Gender"].unique().tolist())

options_payment = df["Payment"].unique().tolist()
default_payment = options_payment
selected_payment = st.sidebar.multiselect("Métodos de Pago", options=options_payment, default=default_payment)

# Filtrado de datos
filtered_df = df[
    (df["Branch"].isin(branches)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1])) &
    (df["Product line"].isin(selected_lines)) &
    (df["Gender"].isin(selected_genders)) &
    (df["Payment"].isin(selected_payment))
]

# ============================
# 3. Funciones de gráficos
# ============================

def plot_evolucion_ventas(df):
    ventas_diarias = df.groupby('Date')['Total'].sum().reset_index()
    ventas_diarias['Date'] = pd.to_datetime(ventas_diarias['Date'])
    ventas_diarias['DayOfWeek'] = ventas_diarias['Date'].dt.day_name()
    ventas_diarias['EsFinDeSemana'] = ventas_diarias['DayOfWeek'].isin(['Saturday', 'Sunday'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ventas_diarias['Date'],
        y=ventas_diarias['Total'],
        mode='lines',
        line=dict(color='#1883ba', width=2),
        name='Ventas Diarias'
    ))
    fines = ventas_diarias[ventas_diarias['EsFinDeSemana']]
    fig.add_trace(go.Scatter(
        x=fines['Date'],
        y=fines['Total'],
        mode='markers',
        marker=dict(color='#a2d4f4', size=8),
        name='Fines de semana'
    ))
    fig.update_layout(
        title='Ventas diarias con fines de semana destacados',
        xaxis_title='Fecha',
        yaxis_title='Ventas Totales',
        xaxis_tickangle=90,
        template='plotly_white',
        legend=dict(y=1.1, orientation='h')
    )
    return fig

def plot_ingresos_por_linea(df):
    grouped = df.groupby("Product line")["Total"].sum().sort_values(ascending=False)
    n_bars = len(grouped)
    colores = (TONOS_AZUL * ((n_bars // len(TONOS_AZUL)) + 1))[:n_bars]

    # Crear la gráfica de barras horizontales
    fig = px.bar(
        x=grouped.values,
        y=grouped.index,
        orientation='h',
        title='Ventas por Línea de Producto',
        color=grouped.index,
        color_discrete_sequence=colores
    )

    # Añadir anotaciones encima de cada barra
    for idx, valor in enumerate(grouped.values):
        fig.add_annotation(
            x=valor,
            y=grouped.index[idx],
            text=f'{valor:,.0f}',
            xanchor='left',
            yanchor='middle',
            showarrow=False,
            font=dict(color='black')
        )

    # Asegurar que las barras largas estén en la parte superior
    fig.update_layout(
        yaxis=dict(autorange='reversed'),
        xaxis_title='Total de Ingresos',
        yaxis_title='Línea de Producto',
        height=400 + 20 * n_bars,
        template='plotly_white'
    )

    return fig



def plot_distribucion_pago(df):
    payment_counts = df["Payment"].value_counts().reset_index()
    payment_counts.columns = ["Método de Pago", "Cantidad"]
    fig = px.pie(
        payment_counts,
        names="Método de Pago",
        values="Cantidad",
        title="Distribución de Métodos de Pago",
        color_discrete_sequence=TONOS_AZUL
    )
    return fig

def plot_ingreso_por_linea_sucursal(df):
    income_grouped = df.groupby(["Branch", "Product line"], observed=True)["gross income"].sum().reset_index()
    n_cols = len(income_grouped['Branch'].unique())
    colores = (TONOS_AZUL * ((n_cols // len(TONOS_AZUL)) + 1))[:n_cols]
    fig = px.bar(
        income_grouped,
        x="Product line",
        y="gross income",
        color="Branch",
        barmode="stack",
        title="Ingreso Bruto por Línea y Sucursal",
        color_discrete_sequence=colores
    )
    return fig

def plot_rating_distribution(df):
    ratings = df['Rating']
    hist = px.histogram(
        df,
        x='Rating',
        nbins=61,
        opacity=0.7,
        color_discrete_sequence=['#74c0e0']
    )
    x_min, x_max = ratings.min(), ratings.max()
    kde = stats.gaussian_kde(ratings)
    x_vals = np.linspace(x_min, x_max, 200)
    kde_vals = kde(x_vals)
    kde_trace = go.Scatter(
        x=x_vals,
        y=kde_vals * len(ratings) * (x_max - x_min ) / 60,
        mode='lines',
        line=dict(color='#0d3d6b', width=2),
        name='KDE'
    )
    fig = go.Figure(data=hist.data)
    fig.add_trace(kde_trace)
    fig.update_layout(
        xaxis=dict(range=[4, 10]),
        title='Distribución de Ratings con KDE',
        xaxis_title='Rating',
        yaxis_title='Frecuencia / Densidad',
        height=400,
        template='plotly_white'
    )
    return fig

def plot_cogs_vs_gross_income(df):
    tonos_azules_local = [
        '#d0eaff',
        '#4690d0',
        '#0d3d6b'
    ]
    fig = px.scatter(
        df,
        x='cogs',
        y='gross income',
        color='Branch',
        opacity=0.5,
        size_max=10,
        title='Relación Costo vs Ingreso Bruto',
        color_discrete_sequence=tonos_azules_local
    )
    return fig

def plot_gasto_tipo_cliente(df):
    fig = px.box(
        df,
        x='Customer type',
        y='Total',
        color='Customer type',
        title='Gasto por Tipo de Cliente',
        color_discrete_sequence=TONOS_AZUL
    )
    return fig

def plot_correlation_matrix(df):
    num_vars = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income', 'Rating']
    corr = df[num_vars].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', ax=ax)
    ax.set_title('Matriz de Correlación entre Variables Numéricas')
    return fig   

# ============================
# 4. Sistema de pestañas dinámico
# ============================

# Lista de tuplas con título y función
graf_list = [
    ("Evolutivo Diario", plot_evolucion_ventas),
    ("Ingreso por Línea", plot_ingresos_por_linea),
    ("Medios de Pago", plot_distribucion_pago),
    ("Ingreso por Sucursal", plot_ingreso_por_linea_sucursal),
    ("Ratings", plot_rating_distribution),
    ("Costo vs Ingreso", plot_cogs_vs_gross_income),
    ("Tipo de Cliente", plot_gasto_tipo_cliente),
    ("Correlaciones", plot_correlation_matrix)
]

# Creamos las pestañas en base a la título
tabs = st.tabs([titulo for titulo, _ in graf_list])

# Genera todas las figuras antes
graficos = {}
for titulo, func in graf_list:
    graficos[titulo] = func(filtered_df)

# Luego en el loop
for i, (titulo, _) in enumerate(graf_list):
    with tabs[i]:
        st.header(titulo)
        try:
            fig_or_ax = graficos[titulo]
            if isinstance(fig_or_ax, plt.Figure):
                st.pyplot(fig_or_ax)
            else:
                st.plotly_chart(fig_or_ax, use_container_width=True)
        except Exception as e:
            st.write(f"Error en {titulo}: {e}")
            
# finalmente, agregamos el footer
st.markdown(
    """
    **Este dashboard se actualiza en tiempo real con los filtros seleccionados.**  
    Para obtener análisis más detallados o específicos, ajusta los filtros y explora las visualizaciones.
    """
)