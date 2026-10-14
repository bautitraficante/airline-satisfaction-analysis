from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="SkyPulse | Airline Passenger Experience",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


NAVY = "#071B33"
BLUE = "#1473E6"
SKY = "#41B8F4"
CYAN = "#5CE1E6"
RED = "#F2645A"
CREAM = "#F5F8FC"
MUTED = "#718096"


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    .stApp {{ background: {CREAM}; color: {NAVY}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #06172C 0%, #0B2A4A 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }}
    [data-testid="stSidebar"] * {{ color: #F8FBFF; }}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
        background: {BLUE};
    }}
    [data-testid="stSidebar"] input {{ color: {NAVY} !important; }}

    .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1450px; }}
    h1, h2, h3 {{ font-family: 'Manrope', sans-serif; color: {NAVY}; }}

    .hero {{
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.5rem;
        border-radius: 28px;
        background:
          radial-gradient(circle at 88% 15%, rgba(92,225,230,.32), transparent 25%),
          linear-gradient(120deg, #06172C 0%, #0A3863 60%, #126EB5 100%);
        color: white;
        box-shadow: 0 22px 55px rgba(7,27,51,.18);
        margin-bottom: 1.4rem;
    }}
    .hero:after {{
        content: "✈";
        position: absolute;
        right: 5%; top: 7%;
        font-size: 8.5rem;
        opacity: .12;
        transform: rotate(-8deg);
    }}
    .eyebrow {{
        text-transform: uppercase; letter-spacing: .18em; font-size: .78rem;
        font-weight: 700; color: {CYAN}; margin-bottom: .7rem;
    }}
    .hero h1 {{ color: white; font-size: clamp(2.1rem, 4vw, 4rem); margin: 0; line-height: 1.03; }}
    .hero p {{ color: rgba(255,255,255,.82); font-size: 1.05rem; max-width: 760px; margin: .9rem 0 0; }}
    .route {{ margin-top: 1.35rem; display: flex; align-items: center; gap: .8rem; font-weight: 700; }}
    .route-line {{ height: 1px; width: 120px; background: repeating-linear-gradient(90deg, {CYAN} 0 8px, transparent 8px 14px); }}

    [data-testid="stMetric"] {{
        background: white;
        border: 1px solid #E2EAF3;
        border-radius: 18px;
        padding: 1.05rem 1.2rem;
        box-shadow: 0 9px 28px rgba(7,27,51,.06);
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: {NAVY}; font-family: 'Manrope', sans-serif; }}

    .insight-card {{
        background: white;
        border: 1px solid #E2EAF3;
        border-left: 5px solid {SKY};
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin: .5rem 0 1rem;
        box-shadow: 0 7px 22px rgba(7,27,51,.05);
    }}
    .insight-card strong {{ color: {NAVY}; }}
    .mini-label {{ color: {MUTED}; text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 700; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: .5rem; background: white; padding: .42rem; border-radius: 14px; border: 1px solid #E2EAF3; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 10px; padding: .55rem 1rem; font-weight: 700; }}
    .stTabs [aria-selected="true"] {{ background: {NAVY}; color: white; }}

    .ai-note {{
        margin-top: 2rem; padding: .9rem 1.1rem; border-radius: 14px;
        color: #526273; background: #EAF1F8; font-size: .86rem;
    }}
    .model-badge {{
        display: inline-flex; align-items: center; gap: .45rem;
        padding: .38rem .7rem; border-radius: 999px;
        background: rgba(20,115,230,.10); color: {BLUE};
        font-size: .78rem; font-weight: 700; margin-bottom: .7rem;
    }}
    .prediction-card {{
        border-radius: 20px; padding: 1.25rem 1.4rem; margin-top: .8rem;
        background: linear-gradient(135deg, #071B33 0%, #0D4F83 100%);
        color: white; box-shadow: 0 16px 35px rgba(7,27,51,.16);
    }}
    .prediction-card strong {{ color: {CYAN}; font-size: 1.25rem; }}
    .footer {{ text-align:center; color:#7C8A98; font-size:.82rem; padding-top:1.2rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    candidates = [
        base / "data" / "processed" / "airline_train_clean.csv",
        base / "airline_train_clean.csv",
        base.parent / "data" / "processed" / "airline_train_clean.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        "No se encontró airline_train_clean.csv. "
        "Guardalo en data/processed/ dentro del proyecto."
    )


@st.cache_resource
def load_model():
    base = Path(__file__).resolve().parent
    candidates = [
        base / "models" / "modelo_satisfaccion_random_forest.joblib",
        base.parent / "models" / "modelo_satisfaccion_random_forest.joblib",
    ]
    for path in candidates:
        if path.exists():
            return joblib.load(path)
    return None


def pct_satisfied(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return frame["satisfaction"].eq("satisfied").mean() * 100


def percentage_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = (
        frame.assign(satisfied_num=frame["satisfaction"].eq("satisfied").astype(int))
        .groupby(column, observed=True)
        .agg(passengers=("satisfied_num", "size"), satisfaction=("satisfied_num", "mean"))
        .reset_index()
    )
    result["satisfaction"] *= 100
    return result


def style_figure(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=NAVY),
        title_font=dict(family="Manrope", size=19, color=NAVY),
        hoverlabel=dict(bgcolor=NAVY, font_color="white"),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(7,27,51,.08)", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


try:
    df = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

modelo_final = load_model()


SERVICE_COLUMNS = [
    "inflight_wifi_service",
    "departure_arrival_time_convenient",
    "ease_of_online_booking",
    "gate_location",
    "food_and_drink",
    "online_boarding",
    "seat_comfort",
    "inflight_entertainment",
    "on_board_service",
    "leg_room_service",
    "baggage_handling",
    "checkin_service",
    "inflight_service",
    "cleanliness",
]

SERVICE_NAMES = {
    "online_boarding": "Embarque online",
    "inflight_entertainment": "Entretenimiento a bordo",
    "seat_comfort": "Comodidad del asiento",
    "on_board_service": "Servicio a bordo",
    "leg_room_service": "Espacio para las piernas",
    "cleanliness": "Limpieza",
    "inflight_wifi_service": "Wifi durante el vuelo",
    "baggage_handling": "Gestión del equipaje",
    "checkin_service": "Servicio de check-in",
    "inflight_service": "Atención durante el vuelo",
    "food_and_drink": "Comida y bebida",
    "ease_of_online_booking": "Facilidad de reserva online",
    "gate_location": "Ubicación de la puerta",
    "departure_arrival_time_convenient": "Conveniencia de horarios",
}

CLASS_NAMES = {"business": "Business", "eco": "Eco", "eco plus": "Eco Plus"}
TRAVEL_NAMES = {"business travel": "Negocios", "personal travel": "Personal"}
CUSTOMER_NAMES = {"loyal customer": "Fiel", "disloyal customer": "No fiel"}
GENDER_NAMES = {"female": "Mujer", "male": "Hombre"}


with st.sidebar:
    st.markdown("## ✈ SkyPulse")
    st.caption("Centro de control de experiencia")
    st.markdown("---")
    st.markdown("### Filtros de vuelo")

    selected_class = st.multiselect(
        "Clase",
        options=list(CLASS_NAMES),
        default=list(CLASS_NAMES),
        format_func=lambda value: CLASS_NAMES[value],
    )
    selected_travel = st.multiselect(
        "Tipo de viaje",
        options=list(TRAVEL_NAMES),
        default=list(TRAVEL_NAMES),
        format_func=lambda value: TRAVEL_NAMES[value],
    )
    selected_customer = st.multiselect(
        "Fidelidad",
        options=list(CUSTOMER_NAMES),
        default=list(CUSTOMER_NAMES),
        format_func=lambda value: CUSTOMER_NAMES[value],
    )
    selected_gender = st.multiselect(
        "Género",
        options=list(GENDER_NAMES),
        default=list(GENDER_NAMES),
        format_func=lambda value: GENDER_NAMES[value],
    )
    age_range = st.slider(
        "Edad",
        int(df["age"].min()),
        int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max())),
    )
    st.markdown("---")
    st.caption("Los filtros actualizan todo el tablero.")


filtered = df[
    df["class"].isin(selected_class)
    & df["type_of_travel"].isin(selected_travel)
    & df["customer_type"].isin(selected_customer)
    & df["gender"].isin(selected_gender)
    & df["age"].between(age_range[0], age_range[1])
].copy()

if filtered.empty:
    st.warning("No hay pasajeros que coincidan con los filtros seleccionados.")
    st.stop()

filtered["satisfaction_numeric"] = filtered["satisfaction"].eq("satisfied").astype(int)
filtered["delay_group"] = pd.cut(
    filtered["arrival_delay_in_minutes"],
    bins=[-1, 0, 15, 30, 60, np.inf],
    labels=["Sin retraso", "1–15 min", "16–30 min", "31–60 min", "+60 min"],
)


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Passenger experience intelligence</div>
      <h1>Airline Satisfaction<br>Control Center</h1>
      <p>Un tablero interactivo para entender qué impulsa la satisfacción, dónde aparecen las fricciones y qué segmentos requieren atención.</p>
      <div class="route"><span>DATA</span><span class="route-line"></span><span>INSIGHT</span><span class="route-line"></span><span>ACTION</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)


satisfaction_rate = pct_satisfied(filtered)
average_arrival_delay = filtered["arrival_delay_in_minutes"].mean()
loyal_share = filtered["customer_type"].eq("loyal customer").mean() * 100

service_corr = (
    filtered[SERVICE_COLUMNS]
    .corrwith(filtered["satisfaction_numeric"])
    .sort_values(ascending=False)
)
top_service = SERVICE_NAMES.get(service_corr.index[0], service_corr.index[0])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Pasajeros analizados", f"{len(filtered):,}".replace(",", "."))
k2.metric("Satisfacción", f"{satisfaction_rate:.1f}%", f"{satisfaction_rate - 43.34:+.1f} pp vs. total")
k3.metric("Retraso medio de llegada", f"{average_arrival_delay:.1f} min")
k4.metric("Servicio más relacionado", top_service)


tab_overview, tab_segments, tab_experience, tab_delays, tab_profiles, tab_model = st.tabs(
    [
        "Torre de control",
        "Segmentos",
        "Experiencia",
        "Puntualidad",
        "Perfiles de riesgo",
        "Modelo predictivo",
    ]
)


with tab_overview:
    left, right = st.columns([1.1, 1])
    with left:
        satisfaction_counts = (
            filtered["satisfaction"]
            .value_counts()
            .rename(index={"satisfied": "Satisfecho", "neutral or dissatisfied": "Neutral o insatisfecho"})
            .reset_index()
        )
        satisfaction_counts.columns = ["Estado", "Pasajeros"]
        fig = px.pie(
            satisfaction_counts,
            names="Estado",
            values="Pasajeros",
            hole=.68,
            color="Estado",
            color_discrete_map={"Satisfecho": SKY, "Neutral o insatisfecho": RED},
            title="Estado general de satisfacción",
        )
        fig.update_traces(textposition="inside", textinfo="percent", marker_line_color="white", marker_line_width=3)
        fig.add_annotation(text=f"<b>{satisfaction_rate:.1f}%</b><br>satisfecho", x=.5, y=.5, showarrow=False, font_size=17)
        st.plotly_chart(style_figure(fig), width="stretch")

    with right:
        by_class = percentage_table(filtered, "class")
        by_class["class"] = by_class["class"].replace(CLASS_NAMES)
        fig = px.bar(
            by_class,
            x="class",
            y="satisfaction",
            text="satisfaction",
            color="satisfaction",
            color_continuous_scale=["#D9E8F6", SKY, BLUE],
            title="Satisfacción por clase",
            labels={"class": "Clase", "satisfaction": "Satisfacción (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_figure(fig), width="stretch")

    st.markdown(
        f"""
        <div class="insight-card"><div class="mini-label">Lectura rápida</div>
        <strong>Menos de la mitad de los pasajeros está satisfecha.</strong> En la muestra completa, Business alcanza 69,43 % de satisfacción, mientras Eco queda en 18,61 %. Usá los filtros para comprobar cómo cambia esta brecha por edad, fidelidad o motivo del viaje.</div>
        """,
        unsafe_allow_html=True,
    )


with tab_segments:
    col1, col2 = st.columns(2)
    with col1:
        travel = percentage_table(filtered, "type_of_travel")
        travel["type_of_travel"] = travel["type_of_travel"].replace(TRAVEL_NAMES)
        fig = px.bar(
            travel,
            x="type_of_travel",
            y="satisfaction",
            text="satisfaction",
            color="type_of_travel",
            color_discrete_sequence=[BLUE, SKY],
            title="Satisfacción por motivo de viaje",
            labels={"type_of_travel": "Tipo de viaje", "satisfaction": "Satisfacción (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_figure(fig), width="stretch")

    with col2:
        customer = percentage_table(filtered, "customer_type")
        customer["customer_type"] = customer["customer_type"].replace(CUSTOMER_NAMES)
        fig = px.bar(
            customer,
            x="customer_type",
            y="satisfaction",
            text="satisfaction",
            color="customer_type",
            color_discrete_sequence=["#A9C8E8", BLUE],
            title="Satisfacción por fidelidad",
            labels={"customer_type": "Cliente", "satisfaction": "Satisfacción (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_figure(fig), width="stretch")

    matrix = (
        filtered.groupby(["type_of_travel", "class"], observed=True)
        .agg(passengers=("satisfaction_numeric", "size"), satisfaction=("satisfaction_numeric", "mean"))
        .reset_index()
    )
    matrix["satisfaction"] *= 100
    matrix["type_of_travel"] = matrix["type_of_travel"].replace(TRAVEL_NAMES)
    matrix["class"] = matrix["class"].replace(CLASS_NAMES)
    fig = px.bar(
        matrix,
        x="class",
        y="satisfaction",
        color="type_of_travel",
        barmode="group",
        text="satisfaction",
        custom_data=["passengers"],
        title="La clase no cuenta toda la historia",
        labels={"class": "Clase", "satisfaction": "Satisfacción (%)", "type_of_travel": "Viaje"},
        color_discrete_map={"Negocios": BLUE, "Personal": RED},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}%<br>n=%{customdata[0]:,.0f}<extra></extra>")
    st.plotly_chart(style_figure(fig, 470), width="stretch")


with tab_experience:
    correlations = (
        filtered[SERVICE_COLUMNS]
        .corrwith(filtered["satisfaction_numeric"])
        .sort_values(ascending=True)
        .reset_index()
    )
    correlations.columns = ["service", "correlation"]
    correlations["service"] = correlations["service"].replace(SERVICE_NAMES)
    correlations["direction"] = np.where(correlations["correlation"] >= 0, "Positiva", "Negativa")

    fig = px.bar(
        correlations,
        x="correlation",
        y="service",
        orientation="h",
        text="correlation",
        color="direction",
        color_discrete_map={"Positiva": BLUE, "Negativa": RED},
        title="Servicios más relacionados con la satisfacción",
        labels={"correlation": "Correlación", "service": "Servicio"},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_figure(fig, 600), width="stretch")

    st.markdown(
        """
        <div class="insight-card"><div class="mini-label">Prioridad de acción</div>
        <strong>Embarque online, entretenimiento y comodidad</strong> lideran la relación con la satisfacción en la muestra completa. La correlación orienta prioridades, pero no prueba causalidad: conviene validar mejoras mediante pruebas controladas.</div>
        """,
        unsafe_allow_html=True,
    )


with tab_delays:
    delay = percentage_table(filtered, "delay_group")
    fig = px.bar(
        delay,
        x="delay_group",
        y="satisfaction",
        text="satisfaction",
        custom_data=["passengers"],
        color="satisfaction",
        color_continuous_scale=[RED, "#F2C36B", SKY],
        title="Satisfacción según el retraso de llegada",
        labels={"delay_group": "Retraso", "satisfaction": "Satisfacción (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", hovertemplate="%{x}<br>Satisfacción: %{y:.1f}%<br>n=%{customdata[0]:,.0f}<extra></extra>")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_figure(fig), width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mediana de demora", f"{filtered['arrival_delay_in_minutes'].median():.0f} min")
    c2.metric("Vuelos sin demora", f"{filtered['arrival_delay_in_minutes'].eq(0).mean()*100:.1f}%")
    c3.metric("Demoras de +60 min", f"{filtered['arrival_delay_in_minutes'].gt(60).mean()*100:.1f}%")


with tab_profiles:
    profiles = (
        filtered.groupby(["customer_type", "type_of_travel", "class"], observed=True)
        .agg(passengers=("satisfaction_numeric", "size"), satisfaction=("satisfaction_numeric", "mean"))
        .reset_index()
    )
    minimum_size = st.slider("Muestra mínima por perfil", 50, 2000, 500, 50)
    profiles = profiles[profiles["passengers"] >= minimum_size].copy()
    profiles["not_satisfied"] = (1 - profiles["satisfaction"]) * 100
    profiles["profile"] = (
        profiles["customer_type"].replace(CUSTOMER_NAMES)
        + " | " + profiles["type_of_travel"].replace(TRAVEL_NAMES)
        + " | " + profiles["class"].replace(CLASS_NAMES)
        + " | n=" + profiles["passengers"].astype(str)
    )
    profiles = profiles.sort_values("not_satisfied", ascending=True)

    if profiles.empty:
        st.info("No hay perfiles que alcancen la muestra mínima seleccionada.")
    else:
        fig = px.bar(
            profiles,
            x="not_satisfied",
            y="profile",
            orientation="h",
            text="not_satisfied",
            color="not_satisfied",
            color_continuous_scale=[SKY, "#F2C36B", RED],
            title="Perfiles con mayor riesgo de no satisfacción",
            labels={"not_satisfied": "Neutral o insatisfecho (%)", "profile": "Perfil"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_figure(fig, 570), width="stretch")

        highest = profiles.iloc[-1]
        largest_opportunity = profiles.assign(affected=profiles["passengers"] * profiles["not_satisfied"] / 100).sort_values("affected").iloc[-1]
        st.markdown(
            f"""
            <div class="insight-card"><div class="mini-label">Riesgo vs. volumen</div>
            El mayor riesgo filtrado corresponde a <strong>{highest['profile']}</strong> ({highest['not_satisfied']:.1f} % no satisfecho).
            La mayor oportunidad por volumen es <strong>{largest_opportunity['profile']}</strong>, porque combina una tasa elevada con una base amplia de pasajeros.</div>
            """,
            unsafe_allow_html=True,
        )


with tab_model:
    st.markdown('<div class="model-badge">🤖 PRIMER MODELO DE MACHINE LEARNING</div>', unsafe_allow_html=True)
    st.subheader("Predicción de satisfacción")
    st.write(
        "Se comparó una regresión logística con dos versiones de Random Forest. "
        "El bosque original fue seleccionado por su rendimiento y estabilidad en validación cruzada."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy de prueba", "96,26 %", "+9,08 pp vs. logística")
    m2.metric("Validación cruzada", "96,10 %", "promedio de 5 divisiones")
    m3.metric("Recall insatisfechos", "97,82 %", "11.484 de 11.740")
    m4.metric("Errores totales", "774", "sobre 20.719 casos")

    left, right = st.columns([.95, 1.2])
    with left:
        confusion = np.array([[11484, 256], [518, 8461]])
        fig = px.imshow(
            confusion,
            text_auto=True,
            color_continuous_scale=[[0, "#EAF1F8"], [1, BLUE]],
            x=["Predicho: no satisfecho", "Predicho: satisfecho"],
            y=["Real: no satisfecho", "Real: satisfecho"],
            title="Matriz de confusión",
            aspect="auto",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_figure(fig, 420), width="stretch")

    with right:
        importance = pd.DataFrame(
            {
                "Variable": [
                    "Embarque online",
                    "Wi-Fi a bordo",
                    "Viaje personal",
                    "Clase Eco",
                    "Entretenimiento a bordo",
                    "Comodidad del asiento",
                    "Espacio para las piernas",
                    "Distancia del vuelo",
                    "Facilidad de reserva online",
                    "Cliente fiel",
                ],
                "Importancia": [18.08, 13.88, 10.91, 6.42, 5.50, 5.14, 4.05, 3.88, 3.72, 3.57],
            }
        ).sort_values("Importancia")
        fig = px.bar(
            importance,
            x="Importancia",
            y="Variable",
            orientation="h",
            text="Importancia",
            color="Importancia",
            color_continuous_scale=["#BFDDF5", BLUE],
            title="Variables más utilizadas por el modelo",
            labels={"Importancia": "Importancia (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_figure(fig, 500), width="stretch")

    st.markdown(
        """
        <div class="insight-card"><div class="mini-label">Cómo interpretar el modelo</div>
        <strong>La importancia no indica dirección ni causalidad.</strong> Muestra cuánto utiliza Random Forest cada variable para tomar decisiones. Para saber si un factor se relaciona con mayor o menor satisfacción, se complementa con el análisis descriptivo y la regresión logística.</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Simulador de pasajero")
    st.caption("Completá un perfil hipotético para obtener una predicción del modelo final.")

    if modelo_final is None:
        st.info(
            "El simulador se activará cuando `modelo_satisfaccion_random_forest.joblib` "
            "esté guardado dentro de la carpeta `models` del proyecto."
        )
    else:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Género", ["female", "male"], format_func=lambda x: GENDER_NAMES[x])
                customer_type = st.selectbox(
                    "Tipo de cliente",
                    ["loyal customer", "disloyal customer"],
                    format_func=lambda x: CUSTOMER_NAMES[x],
                )
                age = st.slider("Edad", 7, 85, 39)
                type_of_travel = st.selectbox(
                    "Tipo de viaje",
                    ["business travel", "personal travel"],
                    format_func=lambda x: TRAVEL_NAMES[x],
                )
                travel_class = st.selectbox(
                    "Clase",
                    ["business", "eco", "eco plus"],
                    format_func=lambda x: CLASS_NAMES[x],
                )
                flight_distance = st.number_input("Distancia del vuelo", 31, 4983, 1000)

            with c2:
                st.markdown("**Experiencia digital y aeropuerto**")
                wifi = st.slider("Wi-Fi a bordo", 0, 5, 3)
                time_convenient = st.slider("Conveniencia de horarios", 0, 5, 3)
                online_booking = st.slider("Facilidad de reserva online", 0, 5, 3)
                gate = st.slider("Ubicación de la puerta", 0, 5, 3)
                online_boarding = st.slider("Embarque online", 0, 5, 3)
                checkin = st.slider("Servicio de check-in", 0, 5, 3)

            with c3:
                st.markdown("**Experiencia durante el vuelo**")
                food = st.slider("Comida y bebida", 0, 5, 3)
                seat = st.slider("Comodidad del asiento", 0, 5, 3)
                entertainment = st.slider("Entretenimiento", 0, 5, 3)
                onboard = st.slider("Servicio a bordo", 0, 5, 3)
                legroom = st.slider("Espacio para las piernas", 0, 5, 3)
                baggage = st.slider("Gestión del equipaje", 1, 5, 3)
                inflight = st.slider("Atención durante el vuelo", 0, 5, 3)
                cleanliness = st.slider("Limpieza", 0, 5, 3)

            d1, d2 = st.columns(2)
            departure_delay = d1.number_input("Retraso de salida (min)", 0, 1600, 0)
            arrival_delay = d2.number_input("Retraso de llegada (min)", 0.0, 1600.0, 0.0)
            submitted = st.form_submit_button("Predecir satisfacción", type="primary", width="stretch")

        if submitted:
            passenger = pd.DataFrame(
                [{
                    "gender": gender,
                    "customer_type": customer_type,
                    "age": age,
                    "type_of_travel": type_of_travel,
                    "class": travel_class,
                    "flight_distance": flight_distance,
                    "inflight_wifi_service": wifi,
                    "departure_arrival_time_convenient": time_convenient,
                    "ease_of_online_booking": online_booking,
                    "gate_location": gate,
                    "food_and_drink": food,
                    "online_boarding": online_boarding,
                    "seat_comfort": seat,
                    "inflight_entertainment": entertainment,
                    "on_board_service": onboard,
                    "leg_room_service": legroom,
                    "baggage_handling": baggage,
                    "checkin_service": checkin,
                    "inflight_service": inflight,
                    "cleanliness": cleanliness,
                    "departure_delay_in_minutes": departure_delay,
                    "arrival_delay_in_minutes": arrival_delay,
                }]
            )
            prediction = int(modelo_final.predict(passenger)[0])
            probability = float(modelo_final.predict_proba(passenger)[0][prediction]) * 100
            label = "Satisfecho" if prediction == 1 else "Neutral o insatisfecho"
            st.markdown(
                f'<div class="prediction-card">Predicción del modelo<br><strong>{label}</strong><br>Confianza estimada: {probability:.1f} %</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Resultado experimental basado en patrones históricos del dataset. "
                "No representa una certeza ni una relación causal."
            )


st.markdown(
    """
    <div class="ai-note"><strong>Transparencia:</strong> la limpieza, el análisis, las decisiones metodológicas y las conclusiones fueron trabajadas por Bautista Traficante como proyecto de aprendizaje. El modelado predictivo fue desarrollado como ejercicio guiado, con asistencia de inteligencia artificial en la estructura del código y la explicación de conceptos. La interfaz interactiva y parte de su implementación también fueron creadas con asistencia de IA.</div>
    <div class="footer">SkyPulse · Airline Passenger Satisfaction Analysis · Portfolio Project</div>
    """,
    unsafe_allow_html=True,
)
