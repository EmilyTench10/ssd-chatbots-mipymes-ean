"""Prototipo Streamlit del sistema de soporte a la decisión para
mipymes del sector servicios: selección entre RAG, fine-tuning e
híbrida para chatbots empresariales.

Trabajo de grado — Maestría en Ciencia de Datos y Maestría en
Inteligencia de Negocios, Universidad Ean.
Autoras: Laura Ximena Agudelo Bastidas, Andrea Carolina Conde Avila,
Emily Del Carmen Tench Pérez. Directora: Carolina María Luque Zabala.

Uso local:
    streamlit run app.py

Despliegue en Streamlit Community Cloud:
    1. Subir este directorio como repositorio GitHub público.
    2. Conectarlo en share.streamlit.io.
    3. Configurar en Secrets:
         LLM_API_KEY = "..."
         LLM_MODELO = "gemini-2.0-flash-exp"
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

HERE = Path(__file__).resolve().parent
MODELO_PATH = HERE / "modelo" / "modelo_v2.pkl"
PERFILES_PATH = HERE / "datos" / "perfiles_243_regla_v2.csv"


# ---------------------------------------------------------------------------
# Preguntas al usuario (Tabla de equivalencias técnicas → operativas)
# ---------------------------------------------------------------------------
PREGUNTAS = {
    "volumen": {
        "pregunta": "¿Cuántas páginas de documentación tiene su empresa "
                    "entre preguntas frecuentes, tarifas, políticas y "
                    "procedimientos?",
        "opciones": [
            ("Menos de 75 páginas", 1),
            ("Entre 75 y 450 páginas", 2),
            ("Entre 450 y 1 500 páginas", 3),
        ],
    },
    "volatilidad": {
        "pregunta": "¿Qué porcentaje de la documentación de su empresa "
                    "cambia cada mes (precios, políticas, servicios)?",
        "opciones": [
            ("Menos del 10 %", 1),
            ("Entre el 10 % y el 25 %", 2),
            ("Más del 25 %", 3),
        ],
    },
    "presupuesto": {
        "pregunta": "¿Qué presupuesto puede destinar al chatbot?",
        "opciones": [
            ("Hasta USD 1 500 iniciales y USD 150 al mes", 1),
            ("Entre USD 1 500 y 5 000 iniciales y USD 150 a 800 al mes", 2),
            ("Más de USD 5 000 iniciales y más de USD 800 al mes", 3),
        ],
    },
    "tiempo": {
        "pregunta": "¿En cuántas semanas necesita tener el chatbot funcionando?",
        "opciones": [
            ("Entre 1 y 4 semanas", 1),
            ("Entre 5 y 8 semanas", 2),
            ("Entre 9 y 12 semanas", 3),
        ],
    },
    "riesgo": {
        "pregunta": "¿Qué tan grave sería que el chatbot le entregue "
                    "una respuesta equivocada a un cliente?",
        "opciones": [
            ("Poco grave (1 a 2 en escala de 1 a 5)", 1),
            ("Grave (3 en escala de 1 a 5)", 2),
            ("Muy grave (4 a 5 en escala de 1 a 5)", 3),
        ],
    },
}

VAR_NOMBRES = {
    "volumen": "volumen de datos",
    "volatilidad": "volatilidad de la información",
    "presupuesto": "presupuesto disponible",
    "tiempo": "tiempo de desarrollo",
    "riesgo": "riesgo por alucinación",
}
NIVELES = {1: "bajo", 2: "medio", 3: "alto"}

DESCRIPCIONES_ARQ = {
    "RAG": "Recuperación de información (RAG). El chatbot busca en sus "
           "documentos en el momento de responder, sin modificar el "
           "modelo de lenguaje. Es más barato, más rápido de implementar "
           "y se adapta fácilmente a cambios en la información.",
    "Fine-tuning": "Ajuste fino. Se entrena el modelo de lenguaje con "
                   "sus datos para que los aprenda. Requiere más tiempo "
                   "e inversión inicial pero puede ajustar el estilo de "
                   "las respuestas.",
    "Híbrida": "Arquitectura híbrida. Combina el ajuste fino con la "
               "recuperación de información. Ofrece un balance entre "
               "adaptación al tono y actualización continua, al costo "
               "de mayor complejidad de implementación.",
}


# ---------------------------------------------------------------------------
# Carga única (cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_recursos():
    with open(MODELO_PATH, "rb") as f:
        info = pickle.load(f)
    perfiles = pd.read_csv(PERFILES_PATH, encoding="utf-8-sig")
    variables = info["features"]
    background = perfiles[variables].astype(int).values
    rng = np.random.default_rng(2026)
    idx = rng.choice(len(background), size=min(50, len(background)),
                     replace=False)
    return info, background[idx]


def predecir(info, perfil, background):
    modelo = info["modelo"]
    variables = info["features"]
    clases = info["clases"]
    X = np.array([[perfil[v] for v in variables]])
    y_pred_idx = int(modelo.predict(X)[0])
    y_pred = clases[y_pred_idx]
    proba = modelo.predict_proba(X)[0]
    proba_dict = {clases[i]: float(proba[i]) for i in range(len(clases))}

    explainer = shap.Explainer(modelo.predict_proba, background)
    explanation = explainer(X)
    sv = explanation.values[0]  # (n_features,) o (n_features, n_classes)
    if sv.ndim == 2:
        sv_pred = sv[:, y_pred_idx]
    else:
        sv_pred = sv
    contrib = {var: float(sv_pred[i]) for i, var in enumerate(variables)}
    return y_pred, proba_dict, contrib


def explicar_lenguaje_natural(perfil, recomendacion, contribuciones):
    """Genera explicación en lenguaje natural usando Gemini.
    Si no hay API key configurada, devuelve una explicación basada en
    reglas (fallback)."""
    try:
        from google import genai
        api_key = st.secrets.get("LLM_API_KEY", None)
        modelo_nombre = st.secrets.get("LLM_MODELO",
                                        "gemini-2.0-flash-exp")
        if not api_key:
            return _explicacion_fallback(perfil, recomendacion,
                                          contribuciones)
        niveles_desc = ", ".join(
            f"{VAR_NOMBRES[v]} {NIVELES[perfil[v]]}" for v in
            ["volumen", "volatilidad", "presupuesto", "tiempo", "riesgo"]
        )
        contribs_ord = sorted(contribuciones.items(),
                              key=lambda kv: abs(kv[1]), reverse=True)
        tabla = "\n".join(
            f"  - {VAR_NOMBRES[v]} ({NIVELES[perfil[v]]}): "
            f"{'empuja hacia' if c > 0 else 'empuja en contra de'} "
            f"{recomendacion} (SHAP = {c:+.3f})"
            for v, c in contribs_ord
        )
        prompt = f"""Eres un asesor que explica a una mipyme en lenguaje
claro por qué un sistema de soporte a la decisión le recomienda una
arquitectura de chatbot.

Perfil de la mipyme: {niveles_desc}.
Arquitectura recomendada por el sistema: {recomendacion}.

El sistema analizó cinco variables y midió cuánto empuja cada una a
favor o en contra de la recomendación (valores SHAP). Los valores
observados fueron:
{tabla}

Redacta una explicación de 3 a 5 frases, en español neutral y
profesional, que:
1. Mencione la arquitectura recomendada y por qué (los dos factores
con mayor peso).
2. Reconozca honestamente cualquier factor que empuje en contra, si
lo hay.
3. Use lenguaje accesible para un dueño de mipyme sin formación
técnica.
No uses listas ni viñetas. No menciones los números SHAP; traduce su
dirección a palabras ("influye mucho a favor", "pesa en contra",
"aporta poco"). Comienza con "La recomendación es...".
"""
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=modelo_nombre, contents=prompt,
            config={"temperature": 0.3},
        )
        return resp.text
    except Exception as e:
        return _explicacion_fallback(perfil, recomendacion,
                                      contribuciones) + \
            f"\n\n(Nota: no se pudo generar la explicación con LLM: {e})"


def _explicacion_fallback(perfil, recomendacion, contribuciones):
    """Explicación basada en reglas por si no hay API key."""
    contribs_ord = sorted(contribuciones.items(),
                          key=lambda kv: abs(kv[1]), reverse=True)
    top2 = [v for v, _ in contribs_ord[:2]]
    partes = [f"La recomendación es adoptar una arquitectura "
              f"**{recomendacion}** para su chatbot."]
    razones = []
    for v in top2:
        c = contribuciones[v]
        if c > 0:
            razones.append(f"{NIVELES[perfil[v]]} {VAR_NOMBRES[v]} "
                           f"respalda esta elección")
        else:
            razones.append(f"{NIVELES[perfil[v]]} {VAR_NOMBRES[v]} "
                           f"pesa en contra")
    if razones:
        partes.append("Los dos factores de mayor peso son: " +
                      "; ".join(razones) + ".")
    return " ".join(partes)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SSD de arquitecturas de chatbot para mipymes",
    page_icon="🤖",
    layout="centered",
)

st.title("Sistema de soporte a la decisión")
st.markdown(
    "**Selección de arquitectura de chatbot para mipymes del sector "
    "servicios**"
)
st.caption(
    "Prototipo del trabajo de grado — Maestría en Ciencia de Datos "
    "y Maestría en Inteligencia de Negocios, Universidad Ean. "
    "Autoras: Laura Ximena Agudelo Bastidas, Andrea Carolina Conde "
    "Avila, Emily Del Carmen Tench Pérez. Directora: Carolina María "
    "Luque Zabala."
)

st.markdown("---")
st.markdown(
    "Responda las cinco preguntas siguientes sobre su empresa. El "
    "sistema le recomendará una de tres arquitecturas (RAG, ajuste "
    "fino o híbrida) y le dará una explicación de por qué."
)

with st.form("formulario"):
    perfil = {}
    for var, info_preg in PREGUNTAS.items():
        etiquetas = [op[0] for op in info_preg["opciones"]]
        seleccionado = st.radio(
            info_preg["pregunta"], etiquetas, key=var, index=1,
        )
        perfil[var] = next(op[1] for op in info_preg["opciones"]
                           if op[0] == seleccionado)
    enviado = st.form_submit_button("Obtener recomendación")

if enviado:
    with st.spinner("Analizando perfil y calculando recomendación..."):
        info, background = cargar_recursos()
        recomendacion, probas, contribuciones = predecir(
            info, perfil, background)

    st.markdown("---")
    st.subheader(f"Recomendación: {recomendacion}")
    st.markdown(DESCRIPCIONES_ARQ[recomendacion])

    st.markdown("### Probabilidades por arquitectura")
    prob_df = pd.DataFrame({
        "Arquitectura": list(probas.keys()),
        "Probabilidad": list(probas.values()),
    }).sort_values("Probabilidad", ascending=False)
    prob_df["Probabilidad"] = prob_df["Probabilidad"].apply(
        lambda x: f"{x:.1%}")
    st.table(prob_df.reset_index(drop=True))

    st.markdown("### Factores que explican la recomendación (SHAP)")
    shap_df = pd.DataFrame({
        "Variable": [VAR_NOMBRES[v] for v in contribuciones],
        "Nivel elegido": [NIVELES[perfil[v]] for v in contribuciones],
        "Contribución": [
            f"{'+' if c > 0 else ''}{c:.3f}"
            for c in contribuciones.values()],
        "Dirección": [
            "A favor" if c > 0 else "En contra"
            for c in contribuciones.values()],
    })
    shap_df = shap_df.sort_values(
        "Contribución",
        key=lambda col: col.str.replace("+", "").astype(float).abs(),
        ascending=False,
    )
    st.table(shap_df.reset_index(drop=True))

    st.markdown("### Explicación en lenguaje natural")
    with st.spinner("Generando explicación..."):
        try:
            explicacion = explicar_lenguaje_natural(
                perfil, recomendacion, contribuciones)
            st.success(explicacion)
        except Exception as e:
            st.warning(
                f"No se pudo generar la explicación en lenguaje "
                f"natural: {e}. Revise los valores SHAP en la tabla "
                "anterior."
            )

    with st.expander("Detalle técnico"):
        st.json({
            "perfil_niveles": perfil,
            "recomendacion": recomendacion,
            "probabilidades": probas,
            "contribuciones_shap": contribuciones,
            "clasificador": info.get("modelo_seleccionado",
                                      "Random Forest"),
        })

st.markdown("---")
st.caption(
    "Modelo predictivo: Random Forest entrenado sobre 237 perfiles "
    "(243 del diseño factorial 3⁵ menos 6 casos retenidos para la "
    "validación H1a). Regla de decisión v2 calibrada en los 6 casos "
    "de calibración. Validación cruzada 5-fold: F1 macro 0,936 ± "
    "0,036. Explicación: TreeExplainer de SHAP con 50 ejemplos de "
    "fondo, traducida a lenguaje natural con Gemini."
)
