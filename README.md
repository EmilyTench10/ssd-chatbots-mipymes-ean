# Sistema de soporte a la decisión para arquitecturas de chatbot en mipymes

Prototipo del trabajo de grado de **Maestría en Ciencia de Datos** y
**Maestría en Inteligencia de Negocios** de la Universidad Ean (Bogotá,
Colombia).

**Autoras:** Laura Ximena Agudelo Bastidas, Andrea Carolina Conde Avila,
Emily Del Carmen Tench Pérez.
**Directora:** Carolina María Luque Zabala.

## Descripción

El sistema recomienda una de tres arquitecturas de chatbot
(**RAG**, **Fine-tuning** o **Híbrida**) para una micro, pequeña o
mediana empresa (mipyme) del sector servicios, a partir de cinco
preguntas sobre su contexto operativo: volumen de datos, volatilidad
de la información, presupuesto, tiempo de desarrollo y riesgo
operativo por alucinación.

La recomendación va acompañada de una **justificación en lenguaje
natural** generada a partir de valores SHAP del clasificador
supervisado.

## Cómo correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para que la explicación en lenguaje natural funcione localmente,
copiar `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml`
y pegar tu API key de Gemini.

## Cómo desplegar en Streamlit Community Cloud (gratis)

1. Subir este directorio como repositorio **público** en GitHub.
2. Ir a [share.streamlit.io](https://share.streamlit.io) e iniciar
   sesión con GitHub.
3. Clic en **New app** → seleccionar el repositorio, branch `main`,
   archivo principal `app.py`.
4. En **Advanced settings → Secrets**, pegar:
   ```toml
   LLM_API_KEY = "tu-api-key-de-gemini"
   LLM_MODELO = "gemini-2.0-flash-exp"
   ```
5. Clic en **Deploy** (toma 2-5 minutos).

## Estructura

```
app.py                        Aplicación Streamlit
requirements.txt              Dependencias Python
modelo/
    modelo_v2.pkl             Clasificador Random Forest entrenado
                              con la regla v2 calibrada
datos/
    perfiles_243_regla_v2.csv Perfiles del diseño factorial para
                              el fondo de SHAP
.streamlit/
    config.toml               Tema visual
    secrets.toml.example      Plantilla de secretos (no subir el real)
```

## Modelo

- **Clasificador:** Random Forest, 200 estimadores, semilla 2026.
- **Entrenamiento:** 237 perfiles del diseño factorial 3⁵ (243 totales
  menos 6 retenidos para la validación de H1a).
- **Etiquetas:** generadas por la **regla v2 calibrada** con los 6
  casos de calibración del experimento.
- **Validación cruzada 5-fold:** F1 macro 0,936 ± 0,036; exactitud
  0,937 ± 0,036.
- **Clases presentes:** RAG, Híbrida. La regla v2 no asigna
  Fine-tuning puro a ningún perfil (restricción R-e documentada en
  la tesis).

## Licencia

Trabajo académico. El código se puede reutilizar con atribución.
