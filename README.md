# Proyecto IA — Predicción de riesgo de ENT asociadas a determinantes sociales (Cesar)

Universidad Popular del Cesar · Predicción de riesgo de enfermedades crónicas no transmisibles
asociadas a determinantes sociales en población dispersa.

Este repositorio reúne el análisis exploratorio (EDA) y la fase de diseño de modelos predictivos,
integrando tres fuentes: **RIPS** (atenciones), **NBI / CNPV-2018** y **proyecciones de población DANE**.

---

## Estructura de carpetas

```
onde lo estoy haciendo/
├── 01_datos_originales/     Fuentes crudas (no se modifican)
│   ├── Registros_..._RIPS_20260922.csv          (4.5 GB — RIPS nacional)
│   ├── CNPV-2018-NBI.xlsx                        (NBI por municipio)
│   └── DCD-area-proypoblacion-Mun-2020-2035-...xlsx  (proyecciones DANE)
│
├── 02_datos_procesados/     Datos derivados (generados por los notebooks)
│   ├── rips_cesar_ent_filtrado.csv    (RIPS filtrado: Cesar + CIE-10 ENT)
│   ├── agg_rips_municipio.csv         (atenciones ENT por municipio)
│   ├── agg_rips_municipio_anio.csv    (atenciones ENT por municipio × año)
│   ├── nbi_cesar_limpio.csv           (NBI limpio, 25 municipios)
│   ├── pop_cesar_2020.csv             (población municipal 2020)
│   └── dataset_final_cesar_ent.csv    (dataset integrado: 25 municipios × 35 variables)
│
├── 03_notebooks/            Entregables principales (ejecutados, con salidas)
│   ├── EDA_ENT_Cesar_RIPS_NBI_Poblacion.ipynb   (Fase 1: análisis exploratorio)
│   └── modelado_predictivo_ent_nbi.ipynb        (Fase 2: modelado y evaluación)
│
├── 04_scripts/              Código de apoyo (uso interno)
│   ├── build_model_nb.py            (genera el notebook de modelado)
│   └── modelado_predictivo_ent_nbi.py  (versión script del notebook de modelado)
│
├── 05_resultados/           Figuras exportadas para el informe
│   ├── figura1_histogramas_tasas.png
│   ├── figura2_boxplots_cabecera_cprd.png
│   ├── figura3_correlacion_nbi_ent.png
│   ├── figura4_prediccion_vs_observacion.png
│   └── figura5_matrices_confusion.png
│
└── 06_documentos/           Documentos de la asignatura / propuesta
    ├── Fase1_Propuesta.docx                  (documento técnico: Fase 1 + Fase 2, sección 10.2)
    ├── Fase1_Propuesta_backup.docx           (respaldo previo a la sección 10.2)
    ├── Unidad II - Machine Learning No supervisado, Validación y Métricas.pdf
    ├── Propuesta ... FASE 1.docx
    ├── Territorial_Health_Intelligence.pdf
    └── Fase 1 ... .rar
```

---

## Cómo ejecutar

Los notebooks **resuelven la raíz del proyecto automáticamente**, así que funcionan sin importar
desde dónde se abran o ejecuten:

```powershell
# Desde la raíz del proyecto
jupyter nbconvert --to notebook --execute --inplace "03_notebooks\EDA_ENT_Cesar_RIPS_NBI_Poblacion.ipynb"
jupyter nbconvert --to notebook --execute --inplace "03_notebooks\modelado_predictivo_ent_nbi.ipynb"
```

> **Nota:** el EDA solo re-ejecuta el filtrado por *streaming* del RIPS (≈2 min) si
> `02_datos_procesados/rips_cesar_ent_filtrado.csv` no existe. Si ya existe, se reutiliza.

### Requisitos (Python 3.12)
`pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl`, `scikit-learn`, `nbformat`, `nbconvert`.

---

## Decisiones metodológicas clave

- **Filtro RIPS**: solo departamento Cesar (`20`) y grupos CIE-10 `I10–I15`, `E10–E14`, `N18`, `J40–J44`.
- **Deduplicación:** se eliminan filas exactamente duplicadas del RIPS filtrado (afectan ~65 % de las
  atenciones). Documentado en el notebook de EDA.
- **Población base:** proyección municipal **2020** (`Área = Total`), por ser el año más cercano al rango
  de los RIPS (2009–2021).
- **Cruce:** 25/25 municipios cruzan las tres fuentes, sin nulos.
- **Variable objetivo:** `Tasa_Total_ENT_1000` (regresión) y sus terciles `Bajo` / `Medio` / `Alto`
  (9/8/8, para la variante de clasificación).
- **Outliers:** se conservan y se reportan (no se eliminan). Valledupar aparece como caso atípico
  (mayor tasa, menor NBI → sesgo de acceso/detección).

---

## Fase 2 — Modelado (resumen de resultados)

**Preprocesamiento.** Estandarización de predictores, **transformación logarítmica** de la tasa objetivo
(por su fuerte asimetría y el valor atípico de Valledupar), partición estratificada 70/15/15
(17/4/4 municipios) y **ingeniería de características**: brecha NBI rural disperso − cabecera,
brecha de miseria rural disperso − cabecera y `log(1 + población)`.

**Predictores (10).** 7 componentes totales del NBI + 2 brechas rural-urbana + `log(población)`.

**Modelos y ajuste de hiperparámetros** (búsqueda en rejilla con validación cruzada de 5 particiones):

| Tarea | Modelos | Hiperparámetro elegido |
|---|---|---|
| Regresión (principal) | Dummy · OLS · **Ridge** · Gradient Boosting | Ridge `alpha`=100; GBR `n`=50, `lr`=0,01, prof=2 |
| Clasificación (variante) | **Regresión Logística** · Gradient Boosting | LR `C`=100; GBC `n`=100, `lr`=0,01, prof=3 |

**Resultados de regresión (CV, escala log):** el OLS generaliza peor que la media (R² = −1,56) por la
colinealidad; la **regularización Ridge** corrige el problema (R² = 0,04; RMSE = 0,44) y el Gradient
Boosting queda muy cerca (R² = 0,04).

**Resultados de clasificación (CV):** Gradient Boosting 0,44 de exactitud y 0,42 de F1 macro;
Regresión Logística 0,32/0,29. Con tres clases el azar es 0,33, de modo que solo el Gradient Boosting
supera levemente el azar.

---

## Advertencia sobre el modelado

Con solo **25 municipios**, las métricas de un split fijo son muy inestables. El notebook de modelado
prioriza la **validación cruzada** y reporta resultados **exploratorios**, no una capacidad predictiva
consolidada.

Hallazgo clave: la relación entre NBI y tasa de atenciones es **negativa** (a mayor NBI, menos atenciones
registradas). Es coherente con un **sesgo de detección/acceso** —donde hay menos servicios hay menos
diagnóstico y registro—, no con menor enfermedad real. La tasa mide en parte **oferta y acceso**, por lo
que el modelo es una herramienta exploratoria que debe leerse junto con el análisis epidemiológico.

El detalle completo de la Fase 2 (preprocesamiento, modelos, hiperparámetros, resultados y limitaciones)
está en la sección **10.2** del documento `06_documentos/Fase1_Propuesta.docx`.
