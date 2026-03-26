# 🌤️ Sistema de Predicción Meteorológica

Sistema completo para predecir valores en variables meteorológicas y detectar automáticamente eventos extremos como olas de frío y calor.

El sistema se adapta automáticamente a las características de cada variable para generar las mejores predicciones.

---

## ¿Qué hace este sistema?

Este sistema toma datos históricos de estaciones meteorológicas y genera:

1. **Predicciones futuras** de la variable meteorológica según los datos de entrada.
2. **Alertas automáticas** cuando se detectan valores extremos (según la configuración de los umbrales).
3. **Gráficos visuales** que presentan las predicciones y alertas.

---

## Requisitos Previos

Antes de instalar, necesitas tener:

- **Python 3.8 o superior** instalado en tu computadora.
- **Datos meteorológicos** en formato Excel (`.xlsx`) con el nombre de la variable que quieres procesar (por ejemplo: `temp_max.xlsx` o `temp_min.xlsx`).

---

## Instalación paso a paso

### Paso 1: Clonar el repositorio

Abre una terminal y clona el proyecto desde Git:

```bash
git clone [url-del-repositorio]
cd weather_new
```

### Paso 2: Instalar las dependencias

```bash
pip install -r requirements.txt
```

Esto instalará todas las librerías necesarias. Puede tardar unos minutos.

### Paso 3: Colocar tus datos

Coloca tus archivos Excel en la carpeta `data/input/`. El nombre del archivo debe coincidir con el nombre de la variable que quieres procesar:

```
weather_new/
└── data/
    └── input/
        └── [variable].xlsx
```

**Importante:** Si vas a usar el sistema de alertas, también necesitas el archivo de umbrales correspondiente en `data/thresholds/`.

---

## Uso básico

### Ejemplos prácticos

El sistema puede procesar cualquier variable meteorológica. Aquí tienes ejemplos concretos para las más comunes:

#### Ejemplo 1: predicción básica

**Para temperatura máxima:**
```bash
python bin/weather_predict.py temp_max
```

**Para temperatura mínima:**
```bash
python bin/weather_predict.py temp_min
```

Esto procesará todas las estaciones y generará predicciones para las próximas 4 semanas (por defecto).

#### Ejemplo 2: predicción con límite de estaciones

Si tienes muchas estaciones y quieres probar con pocas primero:

**Para temperatura máxima:**
```bash
python bin/weather_predict.py temp_max --max-stations 5
```

**Para temperatura mínima:**
```bash
python bin/weather_predict.py temp_min --max-stations 5
```

Esto procesará solo las primeras 5 estaciones.

#### Ejemplo 3: cambiar el horizonte de predicción

Para predecir más o menos semanas en el futuro:

**Para temperatura máxima:**
```bash
# Predecir 6 semanas
python bin/weather_predict.py temp_max --horizon-weeks 6
```

```bash
# O predecir días específicos
python bin/weather_predict.py temp_max --horizon-days 30
```

**Para temperatura mínima:**
```bash
# Predecir 6 semanas
python bin/weather_predict.py temp_min --horizon-weeks 6
```

```bash
# O predecir días específicos
python bin/weather_predict.py temp_min --horizon-days 30
```

---

## Opciones del comando

Puedes personalizar el comportamiento del sistema con estas opciones:

| Opción | Descripción | Ejemplo |
|--------|-------------|---------|
| `--max-stations N` | Procesar solo N estaciones | `--max-stations 5` |
| `--horizon-weeks N` | Predecir N semanas | `--horizon-weeks 6` |
| `--horizon-days N` | Predecir N días | `--horizon-days 30` |
| `--stage [etapa]` | Ejecutar solo una etapa | `--stage prediction` |
| `--no-plots` | No generar gráficos | `--no-plots` |
| `--verbose` | Mostrar más información | `--verbose` |

### Etapas del proceso

El sistema tiene 3 etapas que se ejecutan automáticamente:

1. **Preprocessing** - Limpia y organiza los datos.
2. **Imputation** - Completa valores faltantes.
3. **Prediction** - Genera las predicciones.

Por defecto, se ejecutan las 3 etapas. Si ya tienes datos procesados, puedes ejecutar solo la predicción:

**Para temperatura máxima:**
```bash
python bin/weather_predict.py temp_max --stage prediction
```

**Para temperatura mínima:**
```bash
python bin/weather_predict.py temp_min --stage prediction
```

---

## Dónde encontrar los resultados

Después de ejecutar el sistema, encontrarás los resultados en la carpeta `output/`:

```
output/
├── preprocessing/          # Datos limpios
│   └── [variable]/
│       └── [variable].csv
│
├── imputation/             # Datos completados
│   └── [variable]/
│       ├── csv_files/      # Archivos por estación
│       ├── plots/          # Gráficos de comparación
│       └── statistics_reports/  # Reportes estadísticos
│
└── prediction/              # Predicciones 
    └── [variable]/
        ├── csv_files/      # Predicciones en CSV
        ├── plots/          # Gráficos con predicciones
        ├── models/         # Modelos entrenados
        └── alerts/         # Alertas detectadas 
            ├── [estacion]_alerts.json
            ├── [estacion]_alerts.csv
            └── [estacion]_alerts_summary.json
```

### Archivos importantes

- **CSV de predicciones:** `output/prediction/temp_max/csv_files/[estacion]_predictions.csv` o `output/prediction/temp_min/csv_files/[estacion]_predictions.csv`
- **Gráficos:** `output/prediction/temp_max/plots/[estacion]_prediction.png` o `output/prediction/temp_min/plots/[estacion]_prediction.png`
- **Alertas:** `output/prediction/temp_max/alerts/[estacion]/[estacion]_alerts.json` o `output/prediction/temp_min/alerts/[estacion]/[estacion]_alerts.json`

---

## Sistema de Alertas

El sistema puede detectar automáticamente valores extremos cuando están configurados los umbrales correspondientes.

### Cómo funciona

El sistema compara las predicciones con umbrales históricos definidos por percentiles. Cuando una predicción supera estos umbrales, se genera una alerta clasificada por severidad (CRITICAL o WARNING).

### Dónde ver las Alertas

1. **En los gráficos:** aparecen como marcadores en los gráficos de predicción.
2. **En archivos JSON/CSV:** en la carpeta `output/prediction/temp_max/alerts/` o `output/prediction/temp_min/alerts/`.

### Requisito para Alertas

Necesitas el archivo de umbrales correspondiente en `data/thresholds/` con los umbrales históricos para tu variable.

Si no está presente, el sistema funcionará normalmente pero no generará alertas.

---

## Solución de Problemas comunes

### Error: "Excel file not found"

**Problema:** No encuentra el archivo de datos.

**Solución:**
1. Verifica que el archivo existe en `data/input/`.
2. Verifica que el nombre del archivo coincide .exactamente con el nombre de la variable que usas en el comando.
3. Verifica que estás ejecutando el comando desde la carpeta del proyecto.

### Error: "No imputed data found"

**Problema:** Intentas ejecutar solo predicción pero no hay datos imputados.

**Solución:**
Ejecuta todas las etapas:

**Para temperatura máxima:**
```bash
python bin/weather_predict.py temp_max
```

**Para temperatura mínima:**
```bash
python bin/weather_predict.py temp_min
```

O ejecuta primero imputación y luego predicción:

**Para temperatura máxima:**
```bash
python bin/weather_predict.py temp_max --stage imputation
python bin/weather_predict.py temp_max --stage prediction
```

**Para temperatura mínima:**
```bash
python bin/weather_predict.py temp_min --stage imputation
python bin/weather_predict.py temp_min --stage prediction
```

### El proceso es muy lento

**Solución:**
- Procesa menos estaciones: `--max-stations 5`.
- Desactiva gráficos temporalmente: `--no-plots`.

### No se generan alertas

**Posibles causas:**
1. No hay umbrales configurados (falta el archivo Excel de umbrales).
2. Las predicciones no superan los umbrales (esto es normal, significa que no hay eventos extremos).

**Solución:**
- Verifica que el archivo de umbrales está en `data/thresholds/`.
- Revisa los logs para ver si se detectaron alertas.
- Si no hay alertas, significa que los valores predichos están dentro de rangos normales.

---

## Configuración avanzada

El sistema tiene un archivo de configuración en `config/default.yaml` donde puedes ajustar parámetros técnicos. Por defecto, funciona bien sin modificar nada.

Si necesitas cambiar algo, edita ese archivo. Los cambios más comunes son:

- `default_horizon_weeks`: Semanas de predicción por defecto (actualmente 4).
- `max_stations`: Límite de estaciones (null = todas).
- `enable_plots`: Generar gráficos (true/false).

---

## Estructura del Proyecto

```
weather_new/
├── bin/
│   └── weather_predict.py      # Comando principal
├── config/
│   └── default.yaml             # Configuración
├── data/
│   ├── input/                   # Tus archivos Excel aquí
│   └── thresholds/              # Archivo de umbrales aquí
├── src/                         # Código fuente (no modificar)
├── notebooks/                   # Scripts internos
├── output/                      # Resultados aparecen aquí
├── requirements.txt             # Dependencias
└── README.md                    # Este archivo
```

---


## Preguntas Frecuentes

**P: ¿Cuánto tiempo tarda el proceso?**  
R: Depende del número de estaciones. Con 5 estaciones, aproximadamente 10-15 minutos. Con todas las estaciones, puede tardar varias horas.

**P: ¿Los modelos se guardan?**  
R: Sí, se guardan en `output/prediction/temp_max/models/` o `output/prediction/temp_min/models/` (según la variable) y se pueden reutilizar.

**P: ¿Qué pasa si hay datos faltantes?**  
R: El sistema los completa automáticamente usando métodos inteligentes de imputación.

**P: ¿Puedo cambiar los umbrales de alertas?**  
R: Sí, edita el archivo Excel `Umbrales_Olas de Frío y Calor.xlsx` en `data/thresholds/`

---

## Soporte

Si encuentras problemas o tienes preguntas:

1. Revisa la sección "Solución de Problemas" arriba
2. Revisa los logs en la carpeta `logs/`
3. Ejecuta con `--verbose` para ver más información:

   **Para temperatura máxima:**
   ```bash
   python bin/weather_predict.py temp_max --verbose
   ```

   **Para temperatura mínima:**
   ```bash
   python bin/weather_predict.py temp_min --verbose
   ```

---

## Resumen Rápido

1. **Instalar:** `pip install -r requirements.txt`
2. **Colocar datos:** Archivo Excel en `data/input/` (por ejemplo: `temp_max.xlsx` o `temp_min.xlsx`)
3. **Ejecutar:**

   **Para temperatura máxima:**
   ```bash
   python bin/weather_predict.py temp_max
   ```

   **Para temperatura mínima:**
   ```bash
   python bin/weather_predict.py temp_min
   ```

4. **Ver resultados:** Carpeta `output/prediction/temp_max/` o `output/prediction/temp_min/`

---

**Listo para usar!** 🌤️

