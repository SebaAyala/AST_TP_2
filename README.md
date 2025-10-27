# AST_TP_2
TP 2 de la materia Análisis de Series Temporales 2025

## Descripción

Este proyecto implementa un análisis completo de series temporales para tres series diferentes. El código incluye:

- Estadísticas descriptivas
- Visualización de series temporales
- Tests de estacionariedad (Augmented Dickey-Fuller)
- Análisis de autocorrelación (ACF y PACF)
- Análisis de distribuciones (histogramas y Q-Q plots)

## Estructura del Proyecto

```
AST_TP_2/
├── README.md                      # Este archivo
├── requirements.txt               # Dependencias de Python
├── time_series_analysis.py        # Script principal de análisis
├── time_series_analysis.ipynb     # Notebook Jupyter para análisis interactivo
└── example.py                     # Ejemplo simple de uso
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/SebaAyala/AST_TP_2.git
cd AST_TP_2
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Opción 1: Ejecutar el ejemplo simple

Para un inicio rápido:

```bash
python example.py
```

### Opción 2: Ejecutar el script de Python completo

Para ejecutar el análisis completo con datos de ejemplo:

```bash
python time_series_analysis.py
```

Este script generará:
- Estadísticas descriptivas en la consola
- Gráficos de las tres series temporales
- Resultados de tests de estacionariedad
- Gráficos de ACF y PACF
- Análisis de distribuciones
- Un archivo CSV con los datos: `time_series_data.csv`

### Opción 3: Usar el Notebook Jupyter

Para un análisis más interactivo:

```bash
jupyter notebook time_series_analysis.ipynb
```

El notebook permite:
- Ejecutar el análisis paso a paso
- Modificar parámetros fácilmente
- Agregar análisis adicionales
- Documentar tus hallazgos

### Opción 4: Usar con tus propios datos

Puedes usar la clase `TimeSeriesAnalysis` con tus propios datos:

```python
from time_series_analysis import TimeSeriesAnalysis

# Crear objeto de análisis
tsa = TimeSeriesAnalysis(series_names=['Serie A', 'Serie B', 'Serie C'])

# Cargar datos desde un archivo CSV
tsa.load_data('mis_datos.csv', column_names=['col1', 'col2', 'col3'])

# Ejecutar análisis completo
tsa.run_complete_analysis()
```

## Características del Análisis

### 1. Estadísticas Descriptivas
- Media, mediana, desviación estándar
- Valores mínimo y máximo
- Cuartiles
- Asimetría (skewness) y curtosis

### 2. Tests de Estacionariedad
- Augmented Dickey-Fuller (ADF) test
- Interpretación automática de resultados

### 3. Análisis de Autocorrelación
- Función de autocorrelación (ACF)
- Función de autocorrelación parcial (PACF)

### 4. Análisis de Distribuciones
- Histogramas
- Q-Q plots para verificar normalidad

## Datos de Ejemplo

El script genera tres series temporales de ejemplo:

1. **Serie 1**: Tendencia lineal + estacionalidad + ruido
2. **Serie 2**: Proceso autoregresivo AR(1)
3. **Serie 3**: Paseo aleatorio (random walk)

Estas series están diseñadas para demostrar diferentes propiedades y comportamientos de series temporales.

## Requisitos

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Statsmodels
- SciPy

Ver `requirements.txt` para versiones específicas.

## Autor

Sebastián Ayala

## Licencia

Este proyecto es para uso académico en el contexto del curso Análisis de Series Temporales 2025.
