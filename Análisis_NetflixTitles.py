# URL DEL ARCHIVO CSV: https://www.kaggle.com/datasets/shivamb/netflix-shows

# INSTALACIÓN DE PAQUETES REQUERIDOS

# Ejecutar en CMD: pip install kagglehub

# IMPORTACIÓN DE LIBRERÍAS

import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# RUTA LOCAL DEL ARCHIVO CSV

# local_file_path = ''
online_file_path = kagglehub.dataset_download("shivamb/netflix-shows")

# CARGA DEL DATAFRAME

# df = pd.read_csv(local_file_path)
df = pd.read_csv(f"{online_file_path}/netflix_titles.csv")

# CONFIGURACIÓN DE VISUALIZACIÓN

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# SEPARADOR DE SECCIONES

def print_separator():
    print('\n#-------------------------------------#\n')

# TEXTO EN COLOR NEGRO

def print_black(text):
    print("\033[1m" + text + "\033[0m")

# TEXTO EN COLOR MORADO

def print_purple(text):
    print("\033[95m" + text + "\033[0m")

# EXPLORACIÓN INICIAL DEL DATAFRAME

print_separator()
print_black("EXPLORACIÓN INICIAL DEL DATAFRAME")
print_separator()

print_purple('Columnas del DataFrame.\n')
print(list(df.columns))

print_purple('\nPrimeras 5 Filas Del DataFrame.\n')
print(df.head())                                         

print_purple('\nÚltimas 5 Filas Del DataFrame.\n')
print(df.tail())                                        

print_purple('\nMuestra Aleatoria de 5 Filas Del DataFrame.\n')
print(df.sample(5))                                                                                       

print_purple('\nTamaño Del DataFrame.\n')
print(df.shape)

print_purple('\nInformación General Del DataFrame.\n')
df.info()

# LIMPIEZA DE DATOS

print_separator()
print_black("LIMPIEZA DE DATOS")
print_separator()

# LIMPIEZA DE DATOS - DETECCIÓN DE VALORES FALTANTES

# VALORES FALTANTES POR COLUMNA

print_purple('Valores Faltantes Por Columna.\n')
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0],'\n')

# PORCENTAJE DE VALORES FALTANTES POR COLUMNA

print_purple('Porcentaje De Valores Faltantes Por Columna.\n')                                                                           
missing_percentage = (df.isnull().sum() / len(df)) * 100                                                            
print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

# Sustituir de valores NaN con "Sin Información" en las columnas 'director', 'cast', 'date_added', 'rating' y 'duration'.

for col in ['director', 'cast', 'country', 'date_added', 'rating', 'duration']:
    df[col] = df[col].fillna("Sin Información")

# LIMPIEZA DE DATOS - GESTIÓN DE VALORES DUPLICADOS

# NÚMERO DE FILAS DUPLICADAS

num_duplicates = df.duplicated().sum()
print_purple('\nNúmero De Filas Duplicadas.\n')                                                                                                              
print(f"Filas Duplicadas Encontradas: {num_duplicates}")                                              

# LIMPIEZA DE DATOS - CORRECCIÓN DE TIPOS DE DATOS

# TIPOS DE DATOS ORIGINALES

print_purple('\nTipos De Datos Originales.\n')
print(df.dtypes)

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# TIPOS DE DATOS CORREGIDOS

print_purple('\nTipos De Datos Corregidos.\n')
print(df.dtypes)

# LIMPIEZA DE DATOS - CREACIÓN DE NUEVAS COLUMNAS

print_purple('\nCreación De Nuevas Columnas (Minutos Y Temporadas).\n')
df['minutes'] = df['duration'].str.extract(r'(\d+) min').astype('Int64')
df['seasons'] = df['duration'].str.extract(r'(\d+) [Ss]eason').astype('Int64')

print(df[['duration', 'minutes', 'seasons']].head())

# LIMPIEZA DE DATOS - CORRECCIÓN DE VALORES EN COLUMNA RATING

print_purple('\nCorrección De Valores En Columna Rating.\n')
print("Filas Sucias:\n")
filas_sucias = df[df['rating'].str.contains('min', na=False)]
print(filas_sucias[['rating', 'duration']])

df.loc[df['rating'].str.contains('min', na=False), 'duration'] = df['rating']
df.loc[df['rating'].str.contains('min', na=False), 'rating'] = 'Sin Información'

# Imprimir las filas sucias ahora corregidas en base a los indices de filas_sucias y sólo las columnas 'rating' y 'duration'

print('\nFilas Sucias Corregidas:\n')
print(df.loc[filas_sucias.index, ['rating', 'duration']])

# ANÁLISIS EXPLORATORIO DE DATOS (EDA)

print_separator()
print_black("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print_separator()

# PREGUNTA 1: ¿Cuántas películas (Movie) y cuántas series (TV Show) hay en el dataset? Muestra los porcentajes.

print_purple('Pregunta 1: ¿Cuántas películas (Movie) y cuántas series (TV Show) hay en el dataset? Muestra los porcentajes.\n')

# FILTRADO

conteo_peliculas_series = df['type'].value_counts()
porcentajes = conteo_peliculas_series / conteo_peliculas_series.sum() * 100
print(f"Conteo:\n\n{conteo_peliculas_series}\n")
print(f"Porcentajes:\n\n{porcentajes}")

# GRÁFICO

colors = ['#1f77b4', '#2ca02c']
sns.set_theme(style="white", font_scale=1.1)
plt.figure(figsize=(5, 5))

ax = sns.barplot(
    x=conteo_peliculas_series.index.str.upper(),
    y=conteo_peliculas_series.values,
    hue=conteo_peliculas_series.index,
    palette=colors,
    saturation=0.95,
    legend=False
)

for i, valor in enumerate(conteo_peliculas_series.values):
    porcentaje_actual = porcentajes.iloc[i]
    
    texto_etiqueta = f"{valor}\n({porcentaje_actual:.1f}%)"
    
    ax.text(
        i,
        valor + (valor * 0.02),
        texto_etiqueta,
        ha='center',
        va='bottom',
        fontsize=13,
        fontweight='bold',
        color='#333333'
    )

sns.despine(
    left=True, 
    bottom=False
)

plt.title(
    "DISTRIBUCIÓN DE CONTENIDO: PELÍCULAS VS SERIES DE TV",
    fontsize=22,
    fontweight='heavy',
    loc='center',
    pad=30
)

plt.xlabel(
    "TIPO DE CONTENIDO", 
    fontsize=13, 
    labelpad=15, 
    fontweight='bold', 
    color='gray'
)

plt.ylabel("")
plt.yticks([])

plt.ylim(
    0, 
    conteo_peliculas_series.values.max() * 1.25
)

plt.tight_layout()
plt.show()

# PREGUNTA 2: Top Países: ¿Cuáles son los 5 países que más contenido producen en Netflix? (Nota: ten cuidado con los campos que tienen múltiples países separados por comas).

print_purple('\nPregunta 2: Top Países: ¿Cuáles son los 5 países que más contenido producen en Netflix? (Nota: ten cuidado con los campos que tienen múltiples países separados por comas).\n')

# FILTRADO

top_countries = df['country'].str.split(',').explode().str.strip().value_counts().drop('Sin Información', errors='ignore').head(5)
print(top_countries)

# GRÁFICO

sns.set_theme(style="white", font_scale=1.1)
plt.figure(figsize=(10, 6))

colores_top = ['#E50914'] + ['#B0B0B0'] * (len(top_countries) - 1)

ax = sns.barplot(
    x=top_countries.index.str.upper(),
    y=top_countries.values,
    hue=top_countries.index,
    palette=colores_top,
    legend=False
)

for i, valor in enumerate(top_countries.values):
    color_texto = '#E50914' if i == 0 else '#555555'
    
    ax.text(
        i, 
        valor + (valor * 0.03), 
        f"{valor:,}",         
        ha='center', 
        va='bottom', 
        fontsize=13,            
        fontweight='bold',
        color=color_texto
    )

sns.despine(
    left=True,
    bottom=False
)

plt.title(
    "TOP 5 PAÍSES QUE MÁS CONTENIDO PRODUCEN EN NETFLIX",
    fontsize=22,
    fontweight='heavy',
    loc='center',
    pad=25
)

plt.xlabel("")
plt.ylabel("") 
plt.yticks([])

plt.ylim(
    0, 
    top_countries.values.max() * 1.25
)

plt.tight_layout()
plt.show()

# PREGUNTA 3: Clasificación por edades: ¿Cuál es la clasificación (rating) más común en todo el catálogo?

print_purple('\nPregunta 3: Clasificación por edades: ¿Cuál es la clasificación (rating) más común en todo el catálogo?\n')

# FILTRADO

df = df[df['rating'] != 'Sin Información']
conteo_rating = df['rating'].value_counts()
print(conteo_rating)

sns.set_theme(style="white", font_scale=1.1)
plt.figure(figsize=(10, 8)) 

colores_rating = ['#E50914'] + ['#D6D6D6'] * (len(conteo_rating) - 1)

ax = sns.barplot(
    y=conteo_rating.index, 
    x=conteo_rating.values,
    hue=conteo_rating.index,
    palette=colores_rating,
    legend=False
)

for i, valor in enumerate(conteo_rating.values):
    color_texto = '#E50914' if i == 0 else '#555555'
    
    ax.text(
        valor + (conteo_rating.values.max() * 0.015), 
        i, 
        f"{valor:,}", 
        ha='left',    
        va='center',  
        fontsize=12,            
        fontweight='bold',
        color=color_texto
    )

sns.despine(
    left=True, 
    bottom=True
)

plt.title(
    "CLASIFICACIÓN POR EDAD MÁS COMÚN EN NETFLIX",
    fontsize=22,
    fontweight='heavy',
    loc='left',
    pad=25
)

plt.xlabel("") 
plt.ylabel("") 
plt.xticks([])

plt.xlim(
    0, 
    conteo_rating.values.max() * 1.15
)

for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_color('#333333')

plt.tight_layout()
plt.show()

# PREGUNTA 4: Directores prolíficos: ¿Quién es el director con más títulos en la plataforma? ¿Cuáles son los 10 directores con más títulos en la plataforma?

print_purple('\nPregunta 4: Directores prolíficos: ¿Quién es el director con más títulos en la plataforma? ¿Cuáles son los 10 directores con más títulos en la plataforma?\n')

# FILTRADO

df = df[df['director'] != 'Sin Información']
conteo_directores = df['director'].value_counts()
print(conteo_directores.head(10))

top_directores = conteo_directores.head(10)

# GRÁFICO

sns.set_theme(style="white", font_scale=1.1)
plt.figure(figsize=(10, 7))

colores_directores = ['#E50914'] + ['#D6D6D6'] * (len(top_directores) - 1)

ax = sns.barplot(
    y=top_directores.index, 
    x=top_directores.values,
    palette=colores_directores,
    hue=top_directores.index,
    legend=False
)

for i, valor in enumerate(top_directores.values):
    color_texto = '#E50914' if i == 0 else '#555555'
    
    margen = top_directores.values.max() * 0.02
    
    ax.text(
        valor + margen, 
        i, 
        str(valor), 
        ha='left', 
        va='center', 
        fontsize=12,            
        fontweight='bold',
        color=color_texto
    )

sns.despine(left=True, bottom=True)

plt.title(
    "TOP 10 DIRECTORES CON MÁS TÍTULOS EN LA PLATAFORMA",
    fontsize=22,
    fontweight='heavy',
    loc='left',
    pad=20
)

plt.xlabel("") 
plt.ylabel("") 
plt.xticks([]) 

plt.xlim(
    0, 
    top_directores.values.max() * 1.15
)

for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_color('#333333')

plt.tight_layout()
plt.show()

# PREGUNTA 5: Desglose de géneros: La columna listed_in contiene géneros separados por comas. Encuentra cuántos títulos hay por cada género individual.

print_purple('\nPregunta 5: Desglose de géneros: La columna listed_in contiene géneros separados por comas. Encuentra cuántos títulos hay por cada género individual.\n')

# FILTRADO

conteo_generos = df['listed_in'].str.split(',').explode().str.strip().value_counts().drop('Sin Información', errors='ignore').head(10)
print(conteo_generos)

# GRÁFICO PREGUNTA 5

sns.set_theme(style="white", font_scale=1.1)
plt.figure(figsize=(11, 7)) 

colores_generos = ['#E50914'] + ['#D6D6D6'] * (len(conteo_generos) - 1)

ax = sns.barplot(
    y=conteo_generos.index, 
    x=conteo_generos.values,
    palette=colores_generos,
    hue=conteo_generos.index,
    legend=False
)

for i, valor in enumerate(conteo_generos.values):
    color_texto = '#E50914' if i == 0 else '#555555'
    
    ax.annotate(
        f"{valor:,}",           
        xy=(valor, i),          
        xytext=(10, 0),         
        textcoords="offset points", 
        ha='left', 
        va='center', 
        fontsize=12,            
        fontweight='bold',
        color=color_texto
    )

sns.despine(
    left=True, 
    bottom=True
)

plt.title(
    "TOP 10 GÉNEROS CON MÁS TÍTULOS EN NETFLIX",
    fontsize=22,
    fontweight='heavy',
    loc='left',
    pad=25
)

plt.xlabel("") 
plt.ylabel("") 
plt.xticks([]) 

plt.xlim(
    0, 
    conteo_generos.values.max() * 1.20
)

for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_color('#333333')

plt.tight_layout()
plt.show()

# PREGUNTA 6: ¿Quién es el actor o actriz que aparece en más películas/series de Netflix? ¿Cuáles son los 20 actores/actrices con más títulos en la plataforma?

print_purple('\nPregunta 6: ¿Quién es el actor o actriz que aparece en más filmerias/series de Netflix? ¿Cuáles son los 20 actores/actrices con más títulos en la plataforma?\n')

# FILTRADO

df = df[df['cast'] != 'Sin Información']
conteo_cast = df['cast'].str.split(',').explode().str.strip().value_counts().drop('Sin Información', errors='ignore').head(20)
print(conteo_cast)

top_cast = conteo_cast.head(20)

# GRÁFICO

sns.set_theme(style="white", font_scale=1.1)
plt.figure(figsize=(12, 11)) 

colores_cast = ['#E50914'] + ['#D6D6D6'] * (len(conteo_cast) - 1)

ax = sns.barplot(
    y=conteo_cast.index, 
    x=conteo_cast.values,
    palette=colores_cast,
    hue=conteo_cast.index,
    legend=False
)

for i, valor in enumerate(conteo_cast.values):
    color_texto = '#E50914' if i == 0 else '#555555'
    
    ax.annotate(
        f"{valor:,}",           
        xy=(valor, i),          
        xytext=(10, 0),         
        textcoords="offset points", 
        ha='left', 
        va='center', 
        fontsize=12,            
        fontweight='bold',
        color=color_texto
    )

sns.despine(
    left=True, 
    bottom=True
)

plt.title(
    "TOP 20 ACTORES/ACTRICES CON MÁS APARICIONES EN TÍTULOS DE NETFLIX",
    fontsize=22,
    fontweight='heavy',
    loc='left',
    pad=25
)

plt.xlabel("") 
plt.ylabel("") 
plt.xticks([]) 

plt.xlim(
    0, 
    conteo_cast.values.max() * 1.15
)

for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_color('#333333')

plt.tight_layout()
plt.show()

# PREGUNTA 7: Búsqueda por texto: Filtra y muestra todas las películas que tengan la palabra "Love" en su título.

print_purple('\nPregunta 7: Búsqueda por texto: Filtra y muestra todas las películas que tengan la palabra "Love" en su título.\n')

peliculas_love = df[(df['title'].str.contains('Love', case=False, na=False)) & (df['type'] == 'Movie')]

with pd.option_context('display.max_rows', None):
    print(peliculas_love[['title']])

# PREGUNTA 8: Evolución del contenido: Crea una gráfica de líneas que muestre cuántos títulos se han añadido a Netflix por año (usando la columna date_added ya convertida).

print_purple('\nPregunta 8: Evolución del contenido: Crea una gráfica de líneas que muestre cuántos títulos se han añadido a Netflix por año.\n')

# FILTRADO

contenido_anio = df.groupby(df['date_added'].dt.year)['title'].count()
contenido_anio.index = contenido_anio.index.astype(int)
print(contenido_anio)

# GRÁFICO

sns.set_theme(style="white", font_scale=1.1)
plt.figure(figsize=(12, 6))

plt.plot(
    contenido_anio.index, 
    contenido_anio.values, 
    color='#E50914',          
    linewidth=3.5,            
    marker='o',               
    markersize=8, 
    markerfacecolor='white',  
    markeredgecolor='#E50914',
    markeredgewidth=2
)

plt.fill_between(
    contenido_anio.index, 
    contenido_anio.values, 
    color='#E50914', 
    alpha=0.1
)

for x, y in zip(contenido_anio.index, contenido_anio.values):
    plt.text(
        x, 
        y + (contenido_anio.values.max() * 0.04),
        f"{y:,}",               
        ha='center', 
        va='bottom', 
        fontsize=12,            
        fontweight='bold',
        color='#333333'
    )

sns.despine(left=True)

plt.title(
    "EVOLUCIÓN DE TÍTULOS AÑADIDOS A NETFLIX POR AÑO",
    fontsize=22,
    fontweight='heavy',
    loc='center',
    pad=25
)

plt.xlabel(
    "AÑO", 
    fontsize=15, 
    labelpad=25, 
    fontweight='bold', 
    color='gray'
)

plt.ylabel("") 
plt.yticks([])

plt.xticks(
    contenido_anio.index, 
    color='#555555', 
    fontweight='bold'
)

plt.ylim(
    0, 
    contenido_anio.values.max() * 1.25
)

plt.tight_layout()
plt.show()

# PREGUNTA 9: Antigüedad del contenido: Calcula la diferencia de años entre el año de lanzamiento (release_year) y el año en que se añadió a Netflix. ¿Cuál es la película más antigua añadida recientemente?

print_purple('\nPregunta 9: Antigüedad del contenido: Calcula la diferencia de años entre el año de lanzamiento (release_year) y el año en que se añadió a Netflix. ¿Cuál es la filmería más antiguaañadida recientemente?\n')

df['antiguedad'] = df['release_year'] - df['date_added'].dt.year

print(df[df['antiguedad'] == df['antiguedad'].max()])

# PREGUNTA 10: Meses populares: ¿En qué mes se suele añadir más contenido a la plataforma?

print_purple('\nPregunta 10: Meses populares: ¿En qué mes se suele añadir más contenido a la plataforma?\n')

# FILTRADO

contenido_mes = df.groupby(df['date_added'].dt.month)['title'].count()
contenido_mes.index = contenido_mes.index.astype(int)
print(contenido_mes)

meses_espanol = {
    1: 'ENERO', 2: 'FEBRERO', 3: 'MARZO', 4: 'ABRIL',
    5: 'MAYO', 6: 'JUNIO', 7: 'JULIO', 8: 'AGOSTO',
    9: 'SEPTIEMBRE', 10: 'OCTUBRE', 11: 'NOVIEMBRE', 12: 'DICIEMBRE'
}

contenido_mes.index = contenido_mes.index.map(meses_espanol)

# GRÁFICO

sns.set_theme(style="white", font_scale=1.1)
plt.figure(figsize=(14, 6))

valor_maximo = contenido_mes.values.max()
colores_meses = ['#E50914' if val == valor_maximo else '#D6D6D6' for val in contenido_mes.values]

ax = sns.barplot(
    x=contenido_mes.index,
    y=contenido_mes.values,
    palette=colores_meses,
    hue=contenido_mes.index,
    legend=False
)

for i, valor in enumerate(contenido_mes.values):
    color_texto = '#E50914' if valor == valor_maximo else '#555555'
    
    ax.text(
        i, 
        valor + (valor_maximo * 0.02),
        f"{valor:,}", 
        ha='center', 
        va='bottom', 
        fontsize=15,            
        fontweight='bold',
        color=color_texto
    )

sns.despine(left=True)

plt.title(
    "MESES CON MAYOR VOLUMEN DE ESTRENOS EN NETFLIX",
    fontsize=22,
    fontweight='heavy',
    loc='center',
    pad=25
)

plt.xlabel("") 
plt.ylabel("") 
plt.yticks([]) 

for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')
    tick.set_color('#555555')
    tick.set_fontsize(10)

plt.ylim(
    0, 
    valor_maximo * 1.25
)

plt.tight_layout()
plt.show()

# PREGUNTA 11: Mapa de calor: Crea un heatmap que muestre la cantidad de contenido añadido según el "Mes" vs el "Tipo" (Película o Serie).

# FILTRADO

contenido_pelicula_mes = df[df['type'] == 'Movie']['date_added'].dt.month.value_counts().sort_index()
contenido_serie_mes = df[df['type'] == 'TV Show']['date_added'].dt.month.value_counts().sort_index()

df_heatmap = pd.DataFrame({
    'PELÍCULAS': contenido_pelicula_mes,
    'SERIES': contenido_serie_mes,
})

df_heatmap.index = df_heatmap.index.astype(int)
df_heatmap.index = df_heatmap.index.map(meses_espanol)

# GRÁFICO

sns.set_theme(style="white")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 9), gridspec_kw={'wspace': 0.03})

cmap_pelis = LinearSegmentedColormap.from_list("rojo_netflix", ["#f7f7f7", "#E50914"])
cmap_series = LinearSegmentedColormap.from_list("azul_series", ["#f7f7f7", "#2b5c8f"]) 

sns.heatmap(
    df_heatmap[['PELÍCULAS']],
    annot=True, fmt=",d", cmap=cmap_pelis,
    linewidths=2, linecolor='white', ax=ax1,
    cbar_kws={"orientation": "horizontal", "pad": 0.06, "shrink": 0.8}
)

sns.heatmap(
    df_heatmap[['SERIES']],
    annot=True, fmt=",d", cmap=cmap_series,
    linewidths=2, linecolor='white', ax=ax2,
    cbar_kws={"orientation": "horizontal", "pad": 0.06, "label": "VOLUMEN SERIES", "shrink": 0.8}
)

for ax in [ax1, ax2]:
    ax.xaxis.tick_top() 
    ax.tick_params(axis='both', which='both', length=0) 
    ax.set_ylabel("") 
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, fontweight='bold', color='#333333')

ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=12, fontweight='bold', color='#555555', rotation=0)

ax2.set_yticks([]) 

cbar1 = ax1.collections[0].colorbar
cbar1.set_label("VOLUMEN DE PELÍCULAS", labelpad=15, fontweight='bold', color='#E50914')

cbar2 = ax2.collections[0].colorbar
cbar2.set_label("VOLUMEN DE SERIES", labelpad=15, fontweight='bold', color='#2b5c8f')

fig.suptitle(
    "DENSIDAD DE CONTENIDO AÑADIDO SEGÚN EL MES Y EL TIPO DE CONTENIDO",
    fontsize=22, fontweight='heavy', ha='center'
)

plt.show()

# PREGUNTA 12: Distribución de duración: Haz un histograma de la duración de las películas (en minutos).

# FILTRADO

duraciones = df[df['type'] == 'Movie']['minutes']
valor_maximo = duraciones.max()

# GRÁFICO

sns.set_theme(style="white")
plt.figure(figsize=(12, 6))

ax = sns.histplot(
    duraciones,
    binwidth=5,               
    kde=True,                 
    color='#D6D6D6',          
    edgecolor='white',       
    alpha=1
)

ax.lines[0].set_color('#E50914')
ax.lines[0].set_linewidth(3.5)

plt.xticks(
    ticks=range(0, int(valor_maximo) + 10, 10),
    fontsize=12,
    fontweight='bold',
    color='#555555'
)

plt.yticks(fontsize=11, color='#888888')

ax.yaxis.grid(True, color='#eeeeee')
ax.set_axisbelow(True) 
sns.despine(left=True)

plt.title(
    "DISTRIBUCIÓN DE DURACIÓN DE LAS PELÍCULAS",
    fontsize=22,
    fontweight='heavy',
    loc='center',
    pad=25
)

plt.xlabel(
    "DURACIÓN [MINUTOS]", 
    fontsize=13, 
    labelpad=15, 
    fontweight='bold', 
    color='#555555'
)

plt.ylabel(
    "CANTIDAD DE PELÍCULAS", 
    fontsize=13, 
    labelpad=15, 
    fontweight='bold', 
    color='#555555'
)

plt.xlim(
    0, 
    valor_maximo + 10
)

plt.tight_layout()
plt.show()
