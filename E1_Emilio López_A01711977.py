import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "college_sleep_and_gpa.csv"
REFERENCE_PATH = "study_cohort_reference.csv"

df = pd.read_csv(DATA_PATH)
ref = pd.read_csv(REFERENCE_PATH)

# Verificamos que los datos se hayan cargado de manera correcta
print("------- Data set principal: -------")
print(df.head())
print("------- Shape (filas, columnas): -------", df.shape)
print("------- Data set de referencias: -------")
print(ref.head())

# Revisamos los tipos de datos que guarda el data set y obtenemos datos relevantes
print("------- Información del data set -------")
print(df.info())
print("------- Estadísticas descriptivas (numéricas): -------")
print(df.describe())

# Revisamos valores duplicados por student_id
print("------- Valores Duplicados -------")
print("Duplicados por student_id:", df["student_id"].duplicated().sum())

# Revisamos cuales son los valores duplicados
print(df[df["student_id"].duplicated(keep=False)].sort_values("student_id"))

# Creamos una nueva columna juntanto cohort_code y student_id. ya que los 
# valores duplicados son estudiantes con mismo id pero de diferentes 
# universidades.
df["student_uid"] = df["cohort_code"] + "_" + df["student_id"].astype(str)
print("Duplicados por student_uid: ", df["student_uid"].duplicated().sum())

# Revisamos valores nulos
print("------- Valores Nulos -------")
nulos = pd.concat([df.isnull().sum(), (df.isnull().mean() * 100).round(2)], axis=1)
nulos.columns = ["cantidad", "porcentaje"]
print(nulos)

# Eliminamos valores nulos que no representen un porcentaje alto.
cols_nulos = ["gender", "first_generation", "underrepresented"]
df = df.dropna(subset=cols_nulos)

# Como aun no tenemos claro si estas columnas pueden ser relevantes
# vamos a aplicar 2 estrategias, una en donde tendremos un data set
# con los datos imputados con la mediana y otro con las columnas elimindas.
cols_imputar = ["term_units", "term_load_z"]

df_imputado = df.copy()
for col in cols_imputar:
    df_imputado[col] = df_imputado[col].fillna(df_imputado[col].median())

df_eliminado = df.dropna(subset=cols_imputar)

print("Filas originales:", df.shape[0])
print("Filas df_imputado (mediana):", df_imputado.shape[0])
print("Filas df_eliminado (dropna):", df_eliminado.shape[0])

# Revisamos las columnas redundantes que den informaciión que ya tenemos
# y que no nos aporten mucho, ya que pueden generar ruido.
print("------- Columnas del data set -------")
print(df.info())

# Las columnas redundantes encontradas son las siguientes
columnas_redundantes = [
    "student_id",
    "gpa_change",
    "study",
    "university",
    "semester",
    "avg_sleep_minutes",
    "sleep_midpoint_clock",
    "under_6h_sleep",
    "sleep_bracket",
]

df = df.drop(columns=columnas_redundantes)
df_imputado = df_imputado.drop(columns=columnas_redundantes)
df_eliminado = df_eliminado.drop(columns=columnas_redundantes)
print(df.info())

# Ahora grafiquemos los datos que tenemos para darnos una mejor visión de 
# nuestro data set

# Matriz de correlación (solo con las columnas numéricas) sobre df_imputado
corr = df_imputado.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        valor = corr.iloc[i, j]
        color = "white" if abs(valor) > 0.5 else "black"
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color=color, fontsize=7)
fig.colorbar(im, ax=ax)
ax.set_title("Matriz de correlación")
fig.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)

# Columnas numéricas relevantes para explorar
cols_numericas = [
    "avg_sleep_hours",
    "daytime_sleep_minutes",
    "sleep_midpoint_minutes",
    "bedtime_variability",
    "nights_tracked_fraction",
    "prior_gpa",
    "term_gpa",
    "term_units",
    "term_load_z",
]

# Histogramas: distribución de cada variable numérica
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, col in zip(axes.flatten(), cols_numericas):
    ax.hist(df_imputado[col], bins=30, color="steelblue", edgecolor="black")
    ax.set_title(col)
fig.tight_layout()
plt.savefig("histogramas.png", dpi=150)

# Gráficas de dispersión: cada variable vs term_gpa que es nuestra 
# variable a predecir
cols_predictoras = [c for c in cols_numericas if c != "term_gpa"]
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, col in zip(axes.flatten(), cols_predictoras):
    ax.scatter(df_imputado[col], df_imputado["term_gpa"], s=8, alpha=0.4)
    ax.set_xlabel(col)
    ax.set_ylabel("term_gpa")
axes.flatten()[-1].axis("off")
fig.tight_layout()
plt.savefig("dispersion.png", dpi=150)

# Diferencias de term_gpa entre cohortes (boxplot)
cohortes = sorted(df_imputado["cohort_code"].unique())
datos_por_cohorte = [df_imputado.loc[df_imputado["cohort_code"] == c, "term_gpa"] for c in cohortes]

fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(datos_por_cohorte, labels=cohortes)
ax.set_xlabel("cohort_code")
ax.set_ylabel("term_gpa")
ax.set_title("term_gpa por cohorte")
fig.tight_layout()
plt.savefig("gpa_por_cohorte.png", dpi=150)

# Convertimos las categóricas de texto (gender, cohort_code) a columnas
# binarias de 0 y 1
cols_categoricas = ["gender", "cohort_code"]
df = pd.get_dummies(df, columns=cols_categoricas, drop_first=True, dtype=int)
df_imputado = pd.get_dummies(df_imputado, columns=cols_categoricas, drop_first=True, dtype=int)
df_eliminado = pd.get_dummies(df_eliminado, columns=cols_categoricas, drop_first=True, dtype=int)

print("------- Columnas después del encoding -------")
print(df_imputado.columns.tolist())

# Dividimos en train (80%) y test (20%), revolvemos los índices
# con una semilla fija para que sea reproducible y cortamos en 80%.
def train_test_split(data, train_frac=0.8, seed=1):
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    corte = int(len(data) * train_frac)
    idx_train = indices[:corte]
    idx_test = indices[corte:]
    return data.iloc[idx_train].reset_index(drop=True), data.iloc[idx_test].reset_index(drop=True)

# Eliminamos el student_uid ya que no lo vamos a ocupar más
df_imputado = df_imputado.drop(columns=["student_uid"])
df_eliminado = df_eliminado.drop(columns=["student_uid"])

train_imputado, test_imputado = train_test_split(df_imputado)
train_eliminado, test_eliminado = train_test_split(df_eliminado)

print("------- Tamaños del split (df_imputado) -------")
print("Train:", train_imputado.shape[0], "Test:", test_imputado.shape[0])
print("------- Tamaños del split (df_eliminado) -------")
print("Train:", train_eliminado.shape[0], "Test:", test_eliminado.shape[0])

# ------- Regresión lineal múltiple -------

# Hipótesis
def fun_hipotesis(beta, x, b):
    return np.dot(x, beta) + b


# MSE
def fun_costo(y_predict, y):
    return np.sum((y_predict - y) ** 2) / len(y)


# Optimizacion con gradiente descendiente
def fun_optimizacion(X, y, beta, b, alpha):
    m = len(y)
    y_pred = fun_hipotesis(beta, X, b)
    error = y_pred - y
    d_beta = np.dot(X.T, error) / m
    d_b = np.sum(error) / m
    beta = beta - alpha * d_beta
    b = b - alpha * d_b
    return beta, b, y_pred


# Escalamos a media 0, desviación 1 para que el descenso de gradiente converja bien
def scaling(x, media, desviacion):
    return (x - media) / desviacion


# R2 y RMSE de un modelo ya entrenado, evaluado sobre un DataFrame
def evaluar_regresion(df_eval, columnas_x, columna_y, beta, b):
    X = df_eval[columnas_x].to_numpy(dtype=float)
    y = df_eval[columna_y].to_numpy(dtype=float)
    y_pred = fun_hipotesis(beta, X, b)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / len(y))
    return r2, rmse

# Creamos una funcion para poder evaluar ambos data sets
def entrenar_regresion(train_df, test_df, target="term_gpa", alpha=0.001, max_epochs=10000, r2_target=0.95):
    columnas_x = train_df.columns.drop(target).tolist()

    X_train = train_df[columnas_x].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    X_test = test_df[columnas_x].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    media_train = X_train.mean(axis=0)
    desviacion_train = X_train.std(axis=0)
    desviacion_train = np.where(desviacion_train == 0, 1, desviacion_train)

    X_train_scaled = scaling(X_train, media_train, desviacion_train)
    X_test_scaled = scaling(X_test, media_train, desviacion_train)

    beta = np.zeros(X_train_scaled.shape[1])
    b = 0.0

    cost_history = []
    r2_history = []
    epoch = 0
    while True:
        beta, b, y_pred = fun_optimizacion(X_train_scaled, y_train, beta, b, alpha)
        current_cost = fun_costo(y_pred, y_train)
        ss_res = np.sum((y_train - y_pred) ** 2)
        ss_tot = np.sum((y_train - y_train.mean()) ** 2)
        current_r2 = 1 - ss_res / ss_tot
        cost_history.append(current_cost)
        r2_history.append(current_r2)
        epoch += 1
        if current_r2 >= r2_target or epoch >= max_epochs:
            break

    train_df_scaled = pd.DataFrame(X_train_scaled, columns=columnas_x)
    train_df_scaled[target] = y_train
    test_df_scaled = pd.DataFrame(X_test_scaled, columns=columnas_x)
    test_df_scaled[target] = y_test

    return {
        "beta": beta, "b": b, "epoch": epoch,
        "cost_history": cost_history, "r2_history": r2_history,
        "columnas_x": columnas_x, "target": target,
        "train_df": train_df_scaled, "test_df": test_df_scaled,
    }

# Cargamos los data sets a la funcion
modelo_imputado = entrenar_regresion(train_imputado, test_imputado)
modelo_eliminado = entrenar_regresion(train_eliminado, test_eliminado)

for nombre, modelo in [("df_imputado", modelo_imputado), ("df_eliminado", modelo_eliminado)]:
    r2_train, rmse_train = evaluar_regresion(modelo["train_df"], modelo["columnas_x"], modelo["target"], modelo["beta"], modelo["b"])
    r2_test, rmse_test = evaluar_regresion(modelo["test_df"], modelo["columnas_x"], modelo["target"], modelo["beta"], modelo["b"])
    print(f"------- Resultados {nombre} (terminó en la época {modelo['epoch']}) -------")
    print(f"Train -> R2: {r2_train:.3f}  RMSE: {rmse_train:.3f}")
    print(f"Test  -> R2: {r2_test:.3f}  RMSE: {rmse_test:.3f}")

# Gráfica del modelo: term_gpa predicho vs. term_gpa real (sobre test).
# Entre más cerca estén los puntos de la diagonal, mejor predice el modelo.
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, (nombre, modelo) in zip(axes, [("df_imputado", modelo_imputado), ("df_eliminado", modelo_eliminado)]):
    X_test = modelo["test_df"][modelo["columnas_x"]].to_numpy(dtype=float)
    y_test = modelo["test_df"][modelo["target"]].to_numpy(dtype=float)
    y_pred = fun_hipotesis(modelo["beta"], X_test, modelo["b"])

    r2_test, _ = evaluar_regresion(modelo["test_df"], modelo["columnas_x"], modelo["target"], modelo["beta"], modelo["b"])

    ax.scatter(y_test, y_pred, s=15, alpha=0.5)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="black", linestyle="--", label="predicción perfecta")
    ax.text(0.05, 0.95, f"R2 = {r2_test:.3f}", transform=ax.transAxes, va="top",
            bbox=dict(facecolor="white", edgecolor="black"))
    ax.set_xlabel("term_gpa real")
    ax.set_ylabel("term_gpa predicho")
    ax.set_title(f"{nombre} (test)")
    ax.legend()
    ax.grid(True)
fig.tight_layout()
plt.savefig("prediccion_vs_real.png", dpi=150)

# Gráficas de costo y R2, completas y con zoom en las últimas épocas
zoom_desde = int(len(modelo_imputado["cost_history"]) * 0.8)
epocas_zoom = range(zoom_desde, len(modelo_imputado["cost_history"]))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].plot(modelo_imputado["cost_history"], label="df_imputado")
axes[0, 0].plot(modelo_eliminado["cost_history"], label="df_eliminado")
axes[0, 0].set_title("Costo (completo)")
axes[0, 0].set_xlabel("Épocas")
axes[0, 0].set_ylabel("Costo (MSE)")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(epocas_zoom, modelo_imputado["cost_history"][zoom_desde:], label="df_imputado")
axes[0, 1].plot(epocas_zoom, modelo_eliminado["cost_history"][zoom_desde:], label="df_eliminado")
axes[0, 1].set_title(f"Costo (zoom, desde época {zoom_desde})")
axes[0, 1].set_xlabel("Épocas")
axes[0, 1].set_ylabel("Costo (MSE)")
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(modelo_imputado["r2_history"], label="df_imputado")
axes[1, 0].plot(modelo_eliminado["r2_history"], label="df_eliminado")
axes[1, 0].set_title("R2 (completo)")
axes[1, 0].set_xlabel("Épocas")
axes[1, 0].set_ylabel("R2")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(epocas_zoom, modelo_imputado["r2_history"][zoom_desde:], label="df_imputado")
axes[1, 1].plot(epocas_zoom, modelo_eliminado["r2_history"][zoom_desde:], label="df_eliminado")
axes[1, 1].set_title(f"R2 (zoom, desde época {zoom_desde})")
axes[1, 1].set_xlabel("Épocas")
axes[1, 1].set_ylabel("R2")
axes[1, 1].legend()
axes[1, 1].grid(True)

fig.tight_layout()
plt.savefig("curvas_entrenamiento.png", dpi=150)

