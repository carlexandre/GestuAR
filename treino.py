'''import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
def normalizar_pelo_pulso(X):
    X = np.asarray(X, dtype=np.float32)

    X_norm = []

    for amostra in X:
        pontos = amostra.reshape(21, 3)

        pulso = pontos[0]
        pontos = pontos - pulso

        escala = np.linalg.norm(pontos[9])

        if escala > 0:
            pontos = pontos / escala

        X_norm.append(pontos.flatten())

    return np.array(X_norm, dtype=np.float32)
dataframes = []
for arquivo in Path("Trabalho_Final_ama/data/landmarks").glob("*.csv"):
    classe = arquivo.stem
    df = pd.read_csv(arquivo, header=None)
    if df.shape[1] < 63:
        continue
    df["label"] = classe

    df["origem"] = "dataset1"
    dataframes.append(df)

for arquivo in Path("Trabalho_Final_ama/data/landmarks_imagem").glob("*.csv"):
    classe = arquivo.stem
    df = pd.read_csv(arquivo, header=None)
    if df.shape[1] < 63:
        continue
    df["label"] = classe
    df["origem"] = "dataset2"
    dataframes.append(df)
teste = []
for arquivo in Path("Trabalho_Final_ama/data/landmarks_teste").glob("*.csv"):
    classe = arquivo.stem
    df = pd.read_csv(arquivo, header=None)
    if df.shape[1] < 63:
        continue
    df["label"] = classe
    df["origem"] = "dataset2"
    teste.append(df)

df_total = pd.concat(dataframes, ignore_index=True)
x = df_total.drop(["label", "origem"], axis=1)
x = normalizar_pelo_pulso(x)
y = df_total["label"]
stratify_label = (
    df_total["label"].astype(str)
    + "_"
    + df_total["origem"].astype(str)
)
df_total_teste = pd.concat(teste, ignore_index=True)
xt = df_total_teste.drop(["label", "origem"], axis=1)
xt = normalizar_pelo_pulso(xt)
yt = df_total["label"]
stratify_label_teste = (
    df_total_teste["label"].astype(str)
    + "_"
    + df_total_teste["origem"].astype(str)
)
le = LabelEncoder()
yt = df_total_teste["label"]

y = le.fit_transform(y)
yt = le.transform(yt)

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.2,random_state=42,stratify=stratify_label)
x_teste = np.vstack([x_teste, xt])
y_teste = np.concatenate([y_teste, yt])
def metricas(m):
    y_pred = m.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    prec = precision_score(y_teste, y_pred, average='macro')
    rec = recall_score(y_teste, y_pred, average='macro')
    f1 = f1_score(y_teste, y_pred, average='macro')
    matriz = confusion_matrix(y_teste, y_pred)
    return f1,acc,prec,rec,matriz
best_params = joblib.load("Trabalho_Final_ama/hiperparametros.pkl")

estimadores = {
    "svc": SVC(kernel="rbf",probability=True),
    "RandomForest": RandomForestClassifier(random_state=42,n_jobs=-1),
    "Rede Neural Artificial": MLPClassifier(solver='adam',max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier()
}
modelos = []
metricas_modelos = []
melhor_f1 = -1
melhor_modelo = None

for nome, estimador in estimadores.items():
    print(f"\nModelo: {nome}")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', estimador)
    ])
    params = {}
    for k,v in best_params[nome].items():
        novo_nome = k.split("__")[1]
        params[f"model__{novo_nome}"] = v
    pipeline.set_params(**params)
    pipeline.fit(x_treino, y_treino)

    metricas_modelo = metricas(pipeline)
    print(f"F1: {metricas_modelo[0]:.4f}")
    print(f"Acurácia: {metricas_modelo[1]:.4f}")
    print(f"Precisão: {metricas_modelo[2]:.4f}")
    print(f"Recall: {metricas_modelo[3]:.4f}")
    modelos.append([nome, metricas_modelo[0],pipeline])
    metricas_modelos.append([nome,metricas_modelo])
melhor_modelo = max(modelos, key=lambda x: x[1])
nome = melhor_modelo[0]
modelo_final = melhor_modelo[2]
print(f"\nMelhor modelo: {nome}")
joblib.dump( modelo_final,"Trabalho_Final_ama/modelo_libras.pkl")
joblib.dump(metricas_modelos, "Trabalho_Final_ama/metricas_modelos.pkl")
'''
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
def normalizar_pelo_pulso(X):
    X = np.asarray(X, dtype=np.float32)

    X_norm = []

    for amostra in X:
        pontos = amostra.reshape(21, 3)

        pulso = pontos[0]
        pontos = pontos - pulso

        escala = np.linalg.norm(pontos[9])

        if escala > 0:
            pontos = pontos / escala

        X_norm.append(pontos.flatten())

    return np.array(X_norm, dtype=np.float32)
dataframes = []
for arquivo in Path("Trabalho_Final_ama/data/landmarks").glob("*.csv"):
    classe = arquivo.stem
    df = pd.read_csv(arquivo, header=None)
    if df.shape[1] < 63:
        continue
    df["label"] = classe

    df["origem"] = "dataset1"
    dataframes.append(df)

for arquivo in Path("Trabalho_Final_ama/data/landmarks_imagem").glob("*.csv"):
    classe = arquivo.stem
    df = pd.read_csv(arquivo, header=None)
    if df.shape[1] < 63:
        continue
    df["label"] = classe
    df["origem"] = "dataset2"
    dataframes.append(df)
teste = []
for arquivo in Path("Trabalho_Final_ama/data/landmarks_teste").glob("*.csv"):
    classe = arquivo.stem
    df = pd.read_csv(arquivo, header=None)
    if df.shape[1] < 63:
        continue
    df["label"] = classe
    df["origem"] = "dataset2"
    teste.append(df)

df_total = pd.concat(dataframes, ignore_index=True)
x_treino = df_total.drop(["label", "origem"], axis=1)
x_treino = normalizar_pelo_pulso(x_treino)
y_treino = df_total["label"]
stratify_label = (
    df_total["label"].astype(str)
    + "_"
    + df_total["origem"].astype(str)
)
df_total_teste = pd.concat(teste, ignore_index=True)
x_teste = df_total_teste.drop(["label", "origem"], axis=1)
x_teste = normalizar_pelo_pulso(x_teste)
y_teste = df_total["label"]
stratify_label_teste = (
    df_total_teste["label"].astype(str)
    + "_"
    + df_total_teste["origem"].astype(str)
)
le = LabelEncoder()
y_teste = df_total_teste["label"]

y_treino = le.fit_transform(y_treino)
y_teste = le.transform(y_teste)

def metricas(m):
    y_pred = m.predict(x_teste)
    acc = accuracy_score(y_teste, y_pred)
    prec = precision_score(y_teste, y_pred, average='macro')
    rec = recall_score(y_teste, y_pred, average='macro')
    f1 = f1_score(y_teste, y_pred, average='macro')
    matriz = confusion_matrix(y_teste, y_pred)
    return [f1,acc,prec,rec,matriz]
best_params = joblib.load("Trabalho_Final_ama/hiperparametros.pkl")

estimadores = {
    "svc": SVC(kernel="rbf",probability=True),
    "RandomForest": RandomForestClassifier(random_state=42,n_jobs=-1),
    "Rede Neural Artificial": MLPClassifier(solver='adam',max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier()
}
modelos = []
metricas_modelos = []
melhor_f1 = -1
melhor_modelo = None

for nome, estimador in estimadores.items():
    print(f"\nModelo: {nome}")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', estimador)
    ])
    params = {}
    for k,v in best_params[nome].items():
        novo_nome = k.split("__")[1]
        params[f"model__{novo_nome}"] = v
    pipeline.set_params(**params)
    inicio = time.perf_counter()
    pipeline.fit(x_treino, y_treino)
    fim = time.perf_counter()

    tempo = fim - inicio
    print(f"Tempo de treino: {tempo:.2f} segundos")
    metricas_modelo = metricas(pipeline)
    metricas_modelo.append(tempo)
    print(f"F1: {metricas_modelo[0]:.4f}")
    print(f"Acurácia: {metricas_modelo[1]:.4f}")
    print(f"Precisão: {metricas_modelo[2]:.4f}")
    print(f"Recall: {metricas_modelo[3]:.4f}")
    modelos.append([nome, metricas_modelo[0],pipeline])
    metricas_modelos.append([nome,metricas_modelo])
melhor_modelo = max(modelos, key=lambda x: x[1])
nome = melhor_modelo[0]
modelo_final = melhor_modelo[2]
print(f"\nMelhor modelo: {nome}")
joblib.dump( modelo_final,"Trabalho_Final_ama/modelo_libras.pkl")
joblib.dump(metricas_modelos, "Trabalho_Final_ama/metricas_modelos.pkl")
