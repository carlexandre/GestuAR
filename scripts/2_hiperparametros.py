import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
def normalizar_pelo_pulso(X):
    X = X.to_numpy(dtype=np.float32)

    X_norm = []

    for amostra in X:
        pontos = amostra.reshape(21, 3)

        pulso = pontos[0]
        pontos = pontos - pulso

        escala = np.linalg.norm(pontos[9])

        if escala > 0:
            pontos = pontos / escala

        X_norm.append(pontos.flatten())

    return np.array(X_norm)
dataframes = []
for arquivo in Path("../data/landmarks").glob("*.csv"):
    classe = arquivo.stem
    df = pd.read_csv(arquivo, header=None)
    if df.shape[1] < 63:
        continue
    df["label"] = classe
    df["origem"] = "dataset1"
    dataframes.append(df)

for arquivo in Path("../data/landmarks_imagem").glob("*.csv"):
    classe = arquivo.stem
    df = pd.read_csv(arquivo, header=None)
    if df.shape[1] < 63:
        continue
    df["label"] = classe
    df["origem"] = "dataset2"
    dataframes.append(df)

df_total = pd.concat(dataframes, ignore_index=True)
x = df_total.drop(["label", "origem"], axis=1)
x = normalizar_pelo_pulso(x)
y = df_total["label"]
stratify_label = (df_total["label"].astype(str) + "_" + df_total["origem"].astype(str))

le = LabelEncoder()
y = le.fit_transform(y)

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.2,random_state=42,stratify=stratify_label)
def nomes_modelos(nome):
    if nome == "svc":
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf',probability=True))
        ])
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': [0.001, 0.01, 0.1, 1]
        }
    elif nome == "RandomForest":
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42,n_jobs=-1))
        ])
        param_grid = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [4, 8, None]
        }
    elif nome == "Rede Neural Artificial":
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(solver='adam',max_iter=1000,random_state=42))
        ])
        param_grid = {
            "mlp__hidden_layer_sizes": [(128,),(128, 64),(256, 128)],
            "mlp__alpha": [1e-4, 1e-3],
            "mlp__activation": ['relu'],
            "mlp__learning_rate": ['constant','adaptive']
        }
    else:
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        param_grid = {
            'knn__n_neighbors': [3,5,7,9],
            'knn__weights': ['uniform','distance'],
            'knn__metric': ['euclidean','manhattan']
        }
    return pipeline,param_grid

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
best_params = {}
for nome in ["svc","RandomForest","Rede Neural Artificial","KNN"]:
    print(f"\nBuscando hiperparâmetros de {nome}")
    pipeline, param_grid = nomes_modelos(nome)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=3
    )
    grid.fit(x_treino, y_treino)
    best_params[nome] = grid.best_params_
    print(grid.best_params_)

joblib.dump(best_params, "../models/mediapipe/hiperparametros.pkl")
joblib.dump(le, "../models/mediapipe/label_encoder.pkl")