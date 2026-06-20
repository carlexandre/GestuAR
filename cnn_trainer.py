"""
cnn_trainer.py
==============
Treina uma CNN (MobileNetV2 com fine-tuning) para classificar sinais de LIBRAS
diretamente a partir das imagens, sem uso do MediaPipe.

Salva:
  - Trabalho_Final_ama/modelo_cnn/modelo_cnn.keras  (modelo treinado)
  - Trabalho_Final_ama/modelo_cnn/metricas_cnn.pkl  (dict com métricas e tempo)
  - Trabalho_Final_ama/modelo_cnn/classes.json       (lista de classes na ordem do modelo)
  - Trabalho_Final_ama/modelo_cnn/historico.pkl      (histórico de treino: loss/acc por época)

Uso:
  python cnn_trainer.py
"""

import json
import time
import joblib
import numpy as np
from pathlib import Path

# Pillow com suporte a HEIC
from PIL import Image
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("✅ Suporte a HEIC ativado")
except ImportError:
    print(" pillow-heif não encontrado. Arquivos .HEIC serão ignorados.")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------
INPUT_DIR   = Path("Trabalho_Final_ama/data/dataset_imagens")  # pasta raiz com subpastas por classe
OUTPUT_DIR  = Path("Trabalho_Final_ama/modelo_cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = (224, 224)   # resolução esperada pelo MobileNetV2
BATCH_SIZE  = 32
EPOCHS_HEAD = 10           # epochs treinando só o classificador (backbone congelado)
EPOCHS_FINE = 20           # epochs de fine-tuning (últimas camadas do backbone)
SEED        = 42
EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}

# ---------------------------------------------
# 1. CARREGAMENTO DOS CAMINHOS
# ---------------------------------------------
print("\n--------------- Carregando imagens ------------------------")
paths, labels = [], []

for classe_dir in sorted(INPUT_DIR.iterdir()):
    if not classe_dir.is_dir():
        continue
    label = classe_dir.name.upper()
    imagens = [f for f in classe_dir.iterdir() if f.suffix.lower() in EXTENSIONS]
    for img_path in imagens:
        paths.append(img_path)
        labels.append(label)
    print(f"  [{label}] {len(imagens)} imagens")

print(f"\nTotal: {len(paths)} imagens | {len(set(labels))} classes")

le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)
n_classes = len(classes)

json.dump(classes, open(OUTPUT_DIR / "classes.json", "w"), ensure_ascii=False, indent=2)
joblib.dump(le, OUTPUT_DIR / "label_encoder_cnn.pkl")
print(f"Classes: {classes}")

# ---------------------------------------------
# 2. SPLIT ESTRATIFICADO
# ---------------------------------------------
paths = np.array(paths)
idx = np.arange(len(paths))

idx_train, idx_tmp, y_train, y_tmp = train_test_split(
    idx, y, test_size=0.30, stratify=y, random_state=SEED
)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
)

print(f"\nSplit — treino: {len(idx_train)} | val: {len(idx_val)} | teste: {len(idx_test)}")

# ---------------------------------------------
# 3. FUNÇÃO DE CARREGAMENTO / PRÉ-PROCESSAMENTO
# ---------------------------------------------
def load_image(path: Path) -> np.ndarray:
    """Lê qualquer formato suportado (incl. HEIC) e devolve array float32 [0,1]."""
    try:
        if path.suffix.lower() == ".heic":
            img = Image.open(path).convert("RGB")
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            return np.array(img, dtype=np.float32) / 255.0
        else:
            img = Image.open(path).convert("RGB")
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"  [ERRO] {path}: {e}")
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)


class LibrasDataset(keras.utils.Sequence):
    """Dataset Keras que carrega imagens sob demanda (memory-efficient).

    `path_indices` são índices no array global `paths` (para carregar a imagem).
    `labels`       são os rótulos já alinhados a esses índices (mesmo tamanho).
    Internamente mantemos `order` — posições locais de 0..N-1 — que são
    embaralhadas a cada época; isso evita o IndexError de usar índices globais
    para acessar o array de labels local.
    """

    def __init__(self, path_indices, labels, paths, augment=False, batch_size=BATCH_SIZE):
        self.path_indices = np.array(path_indices)   # índices globais → paths[]
        self.labels       = np.array(labels)         # rótulos locais (mesma ordem)
        self.paths        = paths
        self.augment      = augment
        self.batch_size   = batch_size
        # ordem local: 0, 1, 2, … len-1  (embaralhada a cada época)
        self.order        = np.arange(len(self.labels))

        # Augmentation layers
        self._aug = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
        ])

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch_size))

    def __getitem__(self, batch_idx):
        # posições locais do batch
        local_pos = self.order[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        # converter para índices globais apenas para carregar as imagens
        global_idx = self.path_indices[local_pos]
        X = np.stack([load_image(self.paths[i]) for i in global_idx])
        y = self.labels[local_pos]          # acesso local → sem IndexError
        if self.augment:
            X = self._aug(X, training=True).numpy()
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.order)


ds_train = LibrasDataset(idx_train, y_train, paths, augment=True)
ds_val   = LibrasDataset(idx_val,   y_val,   paths, augment=False)
ds_test  = LibrasDataset(idx_test,  y_test,  paths, augment=False)

# ---------------------------------------------
# 4. CONSTRUÇÃO DO MODELO
# ---------------------------------------------
print("\n-- Construindo modelo ---------------------------------------------")

def build_model(n_classes: int, trainable_base: bool = False):
    backbone = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = trainable_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # MobileNetV2 espera pixels em [-1, 1]; já temos [0, 1], então rescalamos
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = backbone(x, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


model = build_model(n_classes, trainable_base=False)
model.summary(print_fn=lambda s: print(s) if "Total" in s or "Trainable" in s else None)

callbacks_head = [
    EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6),
]

# ---------------------------------------------
# 5. FASE 1 — TREINAR APENAS O CLASSIFICADOR
# ---------------------------------------------
print("\n--- Fase 1: treinando classificador (backbone congelado) ---------------")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

t0 = time.time()
history_head = model.fit(
    ds_train, validation_data=ds_val,
    epochs=EPOCHS_HEAD, callbacks=callbacks_head, verbose=1
)

# ---------------------------------------------
# 6. FASE 2 — FINE-TUNING (últimas 30 camadas do backbone)
# ---------------------------------------------
print("\n--- Fase 2: fine-tuning das últimas 30 camadas --------------------------")
backbone = model.layers[2]           # backbone é a 3ª camada (Input, Rescaling, MobileNetV2)
backbone.trainable = True
for layer in backbone.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),   # LR menor para fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = str(OUTPUT_DIR / "checkpoint_best.keras")
callbacks_fine = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7),
    ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy"),
]

history_fine = model.fit(
    ds_train, validation_data=ds_val,
    epochs=EPOCHS_FINE, callbacks=callbacks_fine, verbose=1
)

tempo_treinamento = time.time() - t0
print(f"\nTempo total de treinamento: {tempo_treinamento:.1f}s ({tempo_treinamento/60:.1f} min)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. AVALIAÇÃO NO CONJUNTO DE TESTE
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Avaliando no conjunto de teste ----------------------------------")
y_pred_proba = model.predict(ds_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# y_true já está alinhado e ordenado em ds_test.labels
y_true = ds_test.labels

acc   = accuracy_score(y_true, y_pred)
prec  = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec   = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
cm    = confusion_matrix(y_true, y_pred)

print(f"\nAcurácia : {acc:.4f}")
print(f"Precisão : {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nRelatório completo:")
print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

# ---------------------------------------------
# 8. SALVAR MODELO E MÉTRICAS
# ---------------------------------------------
model_path = OUTPUT_DIR / "modelo_cnn.keras"
model.save(str(model_path))
print(f"\n✅ Modelo salvo em: {model_path}")

# Histórico combinado
history_combined = {}
for k, v in history_head.history.items():
    history_combined[k] = v
for k, v in history_fine.history.items():
    history_combined.setdefault(k, [])
    history_combined[k].extend(v)

joblib.dump(history_combined, OUTPUT_DIR / "historico.pkl")

metricas_cnn = {
    "nome":              "CNN (MobileNetV2)",
    "accuracy":          float(acc),
    "precision_macro":   float(prec),
    "recall_macro":      float(rec),
    "f1_macro":          float(f1),
    "confusion_matrix":  cm.tolist(),
    "classes":           classes,
    "tempo_treinamento": float(tempo_treinamento),
    "n_treino":          int(len(idx_train)),
    "n_val":             int(len(idx_val)),
    "n_teste":           int(len(idx_test)),
}
joblib.dump(metricas_cnn, OUTPUT_DIR / "metricas_cnn.pkl")
print("✅ Métricas salvas em:", OUTPUT_DIR / "metricas_cnn.pkl")
print("\nTreinamento concluído!")
