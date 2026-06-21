import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# CONFIGURAÇÃO
CNN_DIR      = Path("../models/cnn/230_amostras") # caminho para o modelo cnn
DATASET_DIR  = Path("../data/dataset_imagens")   # para contar distribuição
IMG_SIZE     = (224, 224)

N_EXEMPLOS = 12   # usado quando IMAGENS_TESTE está vazio


# CARREGA DADOS SALVOS
metricas = joblib.load(CNN_DIR / "metricas_cnn.pkl")
classes  = metricas["classes"]
cm       = np.array(metricas["confusion_matrix"])
n_classes = len(classes)

# normaliza a CM por linha (taxa de erro por classe)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.style.use("seaborn-v0_8-whitegrid")
AZUL    = "#2563EB"
LARANJA = "#F59E0B"
VERDE   = "#16A34A"
VERMELHO= "#DC2626"


# GRÁFICO 1 — Distribuição de amostras por classe

print("Gerando gráfico 1: distribuição de amostras...")

contagens = {}
extensoes = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}

if DATASET_DIR.exists():
    for pasta in sorted(DATASET_DIR.iterdir()):
        if pasta.is_dir():
            label = pasta.name.upper()
            n = len([f for f in pasta.iterdir() if f.suffix.lower() in extensoes])
            if n > 0:
                contagens[label] = n
else:
    # fallback: usa totais da CM (soma das linhas = amostras de teste por classe)
    for i, c in enumerate(classes):
        contagens[c] = int(cm[i].sum())

letras   = list(contagens.keys())
valores  = np.array(list(contagens.values()))
media    = valores.mean()
cores_bar = [AZUL if v >= media else LARANJA for v in valores]

fig1, ax1 = plt.subplots(figsize=(14, 4))
bars = ax1.bar(letras, valores, color=cores_bar, edgecolor="white", linewidth=0.6)
ax1.bar_label(bars, padding=3, fontsize=9)
ax1.set_xlabel("Letra", fontsize=12)
ax1.set_ylabel("Nº de amostras", fontsize=12)
ax1.set_title("Distribuição de Amostras por Classe — CNN", fontsize=14, fontweight="bold")
ax1.axhline(media, color=LARANJA, linestyle="--", linewidth=1.5,
            label=f"Média: {media:.0f}")

patch_acima = mpatches.Patch(color=AZUL,   label="≥ média")
patch_abaixo= mpatches.Patch(color=LARANJA,label="< média")
ax1.legend(handles=[patch_acima, patch_abaixo,
           plt.Line2D([0],[0], color=LARANJA, linestyle="--", label=f"Média: {media:.0f}")],
           fontsize=9)

idx_min = np.argmin(valores)
idx_max = np.argmax(valores)
print(f"  Mínimo : {valores[idx_min]} ({letras[idx_min]})")
print(f"  Máximo : {valores[idx_max]} ({letras[idx_max]})")
print(f"  Média  : {media:.1f}")

plt.tight_layout()
plt.savefig(CNN_DIR / "distribuicao_amostras.png", dpi=150, bbox_inches="tight")
plt.show()


# Gera a Matriz de confusão normalizada 
fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=classes)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45, values_format=".2f")
ax.set_title("Matriz de Confusão Normalizada — CNN MobileNetV2", fontsize=14, pad=16)
plt.tight_layout()
plt.savefig(CNN_DIR / "matriz_confusao_normalizada.png", dpi=150, bbox_inches="tight")
plt.show()

# GRÁFICO 2 — Top 10 confusões mais frequentes
print("\nGerando gráfico 2: top confusões...")

erros = []
for i in range(n_classes):
    for j in range(n_classes):
        if i != j and cm[i, j] > 0:
            erros.append({
                "label":    f"{classes[i]}  →  {classes[j]}",
                "real":     classes[i],
                "predito":  classes[j],
                "contagem": int(cm[i, j]),
                "taxa":     cm_norm[i, j] * 100,
            })

erros.sort(key=lambda x: x["contagem"], reverse=True)
top10 = erros[:10]

print("\nTop 10 confusões mais frequentes:")
print(f"{'Real':>6}  →  {'Predito':<8}  {'Qtd':>5}  {'Taxa':>6}")
print("-" * 38)
for e in top10:
    print(f"  {e['real']:>4}  →  {e['predito']:<8}  {e['contagem']:>4}   {e['taxa']:.1f}%")

labels_top  = [e["label"]    for e in top10][::-1]
counts_top  = [e["contagem"] for e in top10][::-1]
taxas_top   = [e["taxa"]     for e in top10][::-1]

fig2, ax2 = plt.subplots(figsize=(10, 5))
barras = ax2.barh(labels_top, counts_top, color=VERMELHO, alpha=0.80, edgecolor="white")

# anotação: contagem + taxa
for bar, cnt, taxa in zip(barras, counts_top, taxas_top):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f"{cnt}  ({taxa:.1f}%)", va="center", fontsize=9)

ax2.set_xlabel("Nº de amostras confundidas", fontsize=11)
ax2.set_title("Top 10 Confusões Mais Frequentes — CNN", fontsize=13, fontweight="bold")
ax2.set_xlim(0, max(counts_top) * 1.25)
plt.tight_layout()
plt.savefig(CNN_DIR / "top_confusoes.png", dpi=150, bbox_inches="tight")
plt.show()


# GRÁFICO 3 — Previsão visual em imagens reais
print("\nGerando gráfico 3: previsão visual...")

# carrega modelo apenas se necessário
model = tf.keras.models.load_model(str(CNN_DIR / "modelo_cnn.keras"))
model.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), verbose=0)  # warm-up

def carregar_e_preprocessar(path):
    try:
        pil = Image.open(path).convert("RGB")
        orig = np.array(pil)                          # para exibir
        resized = np.array(pil.resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32) / 255.0
        return orig, resized
    except Exception as e:
        print(f"  [ERRO] {path}: {e}")
        return None, None

# seleciona aleatoriamente do dataset
todos = []
if DATASET_DIR.exists():
    for pasta in sorted(DATASET_DIR.iterdir()):
        if pasta.is_dir():
            label = pasta.name.upper()
            imgs  = [f for f in pasta.iterdir() if f.suffix.lower() in extensoes]
            for img in imgs:
                todos.append((img, label))
if not todos:
    print("Nenhuma imagem encontrada. Preencha IMAGENS_TESTE ou verifique DATASET_DIR.")
else:
    idx_sel  = np.random.choice(len(todos), min(N_EXEMPLOS, len(todos)), replace=False)
    caminhos = [todos[i] for i in idx_sel]


if caminhos:
    # batch de inferência
    origs, tensors, verdadeiros = [], [], []
    for path, label_real in caminhos:
        orig, tensor = carregar_e_preprocessar(path)
        if orig is not None:
            origs.append(orig)
            tensors.append(tensor)
            verdadeiros.append(label_real)

    batch  = np.stack(tensors)
    probs  = model.predict(batch, verbose=0)
    preds  = np.argmax(probs, axis=1)

    n_imgs = len(origs)
    n_cols = 4
    n_rows = int(np.ceil(n_imgs / n_cols))

    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    for i in range(n_imgs):
        ax   = axes[i]
        pred = classes[preds[i]]
        conf = probs[i][preds[i]] * 100
        real = verdadeiros[i]

        top3_idx = np.argsort(probs[i])[::-1][:3]
        top3_str = "  ".join([f"{classes[k]}:{probs[i][k]*100:.0f}%" for k in top3_idx])

        ax.imshow(origs[i])
        ax.axis("off")

        if real is not None:
            correto  = (pred == real)
            cor      = VERDE if correto else VERMELHO
            simbolo  = "✓" if correto else "✗"
            titulo   = f"{simbolo} Real: {real}  |  Pred: {pred} ({conf:.0f}%)"
        else:
            cor    = AZUL
            titulo = f"Pred: {pred}  ({conf:.0f}%)"

        ax.set_title(titulo, color=cor, fontsize=9, fontweight="bold", pad=4)
        ax.text(0.5, -0.04, top3_str, transform=ax.transAxes,
                ha="center", fontsize=7, color="gray")

    # desliga eixos vazios
    for i in range(n_imgs, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Previsão Visual — CNN MobileNetV2", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(CNN_DIR / "previsao_visual.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nGráficos salvos em:", CNN_DIR)