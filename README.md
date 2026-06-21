# GestuAR

Classificador de sinais do alfabeto em **LIBRAS** (Língua Brasileira de Sinais) usando visão computacional e aprendizado de máquina. O projeto explora e compara duas abordagens distintas: extração de landmarks via **MediaPipe** combinada com modelos clássicos de ML, e classificação direta de imagens com uma **CNN (MobileNetV2)** via transfer learning.

---

## Abordagens

### MediaPipe + ML Clássico
O MediaPipe detecta a mão e extrai 21 pontos de referência (landmarks) em coordenadas x, y, z — totalizando 63 valores por amostra. Esses landmarks são normalizados pelo pulso e escala, e então usados como entrada para quatro modelos:

- **SVM** — Support Vector Machine com kernel RBF
- **KNN** — K-Nearest Neighbors
- **RNA** — Rede Neural Artificial (MLP)
- **Random Forest**

Os hiperparâmetros de cada modelo são otimizados via `GridSearchCV` com validação cruzada estratificada de 10 folds.

### CNN Direta (MobileNetV2)
A imagem bruta (224×224 px) é passada diretamente para uma rede MobileNetV2 pré-treinada no ImageNet. O treinamento ocorre em duas fases: primeiro apenas o classificador (backbone congelado), depois fine-tuning das últimas 30 camadas. Não depende do MediaPipe — a rede aprende os padrões visuais diretamente.

O projeto inclui dois modelos CNN treinados com quantidades diferentes de amostras por classe (50 e 230) para demonstrar o impacto do volume de dados.

---

## Classes

O classificador reconhece **21 letras estáticas** do alfabeto em LIBRAS. Letras que envolvem movimento (H, J, K, X, Z) são excluídas por exigirem análise temporal:

`A B C D E F G I L M N O P Q R S T U V W Y`

---

## Estrutura do Projeto

```
GestuAR/
│
├── data/                          # Dados brutos e processados
│   ├── dataset_imagens/           # Imagens organizadas por classe (A/, B/, …)
│   ├── landmarks/                 # CSVs de landmarks do dataset principal
│   ├── landmarks_imagem/          # CSVs de landmarks extraídos de imagens
│   └── landmarks_teste/           # CSVs do conjunto de teste
│
├── models/                        # Modelos treinados e artefatos
│   ├── mediapipe/
│   │   ├── modelo_libras.pkl      # Melhor modelo MediaPipe (por F1)
│   │   ├── label_encoder.pkl      # Encoder das classes
│   │   ├── hiperparametros.pkl    # Melhores hiperparâmetros encontrados
│   │   └── metricas_modelos.pkl   # Métricas de todos os modelos
│   └── cnn/
│       ├── 50_amostras/           # CNN treinada com 50 imagens/classe
│       │   ├── modelo_cnn.keras
│       │   ├── metricas_cnn.pkl
│       │   ├── historico.pkl
│       │   ├── classes.json
│       │   └── label_encoder_cnn.pkl
│       └── 230_amostras/          # CNN treinada com média de 230 imagens/classe
│           ├── modelo_cnn.keras
│           ├── metricas_cnn.pkl
│           ├── historico.pkl
│           ├── classes.json
│           └── label_encoder_cnn.pkl
│
├── scripts/                       # Pipeline de treinamento (executar em ordem)
│   ├── 1_extrair_landmarks.py     # Extrai landmarks das imagens via MediaPipe
│   ├── 2_hiperparametros.py       # Busca de hiperparâmetros com GridSearchCV
│   ├── 3_treinar_mediapipe.py     # Treina e avalia KNN, SVM, RNA, Random Forest
│   └── 4_treinar_cnn.py           # Treina a CNN com MobileNetV2
│
├── analise/                       # Visualizações e comparativos
│   ├── comparar_modelos.py        # Dashboard HTML comparando todas as abordagens
│   └── metricas_cnn.py            # Gráficos detalhados da CNN
│
├── app/                           # Inferência em tempo real
│   ├── realtime_mediapipe.py      # Predição via MediaPipe + modelo clássico
│   └── realtime_cnn.py            # Predição via CNN direta (suporta IP Cam)
│
└── README.md
```

---

## Como Executar

### Pré-requisitos

```bash
pip install tensorflow mediapipe scikit-learn opencv-python matplotlib joblib pillow pillow-heif
```

### Pipeline de Treinamento

Execute os scripts na ordem abaixo a partir da raiz do projeto:

```bash
# 1. Extrai landmarks das imagens de teste via MediaPipe
python scripts/1_extrair_landmarks.py

# 2. Busca os melhores hiperparâmetros (pode demorar)
python scripts/2_hiperparametros.py

# 3. Treina os modelos clássicos com os hiperparâmetros encontrados
python scripts/3_treinar_mediapipe.py

# 4. Treina a CNN (ajuste MAX_POR_CLASSE no topo do arquivo)
python scripts/4_treinar_cnn.py
```

### Análise dos Resultados

```bash
# Gera dashboard HTML comparando todas as abordagens
python analise/comparar_modelos.py

# Gera gráficos detalhados da CNN (distribuição, confusões, previsão visual)
python analise/metricas_cnn.py
```

### Inferência em Tempo Real

Ambos os scripts suportam **IP Cam** (ex: app IP Webcam no Android) e webcam local. Ajuste `CAMERA_SOURCE` no topo de cada arquivo:

```bash
# Com MediaPipe
python app/realtime_mediapipe.py

# Com CNN direta
python app/realtime_cnn.py
```

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| MediaPipe | Detecção e extração de landmarks da mão |
| TensorFlow / Keras | Treinamento e inferência da CNN |
| scikit-learn | Modelos clássicos, GridSearchCV, métricas |
| OpenCV | Captura de vídeo e visualização |
| Matplotlib | Gráficos e visualizações |
| Pillow / pillow-heif | Leitura de imagens incluindo formato HEIC |
