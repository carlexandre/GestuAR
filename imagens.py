import csv
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from PIL import Image
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("✅ Suporte a HEIC ativado")
except ImportError:
    print("⚠️  pillow-heif não encontrado. Arquivos .HEIC serão ignorados.")

INPUT_DIR = Path("Trabalho_Final_ama/teste")
OUTPUT_DIR = Path("Trabalho_Final_ama/data/landmarks_teste")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
)

for classe_dir in sorted(INPUT_DIR.iterdir()):
    if not classe_dir.is_dir():
        continue

    label = classe_dir.name.upper()
    csv_path = OUTPUT_DIR / f"{label}.csv"
    salvos_classe = 0

    arquivos = [f for f in classe_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".heic")]

    print(f"\n[{label}] {len(arquivos)} imagens encontradas")

    for img_path in sorted(arquivos):
        if img_path.suffix.lower() == ".heic":
            image = Image.open(img_path)
            image_rgb = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [AVISO] Não foi possível ler: {img_path.name}")
            continue

        # MediaPipe espera RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Espelha para corrigir lateralidade
        image_rgb = cv2.flip(image_rgb, 1)

        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            print(f"  [SEM MÃO] {img_path.name}")
            continue

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label
        # Extrai os 63 valores brutos
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )
        coords -= coords[0]
        escala = np.linalg.norm(coords[9])
        if escala > 1e-6:
            coords /= escala
        # 3. Espelha mão direita para o sistema de coordenadas da mão esquerda
        if handedness == "Right":
            coords[:, 0] *= -1.0
        coords = coords.flatten()
        # Salva no CSV da classe
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(coords)
hands.close()