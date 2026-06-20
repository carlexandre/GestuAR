import json
import time
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO — ajuste aqui conforme seu ambiente
# ═══════════════════════════════════════════════════════════════
# Para IP Cam (ex: IP Webcam no Android): use a URL do stream
# Para webcam local: use 0
CAMERA_SOURCE = "http://192.168.0.2:8080/video"   # ← sua IP Cam
# CAMERA_SOURCE = 0                                # ← webcam local

MODEL_PATH   = "Trabalho_Final_ama/modelo_cnn/modelo_cnn.keras"
CLASSES_PATH = "Trabalho_Final_ama/modelo_cnn/classes.json"
IMG_SIZE     = (224, 224)
CONF_LIMIAR  = 0.4    # exibir predição só se confiança ≥ 40%
SMOOTH_N     = 5       # média móvel sobre N frames para suavizar
PREDIZER_A_CADA = 3    # faz predição a cada N frames (reduz lag na IP cam)
# ═══════════════════════════════════════════════════════════════

# Carrega modelo e classes
print("Carregando modelo CNN...")
model   = tf.keras.models.load_model(MODEL_PATH)
classes = json.load(open(CLASSES_PATH))
print(f"✅ Modelo carregado | {len(classes)} classes: {classes}")

# Pré-processamento de um frame 
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    #Recorta região central 1:1, redimensiona e normaliza para [0,1].
    h, w  = frame.shape[:2]
    side  = min(h, w)
    y0    = (h - side) // 2
    x0    = (w - side) // 2
    crop  = frame[y0:y0+side, x0:x0+side]
    resized = cv2.resize(crop, IMG_SIZE)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

# Abre câmera 
print(f"Conectando à câmera: {CAMERA_SOURCE}")
cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print("❌ Não foi possível abrir a câmera.")
    print("Verifique se o CAMERA_SOURCE está correto e se o celular")
    print("está na mesma rede Wi-Fi que o PC.")
    exit(1)

print("✅ Câmera aberta! Pressione Q ou ESC para sair.")

# Estado da predição 
prob_buffer  = []
ultima_letra = "..."
ultima_conf  = 0.0
ultimo_top3  = []
frame_cnt    = 0
fps_timer    = time.time()
fps          = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("❌ Falha ao capturar frame. Reconectando...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        continue

    frame_cnt += 1

    # FPS (calculado a cada 10 frames)
    if frame_cnt % 10 == 0:
        fps = 10 / (time.time() - fps_timer)
        fps_timer = time.time()

    # Predição (a cada PREDIZER_A_CADA frames)
    if frame_cnt % PREDIZER_A_CADA == 0:
        x = preprocess_frame(frame)
        probs = model.predict(np.expand_dims(x, 0), verbose=0)[0]

        prob_buffer.append(probs)
        if len(prob_buffer) > SMOOTH_N:
            prob_buffer.pop(0)
        avg_probs = np.mean(prob_buffer, axis=0)

        idx = int(np.argmax(avg_probs))
        ultima_conf = float(avg_probs[idx])
        ultima_letra = classes[idx] if ultima_conf >= CONF_LIMIAR else "?"

        top3_idx = np.argsort(avg_probs)[::-1][:3]
        ultimo_top3 = [(classes[i], float(avg_probs[i]) * 100) for i in top3_idx]

    # Desenho 
    h, w = frame.shape[:2]

    # painel lateral semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, 220), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # letra principal
    cor_letra = (0, 255, 100) if ultima_conf >= CONF_LIMIAR else (80, 80, 255)
    cv2.putText(frame, ultima_letra,
                (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 3.8, cor_letra, 6, cv2.LINE_AA)

    # confiança
    cor_conf = (0,255,0) if ultima_conf >= 0.60 else \
               (0,165,255) if ultima_conf >= 0.40 else (0,0,255)
    cv2.putText(frame, f"Conf: {ultima_conf*100:.1f}%",
                (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_conf, 2, cv2.LINE_AA)

    # top 3
    cv2.putText(frame, "Top 3:", (15, 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    for i, (letra, conf) in enumerate(ultimo_top3):
        cv2.putText(frame, f"  {letra}: {conf:.1f}%",
                    (15, 178 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # FPS e instrução
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1, cv2.LINE_AA)
    cv2.putText(frame, "Pressione Q para sair",
                (w//2 - 120, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1, cv2.LINE_AA)

    # quadrado guia (região que a CNN analisa)
    side  = min(h, w)
    y0_sq = (h - side) // 2
    x0_sq = (w - side) // 2
    cv2.rectangle(frame, (x0_sq, y0_sq), (x0_sq + side, y0_sq + side),
                  (100, 200, 255), 2)
    cv2.putText(frame, "area analisada",
                (x0_sq + 4, y0_sq + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,200,255), 1)

    cv2.imshow("LIBRAS CNN — IP Cam", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Encerrado.")