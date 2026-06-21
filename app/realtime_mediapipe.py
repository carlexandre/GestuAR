import cv2
import mediapipe as mp
import joblib
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
modelo = joblib.load("models/mediapipe/modelo_libras.pkl")
le = joblib.load("models/mediapipe/label_encoder.pkl")

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    nome_classe = None
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = None
        if results.multi_handedness:
          handedness = results.multi_handedness[i].classification[0].label
        coords = np.array(
          [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
          dtype=np.float32
        )
        coords -= coords[0]
        
        escala = np.linalg.norm(coords[9])
        if escala > 1e-6:
          coords /= escala
        if handedness == "Right":
          coords[:, 0] *= -1.0
        coords = coords.flatten()
        probabilidades = modelo.predict_proba([coords])[0]
        confianca = probabilidades.max()          
        predicao = int(np.argmax(probabilidades))
        nome_classe = le.inverse_transform([predicao])[0]
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    if nome_classe != None:
      texto = f"{nome_classe} {confianca * 100:.0f}%"
      cv2.putText(image, texto, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3,cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()