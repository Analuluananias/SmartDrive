
import cv2
import time
import serial
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from eye_detector import (
    draw_eye_landmarks,
    calculate_both_eyes_ear
)

from config import (
    EAR_THRESHOLD,
    EYES_CLOSED_SECONDS
)

from metrics import DrowsinessMetrics
from drowsiness import classify_drowsiness


# ==========================================
# CONFIGURAÇÃO DO ESP32
# ==========================================

esp32 = serial.Serial("COM5", 115200, timeout=1)

time.sleep(2)

ultimo_comando = None


def enviar_comando(comando):
    global ultimo_comando

    if comando != ultimo_comando:
        esp32.write((comando + "\n").encode())
        esp32.flush()

        ultimo_comando = comando

        print("Comando enviado:", comando)


# ==========================================
# CONFIGURAÇÃO DO MEDIAPIPE
# ==========================================

MODEL_PATH = "assets/models/face_landmarker.task"


def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERRO: Não foi possível acessar a câmera.")
        esp32.close()
        return

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.FaceLandmarker.create_from_options(
        options
    )

    metrics = DrowsinessMetrics()

    print("SmartDrive - detector facial iniciado!")
    print("ESP32 conectado!")
    print("Pressione ESC para fechar.")

    frame_timestamp = 0
    eyes_closed_start = None

    # Controle da sonolência contínua
    drowsiness_start = None

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("ERRO: Não foi possível capturar a imagem.")
                break

            # Espelha a câmera
            frame = cv2.flip(frame, 1)

            # Converte BGR para RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Cria imagem para o MediaPipe
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            frame_timestamp += 33

            # Detecta o rosto
            result = detector.detect_for_video(
                mp_image,
                frame_timestamp
            )

            # ==================================
            # ROSTO DETECTADO
            # ==================================

            if result.face_landmarks:

                for face_landmarks in result.face_landmarks:

                    draw_eye_landmarks(
                        frame,
                        face_landmarks
                    )

                    # Calcula EAR
                    left_ear, right_ear, average_ear = (
                        calculate_both_eyes_ear(
                            face_landmarks
                        )
                    )

                    current_time = time.time()

                    # Verifica olhos fechados
                    eyes_closed = (
                        average_ear < EAR_THRESHOLD
                    )

                    # Atualiza métricas
                    metrics.update(
                        eyes_closed,
                        current_time
                    )

                    # Controla tempo de olhos fechados
                    if eyes_closed:

                        if eyes_closed_start is None:
                            eyes_closed_start = current_time

                        closed_duration = (
                            current_time - eyes_closed_start
                        )

                    else:

                        eyes_closed_start = None
                        closed_duration = 0.0

                    # Calcula PERCLOS
                    perclos = metrics.get_perclos()

                    # Classifica sonolência
                    drowsiness_state = classify_drowsiness(
                        average_ear,
                        closed_duration,
                        perclos
                    )

                    # ==================================
                    # CONTROLE DO TEMPO DE SONOLÊNCIA
                    # ==================================

                    if drowsiness_state in ("ATENCAO", "ALERTA"):

                        if drowsiness_start is None:
                            drowsiness_start = current_time

                        drowsiness_duration = (
                            current_time - drowsiness_start
                        )

                        if drowsiness_duration >= 10:
                            enviar_comando("SONOLENCIA_10")
                        else:
                            enviar_comando("SONOLENCIA")

                    else:

                        drowsiness_start = None
                        drowsiness_duration = 0.0

                        enviar_comando("ROSTO")

                    # ==================================
                    # INFORMAÇÕES NA TELA
                    # ==================================

                    cv2.putText(
                        frame,
                        f"EAR: {average_ear:.3f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Olhos fechados: {closed_duration:.1f}s",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Piscadas: {metrics.total_blinks}",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"PERCLOS: {perclos:.1f}%",
                        (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    # ==================================
                    # ESTADO DE SONOLÊNCIA
                    # ==================================

                    if drowsiness_state == "NORMAL":

                        estado_texto = "NORMAL"
                        cor_estado = (0, 255, 0)

                    elif drowsiness_state == "ATENCAO":

                        estado_texto = "POSSIVEL SONOLENCIA"
                        cor_estado = (0, 255, 255)

                    else:

                        estado_texto = "SONOLENCIA DETECTADA"
                        cor_estado = (0, 0, 255)

                    cv2.putText(
                        frame,
                        estado_texto,
                        (20, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cor_estado,
                        2
                    )

                    # Mostra tempo de sonolência contínua
                    if drowsiness_start is not None:

                        cv2.putText(
                            frame,
                            f"Tempo de sonolencia: {drowsiness_duration:.1f}s",
                            (20, 265),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2
                        )

                    # Rosto detectado
                    cv2.putText(
                        frame,
                        "ROSTO DETECTADO",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

            # ==================================
            # ROSTO NÃO DETECTADO
            # ==================================

            else:

                # Zera o contador de sonolência
                drowsiness_start = None
                eyes_closed_start = None

                # Envia comando ao ESP32
                enviar_comando("SEM_ROSTO")

                cv2.putText(
                    frame,
                    "ROSTO NAO DETECTADO",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

            # ==================================
            # MOSTRA A CÂMERA
            # ==================================

            cv2.imshow(
                "SmartDrive - Face Detector",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

    finally:

        enviar_comando("ROSTO")

        detector.close()
        camera.release()
        cv2.destroyAllWindows()

        esp32.close()

        print("SmartDrive - detector encerrado.")


if __name__ == "__main__":
    main()
