import cv2
import time
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


MODEL_PATH = "assets/models/face_landmarker.task"


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERRO: Não foi possível acessar a câmera.")
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

    print("SmartDrive - detector facial iniciado!")
    print("Pressione ESC para fechar.")

    frame_timestamp = 0
    eyes_closed_start = None

    while True:
        success, frame = camera.read()

        if not success:
            print("ERRO: Não foi possível capturar a imagem.")
            break

        # Espelha a câmera
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

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

        # Verifica se encontrou um rosto
        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                # Desenha os pontos dos olhos
                draw_eye_landmarks(
                    frame,
                    face_landmarks
                )

                # Calcula o EAR
                left_ear, right_ear, average_ear = (
                    calculate_both_eyes_ear(
                        face_landmarks
                    )
                )

                current_time = time.time()

                if average_ear < EAR_THRESHOLD:

                    if eyes_closed_start is None:
                        eyes_closed_start = current_time

                    closed_duration = (
                        current_time - eyes_closed_start
                    )

                else:

                    eyes_closed_start = None
                    closed_duration = 0.0

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

                if closed_duration >= EYES_CLOSED_SECONDS:

                    cv2.putText(
                        frame,
                        "POSSIVEL SONOLENCIA",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                # Mostra o EAR na tela
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
                "ROSTO DETECTADO",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "ROSTO NAO DETECTADO",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        # Mostra a câmera
        cv2.imshow(
            "SmartDrive - Face Detector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    detector.close()
    camera.release()
    cv2.destroyAllWindows()

    print("SmartDrive - detector encerrado.")


if __name__ == "__main__":
    main()