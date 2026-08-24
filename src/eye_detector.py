import cv2
import math

from config import LEFT_EYE, RIGHT_EYE


def euclidean_distance(point1, point2):
    """
    Calcula a distância entre dois pontos.
    """

    return math.sqrt(
        (point1.x - point2.x) ** 2
        + (point1.y - point2.y) ** 2
    )


def calculate_ear(face_landmarks, eye_indices):
    """
    Calcula o Eye Aspect Ratio (EAR).

    Quanto menor o valor, mais fechado está o olho.
    """

    p1 = face_landmarks[eye_indices[0]]
    p2 = face_landmarks[eye_indices[1]]
    p3 = face_landmarks[eye_indices[2]]
    p4 = face_landmarks[eye_indices[3]]
    p5 = face_landmarks[eye_indices[4]]
    p6 = face_landmarks[eye_indices[5]]

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)

    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


def calculate_both_eyes_ear(face_landmarks):
    """
    Calcula o EAR dos dois olhos.
    """

    left_ear = calculate_ear(
        face_landmarks,
        LEFT_EYE
    )

    right_ear = calculate_ear(
        face_landmarks,
        RIGHT_EYE
    )

    average_ear = (
        left_ear + right_ear
    ) / 2.0

    return left_ear, right_ear, average_ear


def draw_eye_landmarks(frame, face_landmarks):
    """
    Desenha os pontos principais dos olhos.
    """

    height, width, _ = frame.shape

    for eye in [LEFT_EYE, RIGHT_EYE]:

        for index in eye:

            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (255, 0, 0),
                -1
            )