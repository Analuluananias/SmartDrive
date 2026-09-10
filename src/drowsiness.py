from config import EAR_THRESHOLD, EYES_CLOSED_SECONDS


def classify_drowsiness(ear, closed_duration, perclos):
    """
    Classifica o estado do motorista
    com base nos sinais dos olhos.

    Esta classificação é apenas um protótipo
    para o projeto SmartDrive.
    """

    # Olhos fechados por tempo prolongado
    if closed_duration >= EYES_CLOSED_SECONDS:
        return "ALERTA"

    # PERCLOS elevado
    if perclos >= 20:
        return "POSSIVEL SONOLENCIA"

    # EAR abaixo do limite
    if ear < EAR_THRESHOLD:
        return "ATENCAO"

    # Situação normal
    return "NORMAL"