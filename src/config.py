# ==============================
# LANDMARKS DOS OLHOS
# ==============================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144,
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380,
]


# ==============================
# CONFIGURAÇÕES DE SONOLÊNCIA
# ==============================

# Valor abaixo do qual consideramos
# que o olho está fechado.
#
# Seu teste:
# Aberto: aproximadamente 0.390 - 0.600
# Fechado: aproximadamente 0.040
#
# Começaremos com um valor conservador.
EAR_THRESHOLD = 0.20


# Tempo mínimo de olhos fechados
# para gerar um possível alerta.
#
# NÃO é o alerta final de sonolência.
# Será ajustado durante os testes.
EYES_CLOSED_SECONDS = 1.5