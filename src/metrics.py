import time


class DrowsinessMetrics:
    """
    Responsável por acompanhar o comportamento dos olhos.
    """

    def __init__(self):
        self.eyes_closed_start = None

        self.total_blinks = 0
        self.blink_in_progress = False

        self.window_start = time.time()

        self.closed_time = 0.0
        self.total_time = 0.0

    def update(self, eyes_closed, current_time=None):
        """
        Atualiza as métricas a cada frame.

        eyes_closed:
            True  -> olhos fechados
            False -> olhos abertos
        """

        if current_time is None:
            current_time = time.time()

        # Detecta o início do fechamento
        if eyes_closed:

            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time

            if not self.blink_in_progress:
                self.blink_in_progress = True

        # Detecta a abertura novamente
        else:

            if self.eyes_closed_start is not None:

                closed_duration = (
                    current_time - self.eyes_closed_start
                )

                # Fechamento curto = piscada
                if closed_duration < 0.8:
                    self.total_blinks += 1

                self.closed_time += closed_duration

            self.eyes_closed_start = None
            self.blink_in_progress = False

        # Tempo total da janela
        self.total_time = (
            current_time - self.window_start
        )

    def get_perclos(self):
        """
        Calcula o PERCLOS.

        PERCLOS = porcentagem do tempo
        em que os olhos ficaram fechados.
        """

        if self.total_time <= 0:
            return 0.0

        return (
            self.closed_time / self.total_time
        ) * 100

    def reset_window(self):
        """
        Reinicia a janela de cálculo.
        """

        self.window_start = time.time()
        self.closed_time = 0.0
        self.total_time = 0.0