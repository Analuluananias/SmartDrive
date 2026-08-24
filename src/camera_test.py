import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERRO: Não foi possível acessar a câmera.")
        return

    print("SmartDrive - câmera iniciada!")
    print("Pressione ESC para fechar.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERRO: Não foi possível capturar a imagem.")
            break

        cv2.imshow("SmartDrive - Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

    camera.release()
    cv2.destroyAllWindows()

    print("SmartDrive - câmera encerrada.")


if __name__ == "__main__":
    main()