
import cv2
from pathlib import Path

def capturar_rostos():

    codigo = input("Digite o código do aluno: ").strip()
    nome = input("Digite o nome do aluno: ").strip()

    if not codigo or not nome:
        print("Código e nome são obrigatórios.")
        return

    if not codigo.isdigit():
        print("O código deve conter apenas números.")
        return

    # Diretório de armazenamento
    raiz = Path(__file__).resolve().parents[1]
    pasta = raiz / "dataset" / codigo
    pasta.mkdir(parents=True, exist_ok=True)

   # Caminho do classificador Haar Cascade
caminho_cascade = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Carregar detector facial
detector = cv2.CascadeClassifier(caminho_cascade)

# Verificar se o arquivo foi carregado
if detector.empty():
    print("ERRO: Não foi possível carregar o Haar Cascade.")
    print("Caminho utilizado:", caminho_cascade)
    camera.release()
    return

print("Detector facial carregado com sucesso!")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Não foi possível acessar a webcam.")
        return

    contador = len(list(pasta.glob("*.jpg")))

    print(f"Captura iniciada para {nome}.")
    print("Pressione S para salvar uma imagem.")
    print("Pressione Q para sair.")

    while True:
        sucesso, frame = camera.read()

        if not sucesso:
            print("Falha na captura.")
            break

        cinza = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )

        rostos = detector.detectMultiScale(
            cinza,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in rostos:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Imagens: {contador}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Cadastro Facial", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("s"):
            if len(rostos) == 1:
                x, y, w, h = rostos[0]

                rosto = cinza[y:y+h, x:x+w]
                rosto = cv2.resize(rosto, (200, 200))

                contador += 1

                arquivo = pasta / f"{contador:03d}.jpg"

                cv2.imwrite(str(arquivo), rosto)

                print(f"Imagem salva: {arquivo.name}")

            else:
                print("Posicione apenas um rosto na câmera.")

        elif tecla == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print(f"Total de imagens: {contador}")


if __name__ == "__main__":
    capturar_rostos()
