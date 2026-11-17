
import cv2
import os

nome = str(input("Digite o nome da pessoa a ser cadastrada: "))

NOME = nome

PASTA = os.path.join("banco_faces", NOME)

os.makedirs(PASTA, exist_ok=True)

detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

contador = len(os.listdir(PASTA))

print("Cadastro iniciado.")
print("Pressione ESPACO para capturar uma imagem.")
print("Pressione Q para sair.")

while True:

    conectado, frame = camera.read()

    if not conectado:
        break

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        cinza,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Fotos cadastradas: {contador}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Cadastro Facial", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord(' '):

        if len(faces) == 1:

            x, y, w, h = faces[0]

            rosto = cinza[y:y+h, x:x+w]

            rosto = cv2.resize(rosto, (90, 120))

            contador += 1

            caminho = os.path.join(
                PASTA,
                f"rosto_{contador:03d}.jpg"
            )

            cv2.imwrite(caminho, rosto)

            print("Imagem salva:", caminho)

        else:
            print("Mantenha apenas um rosto na câmera.")

    elif tecla == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()