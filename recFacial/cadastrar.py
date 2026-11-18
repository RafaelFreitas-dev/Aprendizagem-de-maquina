import cv2
import os
import time

# ==========================================
# CONFIGURAÇÕES
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA_FACES = os.path.join(BASE_DIR, "banco_faces")

ARQUIVO_PROTO = os.path.join(
    BASE_DIR,
    "deploy.prototxt.txt"
)

ARQUIVO_MODELO = os.path.join(
    BASE_DIR,
    "res10_300x300_ssd_iter_140000.caffemodel"
)

MAX_FOTOS = 30
INTERVALO = 1

CONFIANCA_MINIMA = 0.40

LARGURA_FACE = 90
ALTURA_FACE = 120

TAMANHO_MINIMO_FACE = 35


# ==========================================
# CADASTRO DA PESSOA
# ==========================================

nome = input("Digite o nome da pessoa: ").strip()

if not nome:
    print("Nome inválido!")
    raise SystemExit

# Evita criar pastas com caracteres problemáticos
nome = "".join(
    caractere for caractere in nome
    if caractere not in r'\/:*?"<>|'
).strip()

if not nome:
    print("Nome inválido!")
    raise SystemExit

pasta_pessoa = os.path.join(PASTA_FACES, nome)

os.makedirs(pasta_pessoa, exist_ok=True)


# ==========================================
# CARREGAR DETECTOR SSD
# ==========================================

detector = cv2.dnn.readNetFromCaffe(
    ARQUIVO_PROTO,
    ARQUIVO_MODELO
)


# ==========================================
# INICIAR WEBCAM
# ==========================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("Não foi possível abrir a câmera!")
    raise SystemExit

print("\nIniciando cadastro facial...")
print("Mantenha apenas um rosto na imagem.")
print("Pressione Q para sair.\n")


# ==========================================
# CONTADOR DE FOTOS
# ==========================================

extensoes = (".jpg", ".jpeg", ".png")

contador = len([
    arquivo
    for arquivo in os.listdir(pasta_pessoa)
    if arquivo.lower().endswith(extensoes)
])

ultimo_cadastro = 0


# ==========================================
# LOOP PRINCIPAL
# ==========================================

while True:

    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao capturar imagem.")
        break

    altura, largura = frame.shape[:2]

    imagem = cv2.resize(frame, (300, 300))

    blob = cv2.dnn.blobFromImage(
        imagem,
        1.0,
        (300, 300),
        (104, 117, 123)
    )

    detector.setInput(blob)

    deteccoes = detector.forward()

    rostos = []

    for i in range(deteccoes.shape[2]):

        confianca = deteccoes[0, 0, i, 2]

        if confianca < CONFIANCA_MINIMA:
            continue

        caixa = deteccoes[0, 0, i, 3:7] * [
            largura,
            altura,
            largura,
            altura
        ]

        x1, y1, x2, y2 = caixa.astype("int")

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(largura, x2)
        y2 = min(altura, y2)

        largura_rosto = x2 - x1
        altura_rosto = y2 - y1

        if (
            largura_rosto >= TAMANHO_MINIMO_FACE
            and altura_rosto >= TAMANHO_MINIMO_FACE
        ):
            rostos.append((x1, y1, x2, y2))

    # ======================================
    # CAPTURAR APENAS UM ROSTO
    # ======================================

    if len(rostos) == 1:

        x1, y1, x2, y2 = rostos[0]

        rosto = frame[y1:y2, x1:x2]

        rosto = cv2.cvtColor(
            rosto,
            cv2.COLOR_BGR2GRAY
        )

        rosto = cv2.resize(
            rosto,
            (LARGURA_FACE, ALTURA_FACE)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Rosto detectado",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        agora = time.monotonic()

        if (
            contador < MAX_FOTOS
            and agora - ultimo_cadastro >= INTERVALO
        ):

            contador += 1

            caminho = os.path.join(
                pasta_pessoa,
                f"rosto_{contador:03d}.jpg"
            )

            cv2.imwrite(caminho, rosto)

            ultimo_cadastro = agora

            print(f"Foto {contador}/{MAX_FOTOS} cadastrada!")

    elif len(rostos) > 1:

        cv2.putText(
            frame,
            "Apenas uma pessoa na camera",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Nenhum rosto detectado",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # ======================================
    # EXIBIR CONTADOR
    # ======================================

    cv2.putText(
        frame,
        f"Fotos: {contador}/{MAX_FOTOS}",
        (20, altura - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Cadastro Facial", frame)

    # ======================================
    # FINALIZAR
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if contador >= MAX_FOTOS:
        print("\nCadastro concluído!")
        break


# ==========================================
# LIBERAR RECURSOS
# ==========================================

camera.release()
cv2.destroyAllWindows()

print(f"\nCadastro finalizado: {contador} fotos.")
