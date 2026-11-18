
import cv2
import numpy as np
import os
import pickle
import sys

from datetime import datetime
from helper_functions import resize_video


# ==========================================
# CAMINHOS DOS ARQUIVOS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARQUIVO_TREINAMENTO = os.path.join(
    BASE_DIR, "lbph_classifier.yml"
)

ARQUIVO_NOMES = os.path.join(
    BASE_DIR, "face_names.pickle"
)

ARQUIVO_PROTO = os.path.join(
    BASE_DIR, "deploy.prototxt.txt"
)

ARQUIVO_MODELO = os.path.join(
    BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel"
)

ARQUIVO_LOG = os.path.join(
    BASE_DIR, "meuarquivo.txt"
)


# ==========================================
# CONFIGURAÇÕES
# ==========================================

LARGURA_FACE = 90
ALTURA_FACE = 120

# Distância máxima do LBPH.
# Quanto menor a distância, melhor a correspondência.
THRESHOLD = 80

# Confiança mínima do detector SSD.
CONFIANCA_MINIMA = 0.40

MAX_WIDTH = 800


# ==========================================
# VERIFICAR ARQUIVOS
# ==========================================

arquivos = [
    ARQUIVO_TREINAMENTO,
    ARQUIVO_NOMES,
    ARQUIVO_PROTO,
    ARQUIVO_MODELO
]

for arquivo in arquivos:
    if not os.path.exists(arquivo):
        print(f"Arquivo não encontrado: {arquivo}")
        sys.exit()


# ==========================================
# CARREGAR RECONHECEDOR LBPH
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(ARQUIVO_TREINAMENTO)


# ==========================================
# CARREGAR NOMES
# ==========================================

with open(ARQUIVO_NOMES, "rb") as f:
    original_labels = pickle.load(f)

# Aceita tanto {nome: id} quanto {id: nome}.
if not isinstance(original_labels, dict):
    print("Formato inválido no arquivo face_names.pickle")
    sys.exit()

if all(isinstance(v, (int, np.integer)) for v in original_labels.values()):
    face_names = {
        int(v): str(k)
        for k, v in original_labels.items()
    }
else:
    face_names = {
        int(k): str(v)
        for k, v in original_labels.items()
    }

print("Pessoas carregadas:")
for pessoa_id, nome in face_names.items():
    print(f"ID {pessoa_id}: {nome}")


# ==========================================
# CARREGAR DETECTOR SSD
# ==========================================

network = cv2.dnn.readNetFromCaffe(
    ARQUIVO_PROTO,
    ARQUIVO_MODELO
)


# ==========================================
# RECONHECER ROSTOS
# ==========================================

def recognize_faces(frame):

    resultado_frame = frame.copy()

    gray = cv2.cvtColor(
        resultado_frame,
        cv2.COLOR_BGR2GRAY
    )

    altura, largura = gray.shape

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 117.0, 123.0)
    )

    network.setInput(blob)
    detections = network.forward()

    ultimo_nome = ""

    for i in range(detections.shape[2]):

        confianca_detector = detections[0, 0, i, 2]

        if confianca_detector < CONFIANCA_MINIMA:
            continue

        bbox = detections[0, 0, i, 3:7] * np.array([
            largura,
            altura,
            largura,
            altura
        ])

        x1, y1, x2, y2 = bbox.astype("int")

        # Limitar coordenadas ao tamanho da imagem.
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(largura, x2)
        y2 = min(altura, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Recortar o rosto.
        face_roi = gray[y1:y2, x1:x2]

        # Mesmo tamanho utilizado no cadastro.
        face_roi = cv2.resize(
            face_roi,
            (LARGURA_FACE, ALTURA_FACE)
        )

        # Previsão LBPH.
        prediction, distancia = recognizer.predict(face_roi)

        nome = face_names.get(
            int(prediction),
            "ID desconhecido"
        )

        if nome != "ID desconhecido" and distancia <= THRESHOLD:
            texto = f"{nome} -> {distancia:.2f}"
            cor = (0, 255, 0)
            ultimo_nome = nome
        else:
            texto = f"Nao identificado -> {distancia:.2f}"
            cor = (0, 0, 255)

        cv2.rectangle(
            resultado_frame,
            (x1, y1),
            (x2, y2),
            cor,
            2
        )

        cv2.putText(
            resultado_frame,
            texto,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            cor,
            2
        )

    return resultado_frame, ultimo_nome


# ==========================================
# GRAVAR IDENTIFICAÇÃO
# ==========================================

def grava(nome):

    if not nome:
        print("Nenhuma pessoa identificada para registrar.")
        return

    agora = datetime.now()

    data_hora = agora.strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    conteudo = f"{data_hora} - {nome}\n"

    with open(ARQUIVO_LOG, "a", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)

    print(f"Registro salvo: {conteudo.strip()}")


# ==========================================
# INICIAR WEBCAM
# ==========================================

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Erro: não foi possível acessar a webcam.")
    sys.exit()

ultimo_nome_reconhecido = ""


# ==========================================
# LOOP PRINCIPAL
# ==========================================

while True:

    ret, frame = cam.read()

    if not ret:
        print("Erro ao capturar imagem.")
        break

    if MAX_WIDTH is not None and frame.shape[1] > MAX_WIDTH:
        video_width, video_height = resize_video(
            frame.shape[1],
            frame.shape[0],
            MAX_WIDTH
        )

        frame = cv2.resize(
            frame,
            (video_width, video_height)
        )

    processed_frame, nome = recognize_faces(frame)

    if nome:
        ultimo_nome_reconhecido = nome

    cv2.imshow(
        "Reconhecimento Facial",
        processed_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        grava(ultimo_nome_reconhecido)
        break


# ==========================================
# FINALIZAR
# ==========================================

cam.release()
cv2.destroyAllWindows()

print("Programa finalizado!")
