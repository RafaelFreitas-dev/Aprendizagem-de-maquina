
import cv2
import numpy as np
import os
import pickle
import sys

from datetime import datetime
from helper_functions import resize_video


# ==========================================
# CONFIGURACOES
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

recognizer = "lbph"

training_data = os.path.join(
    BASE_DIR,
    "lbph_classifier.yml"
)

CAMINHO_NOMES = os.path.join(
    BASE_DIR,
    "face_names.pickle"
)

threshold = 70

max_width = 800


# ==========================================
# VERIFICAR ARQUIVOS
# ==========================================

arquivos_necessarios = [
    training_data,
    CAMINHO_NOMES,
    os.path.join(BASE_DIR, "deploy.prototxt.txt"),
    os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
]

for arquivo in arquivos_necessarios:

    if not os.path.exists(arquivo):
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {arquivo}"
        )


# ==========================================
# CARREGAR RECONHECEDOR
# ==========================================

def load_recognizer(option, training_data):

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "Modulo cv2.face nao encontrado. "
            "Instale opencv-contrib-python."
        )

    if option == "eigenfaces":
        face_classifier = cv2.face.EigenFaceRecognizer_create()

    elif option == "fisherfaces":
        face_classifier = cv2.face.FisherFaceRecognizer_create()

    elif option == "lbph":
        face_classifier = cv2.face.LBPHFaceRecognizer_create()

    else:
        raise ValueError("Algoritmo de reconhecimento invalido.")

    face_classifier.read(training_data)

    return face_classifier


face_classifier = load_recognizer(
    recognizer,
    training_data
)


# ==========================================
# CARREGAR NOMES
# ==========================================

with open(CAMINHO_NOMES, "rb") as arquivo:

    original_labels = pickle.load(arquivo)

    face_names = {
        valor: chave
        for chave, valor in original_labels.items()
    }


# ==========================================
# REGISTRAR IDENTIFICACAO
# ==========================================

def grava(nome):

    nome_arquivo = os.path.join(
        BASE_DIR,
        "meuarquivo.txt"
    )

    agora = datetime.now()

    data_hora = agora.strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    conteudo = f"{data_hora} - {nome}\n"

    with open(
        nome_arquivo,
        "a",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(conteudo)

    print("Identificacao registrada:", conteudo.strip())


# ==========================================
# RECONHECIMENTO FACIAL SSD + LBPH
# ==========================================

def recognize_faces(
    network,
    face_classifier,
    orig_frame,
    face_names,
    threshold,
    conf_min=0.9
):

    frame = orig_frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 117.0, 123.0)
    )

    network.setInput(blob)

    detections = network.forward()

    nome_identificado = None

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < conf_min:
            continue

        bbox = detections[0, 0, i, 3:7] * np.array(
            [w, h, w, h]
        )

        start_x, start_y, end_x, end_y = bbox.astype("int")

        # Ajustar coordenadas aos limites do frame
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w, end_x)
        end_y = min(h, end_y)

        if end_x <= start_x or end_y <= start_y:
            continue

        face_roi = gray[
            start_y:end_y,
            start_x:end_x
        ]

        if face_roi.size == 0:
            continue

        face_roi = cv2.resize(
            face_roi,
            (90, 120)
        )

        prediction, conf = face_classifier.predict(
            face_roi
        )

        if conf <= threshold:

            pred_name = face_names.get(
                prediction,
                "Not identified"
            )

            if pred_name != "Not identified":
                nome_identificado = pred_name

        else:

            pred_name = "Not identified"

        texto = f"{pred_name} -> {conf:.2f}"

        cor = (
            (0, 255, 0)
            if pred_name != "Not identified"
            else (0, 0, 255)
        )

        cv2.rectangle(
            frame,
            (start_x, start_y),
            (end_x, end_y),
            cor,
            2
        )

        cv2.putText(
            frame,
            texto,
            (start_x, max(20, start_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            cor,
            2
        )

    return frame, nome_identificado


# ==========================================
# CARREGAR DETECTOR SSD
# ==========================================

network = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, "deploy.prototxt.txt"),
    os.path.join(
        BASE_DIR,
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
)


# ==========================================
# INICIAR CAMERA
# ==========================================

cam = cv2.VideoCapture(0)

if not cam.isOpened():

    print("Erro: nao foi possivel acessar a webcam.")

    sys.exit()


print("Reconhecimento facial iniciado.")
print("Pressione Q para registrar e sair.")


# ==========================================
# LOOP PRINCIPAL
# ==========================================

ultimo_nome = None

while True:

    ret, frame = cam.read()

    if not ret:
        print("Erro ao capturar imagem.")
        break

    if max_width is not None:

        video_width, video_height = resize_video(
            frame.shape[1],
            frame.shape[0],
            max_width
        )

        frame = cv2.resize(
            frame,
            (video_width, video_height)
        )

    processed_frame, nome_identificado = recognize_faces(
        network,
        face_classifier,
        frame,
        face_names,
        threshold
    )

    if nome_identificado is not None:

        ultimo_nome = nome_identificado

    cv2.imshow(
        "Recognizing Faces",
        processed_frame
    )

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):

        if ultimo_nome is not None:

            grava(ultimo_nome)

        else:

            print("Nenhuma pessoa identificada para registrar.")

        break


print("Finished!")

cam.release()
cv2.destroyAllWindows()
