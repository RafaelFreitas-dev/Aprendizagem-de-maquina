
import cv2
import os
import numpy as np
import pickle

PASTA = "banco_faces"

faces = []
ids = []

nomes = {}

detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

contador_id = 0

for nome in os.listdir(PASTA):

    caminho_pessoa = os.path.join(PASTA, nome)

    if not os.path.isdir(caminho_pessoa):
        continue

    contador_id += 1
    nomes[nome] = contador_id

    for arquivo in os.listdir(caminho_pessoa):

        caminho = os.path.join(caminho_pessoa, arquivo)

        imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

        if imagem is None:
            continue

        imagem = cv2.resize(imagem, (90, 120))

        faces.append(imagem)
        ids.append(contador_id)

if len(faces) == 0:
    raise RuntimeError("Nenhuma imagem encontrada!")

reconhecedor = cv2.face.LBPHFaceRecognizer_create()

reconhecedor.train(
    faces,
    np.array(ids, dtype=np.int32)
)

reconhecedor.write("lbph_classifier.yml")

with open("face_names.pickle", "wb") as arquivo:
    pickle.dump(nomes, arquivo)

print("Treinamento concluído!")
print("Pessoas cadastradas:", nomes)
print("Quantidade de imagens:", len(faces))
