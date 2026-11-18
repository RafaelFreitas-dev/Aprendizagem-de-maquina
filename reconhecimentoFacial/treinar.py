
import cv2
import os
import numpy as np
import pickle

# Diretório principal do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PASTA = os.path.join(BASE_DIR, "banco_faces")

CAMINHO_CLASSIFICADOR = os.path.join(
    BASE_DIR,
    "lbph_classifier.yml"
)

CAMINHO_NOMES = os.path.join(
    BASE_DIR,
    "face_names.pickle"
)

# Verificar módulo LBPH
if not hasattr(cv2, "face"):
    raise RuntimeError(
        "Modulo cv2.face nao encontrado. "
        "Instale opencv-contrib-python."
    )

# Listas de treinamento
faces = []
ids = []

nomes = {}

contador_id = 0

# Verificar banco de faces
if not os.path.exists(PASTA):
    raise RuntimeError("Pasta banco_faces nao encontrada!")

# Percorrer pessoas cadastradas
for nome in sorted(os.listdir(PASTA)):

    caminho_pessoa = os.path.join(PASTA, nome)

    if not os.path.isdir(caminho_pessoa):
        continue

    contador_id += 1

    nomes[nome] = contador_id

    print(f"Processando: {nome}")

    # Percorrer imagens da pessoa
    for arquivo in sorted(os.listdir(caminho_pessoa)):

        if not arquivo.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        caminho = os.path.join(caminho_pessoa, arquivo)

        imagem = cv2.imread(
            caminho,
            cv2.IMREAD_GRAYSCALE
        )

        if imagem is None:
            print("Imagem invalida:", caminho)
            continue

        imagem = cv2.resize(imagem, (90, 120))

        faces.append(imagem)
        ids.append(contador_id)

# Verificar imagens encontradas
if len(faces) == 0:
    raise RuntimeError("Nenhuma imagem encontrada!")

# Criar reconhecedor LBPH
reconhecedor = cv2.face.LBPHFaceRecognizer_create()

# Treinar modelo
reconhecedor.train(
    faces,
    np.array(ids, dtype=np.int32)
)

# Salvar classificador
reconhecedor.write(CAMINHO_CLASSIFICADOR)

# Salvar nomes e IDs
with open(CAMINHO_NOMES, "wb") as arquivo:
    pickle.dump(nomes, arquivo)

print("\nTreinamento concluido!")
print("Pessoas cadastradas:", nomes)
print("Quantidade de imagens:", len(faces))
print("Classificador salvo:", CAMINHO_CLASSIFICADOR)
