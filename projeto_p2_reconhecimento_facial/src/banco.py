
import sqlite3
from pathlib import Path
from datetime import datetime

# Localização do banco de dados
RAIZ = Path(__file__).resolve().parents[1]
PASTA_BANCO = RAIZ / "database"

PASTA_BANCO.mkdir(parents=True, exist_ok=True)

CAMINHO_BANCO = PASTA_BANCO / "presencas.db"


def conectar():
    return sqlite3.connect(CAMINHO_BANCO)


def inicializar_banco():

    conexao = conectar()
    cursor = conexao.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alunos (
            codigo TEXT PRIMARY KEY,
            nome TEXT NOT NULL,
            data_cadastro TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS presencas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codigo TEXT NOT NULL,
            data_hora TEXT NOT NULL,
            FOREIGN KEY (codigo) REFERENCES alunos(codigo)
        )
    """)

    conexao.commit()
    conexao.close()

    print("Banco de dados inicializado!")


def cadastrar_aluno(codigo, nome):

    inicializar_banco()

    conexao = conectar()
    cursor = conexao.cursor()

    cursor.execute(
        "SELECT nome FROM alunos WHERE codigo = ?",
        (codigo,)
    )

    aluno = cursor.fetchone()

    if aluno:
        conexao.close()

        return aluno[0] == nome

    data = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    cursor.execute("""
        INSERT INTO alunos (codigo, nome, data_cadastro)
        VALUES (?, ?, ?)
    """, (codigo, nome, data))

    conexao.commit()
    conexao.close()

    print(f"Aluno {nome} cadastrado!")

    return True


def listar_alunos():

    conexao = conectar()
    cursor = conexao.cursor()

    cursor.execute("""
        SELECT codigo, nome FROM alunos
        ORDER BY codigo
    """)

    alunos = cursor.fetchall()

    conexao.close()

    return alunos


if __name__ == "__main__":
    inicializar_banco()
    print(listar_alunos())
