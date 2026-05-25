from pathlib import Path
from flask import Flask, request, jsonify, send_file
from functools import wraps
from dotenv import load_dotenv
from main import (
    extrair_metadados,
    categorizar_metadados,
    ela,
    analise_forense,
)
from modelozudo import classificar_bancada
import pandas as pd
load_dotenv()



app = Flask(__name__)
BASE_DIR = Path(__file__).parent.resolve()



LINK_FORMULARIO = "https://docs.google.com/forms/d/e/1FAIpQLSfZ0Ii22rdkT3oJKhd5z-GRRfMdxGzuf4rc1LvABdekFj7bfQ/viewform"     
CAMINHO_PLANILHA = BASE_DIR / "planilha.xlsx"    

TIPOS_IMAGEM = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".tiff", ".tif", ".gif", ".bmp", ".raw",
})


def require_file(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "file" not in request.files:
            return jsonify({"erro": "Nenhum arquivo enviado."}), 400

        arquivo = request.files["file"]
        if not arquivo.filename:
            return jsonify({"erro": "Nome de arquivo vazio."}), 400

        conteudo = arquivo.read()
        nome = arquivo.filename
        return f(conteudo, nome, *args, **kwargs)
    return decorated_function


@app.route("/")
def index():
    return send_file(BASE_DIR / "app.html")


@app.route("/assets/<nome_arquivo>")
def servir_asset(nome_arquivo):
    caminho = BASE_DIR / nome_arquivo
    if caminho.exists():
        return send_file(caminho)
    return "", 404


@app.route("/analyze/quick", methods=["POST"])
@require_file
def analyze_quick(conteudo, nome):
    try:
        exif = extrair_metadados(conteudo, nome)
        categorias = categorizar_metadados(exif)
        return jsonify({"exif": exif, "categorias": categorias})
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    except Exception as e:
        return jsonify({"erro": f"Erro inesperado: {e}"}), 500


@app.route("/analyze/deep", methods=["POST"])
@require_file
def analyze_deep(conteudo, nome):
    try:
        sufixo = Path(nome).suffix.lower()

        if sufixo not in TIPOS_IMAGEM:
            return jsonify({"erro": "Analise profunda disponivel apenas para imagens."}), 400

        exif = extrair_metadados(conteudo, nome)
        categorias = categorizar_metadados(exif)
        ela_b64 = ela(conteudo, sufixo)
        forense = analise_forense(conteudo)

        return jsonify({
            "exif": exif,
            "categorias": categorias,
            "ela": ela_b64,
            "forense": forense,
        })
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    except Exception as e:
        return jsonify({"erro": f"Erro inesperado: {e}"}), 500


@app.route("/analyze/full", methods=["POST"])
@require_file
def analyze_full(conteudo, nome):
    try:
        sufixo = Path(nome).suffix.lower()

        if sufixo not in TIPOS_IMAGEM:
            return jsonify({"erro": "Analise completa disponivel apenas para imagens."}), 400

        exif = extrair_metadados(conteudo, nome)
        categorias = categorizar_metadados(exif)

        ela_b64 = ela(conteudo, sufixo)
        forense = analise_forense(conteudo)

        features = [
            forense.get("ela_media", 0.0),
            forense.get("ela_desvio", 0.0),
            forense.get("variancia_ruido", 0.0),
            forense.get("fft_simetria", 0.0),
            forense.get("correlacao_rgb", {}).get("rg", 0.0),
            forense.get("correlacao_rgb", {}).get("rb", 0.0),
            forense.get("correlacao_rgb", {}).get("gb", 0.0),
            forense.get("aberracao_cromatica", 0.0),
            forense.get("gradiente_media", 0.0),
            forense.get("gradiente_desvio", 0.0),
        ]

        bancada = classificar_bancada(conteudo, features)

        return jsonify({
            "exif": exif,
            "categorias": categorias,
            "ela": ela_b64,
            "forense": forense,
            "bancada": bancada
        })
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    except Exception as e:
        return jsonify({"erro": f"Erro inesperado: {e}"}), 500


@app.route("/formulario", methods=["GET"])
def formulario():
    return jsonify({"link": LINK_FORMULARIO})


@app.route("/respostas", methods=["GET"])
def respostas():
    try:
        if not CAMINHO_PLANILHA.exists():
            return jsonify({"erro": "Nenhuma planilha encontrada."}), 404

        df = pd.read_excel(CAMINHO_PLANILHA)
        colunas = list(df.columns)
        dados = df.to_dict(orient="records")

        return jsonify({
            "colunas": colunas,
            "dados": dados,
            "total": len(dados),
        })
    except Exception as e:
        return jsonify({"erro": f"Erro ao ler planilha: {e}"}), 500


if __name__ == "__main__":
    import webbrowser
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
