import csv
import io
import logging
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
import cv2
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ELA_QUALIDADE = 90
ELA_FATOR_AMPLIFICACAO = 15
ELA_VALOR_MAX = 255

PROJETO = Path(__file__).parent.resolve()

COLUNAS = [
    "arquivo",
    "rotulo",
    "ela_media",
    "ela_desvio",
    "variancia_ruido",
    "fft_simetria",
    "corr_rg",
    "corr_rb",
    "corr_gb",
    "aberracao_cromatica",
    "gradiente_media",
    "gradiente_desvio",
]


random.seed(42)

def _ela_features(conteudo):
    try:
        imagem_original = Image.open(io.BytesIO(conteudo)).convert("RGB")

        buffer_recomp = io.BytesIO()
        imagem_original.save(buffer_recomp, format="JPEG", quality=ELA_QUALIDADE)
        buffer_recomp.seek(0)
        imagem_recomprimida = Image.open(buffer_recomp).convert("RGB")

        diferenca = ImageChops.difference(imagem_original, imagem_recomprimida)
        arr = np.array(diferenca, dtype=np.float32)
        arr = np.clip(arr * ELA_FATOR_AMPLIFICACAO, 0, ELA_VALOR_MAX)

        return float(np.mean(arr)), float(np.std(arr))
    except Exception:
        return 0.0, 0.0


def _ruido_variancia(img_cinza):
    kernel_srm = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4.0
    residuo = cv2.filter2D(img_cinza, -1, kernel_srm)

    tamanho_bloco = 16
    h, w = residuo.shape
    n_linhas = h // tamanho_bloco
    n_colunas = w // tamanho_bloco

    if n_linhas == 0 or n_colunas == 0:
        return 0.0

    variancias = []
    for idx_linha in range(n_linhas):
        for idx_coluna in range(n_colunas):
            r = idx_linha * tamanho_bloco
            c = idx_coluna * tamanho_bloco
            bloco = residuo[r:r + tamanho_bloco, c:c + tamanho_bloco]
            variancias.append(float(np.var(bloco)))

    return float(np.mean(variancias))


def _fft_simetria(img_cinza):
    f = np.fft.fft2(img_cinza.astype(np.float32))
    f_deslocado = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(f_deslocado))

    h, w = magnitude.shape
    metade_superior = magnitude[:h // 2, :]
    metade_inferior = np.flipud(magnitude[h // 2:, :])
    min_h = min(metade_superior.shape[0], metade_inferior.shape[0])
    diff = np.abs(metade_superior[:min_h] - metade_inferior[:min_h])
    simetria = float(1.0 - (np.mean(diff) / (np.max(magnitude) + 1e-9)))

    return simetria


def _correlacao_rgb(img_cv):
    b, g, r = cv2.split(img_cv.astype(np.float32))

    def corr(a, b_):
        a_plano = a.flatten()
        b_plano = b_.flatten()
        if np.std(a_plano) < 1e-9 or np.std(b_plano) < 1e-9:
            return 1.0
        return float(np.corrcoef(a_plano, b_plano)[0, 1])

    return corr(r, g), corr(r, b), corr(g, b)


def _aberracao_cromatica(img_cv):
    img_suavizada = cv2.GaussianBlur(img_cv, (3, 3), 0)
    b, g, r = cv2.split(img_suavizada)

    bordas_r = cv2.Canny(r, 30, 90)
    bordas_g = cv2.Canny(g, 30, 90)
    bordas_b = cv2.Canny(b, 30, 90)

    mascara = (bordas_r > 0) | (bordas_g > 0) | (bordas_b > 0)
    total_pixels = float(np.count_nonzero(mascara)) + 1e-9

    diff_rg = np.count_nonzero(cv2.bitwise_xor(bordas_r, bordas_g) & mascara)
    diff_rb = np.count_nonzero(cv2.bitwise_xor(bordas_r, bordas_b) & mascara)
    diff_gb = np.count_nonzero(cv2.bitwise_xor(bordas_g, bordas_b) & mascara)

    desalinhamento_total = float(diff_rg + diff_rb + diff_gb)
    return desalinhamento_total / (2.0 * total_pixels)


def _gradiente_features(img_cv):
    img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
    canal_y, _, _ = cv2.split(img_yuv)

    sobel_x = cv2.Sobel(canal_y, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(canal_y, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    return float(np.mean(magnitude)), float(np.std(magnitude))


def extrair_de_imagem(caminho):
    try:
        caminho = Path(caminho)
        if not caminho.exists():
            return None
        conteudo = caminho.read_bytes()

        imagem_np = np.frombuffer(conteudo, np.uint8)
        img_cv = cv2.imdecode(imagem_np, cv2.IMREAD_COLOR)
        if img_cv is None:
            return None

        img_cv = cv2.resize(img_cv, (256, 256))
        conteudo_redim = cv2.imencode(".jpg", img_cv)[1].tobytes()

        img_cinza = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        ela_media, ela_desvio = _ela_features(conteudo_redim)
        variancia = _ruido_variancia(img_cinza)
        simetria = _fft_simetria(img_cinza)
        corr_rg, corr_rb, corr_gb = _correlacao_rgb(img_cv)
        aber = _aberracao_cromatica(img_cv)
        grad_media, grad_desvio = _gradiente_features(img_cv)

        return [
            ela_media,
            ela_desvio,
            variancia,
            simetria,
            corr_rg,
            corr_rb,
            corr_gb,
            aber,
            grad_media,
            grad_desvio,
        ]
    except Exception as exc:
        logger.warning("Falha ao processar %s: %s", caminho.name if hasattr(caminho, 'name') else caminho, exc)
        return None


def salvar_features_csv(saida_csv, amostras, base_relativa):
    """
    Grava as amostras de imagens extraídas em um arquivo CSV.
    amostras: lista de tuplas (caminho_imagem, rotulo)
    base_relativa: Path usado para calcular o caminho relativo
    """
    saida_csv = Path(saida_csv)
    saida_csv.parent.mkdir(parents=True, exist_ok=True)
    base_relativa = Path(base_relativa)

    with open(saida_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(COLUNAS)

        for caminho, rotulo in tqdm(amostras, desc=f"Salvando em {saida_csv.name}"):
            features = extrair_de_imagem(caminho)
            if features is None:
                continue
            
            try:
                caminho_relativo = caminho.relative_to(base_relativa).as_posix()
            except ValueError:
                caminho_relativo = caminho.name

            linha = [caminho_relativo, rotulo] + [round(v, 6) for v in features]
            escritor.writerow(linha)


def obter_imagens_diretorio(pasta, extensoes=None):
    if extensoes is None:
        extensoes = [".jpg", ".jpeg", ".png", ".webp"]
    pasta = Path(pasta)
    if not pasta.exists():
        return []
    return [p for p in pasta.rglob("*") if p.is_file() and p.suffix.lower() in extensoes]


def processar_dataset(dataset_id, max_por_classe=2000):
    """
    Controla o fluxo de listagem, split e extração de características dos 4 datasets.
    """
    logger.info("==========================================")
    logger.info("Processando Dataset %d (Limite por classe: %d)", dataset_id, max_por_classe)
    logger.info("==========================================")

    if dataset_id == 1:

        base_dir = PROJETO / "archive" / "Data Set 1" / "Data Set 1"
        saida_dir = PROJETO / "archive"
        if not base_dir.exists():
            logger.error("Pasta do Dataset 1 nao encontrada: %s", base_dir)
            return

        splits_map = {
            "train": ("train", "features_train_1.csv"),
            "validation": ("validation", "features_valid_1.csv"),
            "test": ("test", "features_test_1.csv")
        }

        for split_nome, (pasta_split, csv_nome) in splits_map.items():
            reais = obter_imagens_diretorio(base_dir / pasta_split / "real")
            fakes = obter_imagens_diretorio(base_dir / pasta_split / "fake")
            
            logger.info("Dataset 1 [%s] Encontrados -> Reais: %d, Fakes: %d", split_nome, len(reais), len(fakes))
            

            random.shuffle(reais)
            random.shuffle(fakes)
            
            reais_selecionados = reais[:max_por_classe]
            fakes_selecionados = fakes[:max_por_classe]
            
            amostras = [(r, 0) for r in reais_selecionados] + [(f, 1) for f in fakes_selecionados]
            random.shuffle(amostras)
            
            salvar_features_csv(saida_dir / csv_nome, amostras, saida_dir)

    elif dataset_id == 2:

        base_dir = PROJETO / "archive (1)" / "140k Real and Fake Face-ela"
        saida_dir = PROJETO / "archive (1)"
        if not base_dir.exists():
            logger.error("Pasta do Dataset 2 nao encontrada: %s", base_dir)
            return

        splits_map = {
            "train": ("Train", "features_train_2.csv"),
            "validation": ("Validation", "features_valid_2.csv"),
            "test": ("Test", "features_test_2.csv")
        }

        for split_nome, (pasta_split, csv_nome) in splits_map.items():
            reais = obter_imagens_diretorio(base_dir / pasta_split / "real")
            fakes = obter_imagens_diretorio(base_dir / pasta_split / "fake")
            
            logger.info("Dataset 2 [%s] Encontrados -> Reais: %d, Fakes: %d", split_nome, len(reais), len(fakes))
            

            random.shuffle(reais)
            random.shuffle(fakes)
            
            reais_selecionados = reais[:max_por_classe]
            fakes_selecionados = fakes[:max_por_classe]
            
            amostras = [(r, 0) for r in reais_selecionados] + [(f, 1) for f in fakes_selecionados]
            random.shuffle(amostras)
            
            salvar_features_csv(saida_dir / csv_nome, amostras, saida_dir)

    elif dataset_id == 3:

        base_dir = PROJETO / "archive (2)"
        saida_dir = PROJETO / "archive (2)"
        if not base_dir.exists():
            logger.error("Pasta do Dataset 3 nao encontrada: %s", base_dir)
            return

        reais = obter_imagens_diretorio(base_dir / "real_dataset")
        fakes = obter_imagens_diretorio(base_dir / "Ai_generated_dataset")
        
        logger.info("Dataset 3 Encontrados -> Reais: %d, Fakes: %d", len(reais), len(fakes))

        random.shuffle(reais)
        random.shuffle(fakes)


        reais = reais[:max_por_classe]
        fakes = fakes[:max_por_classe]


        def dividir_lista(lista):
            n = len(lista)
            n_train = int(n * 0.70)
            n_valid = int(n * 0.15)
            return lista[:n_train], lista[n_train:n_train + n_valid], lista[n_train + n_valid:]

        reais_train, reais_valid, reais_test = dividir_lista(reais)
        fakes_train, fakes_valid, fakes_test = dividir_lista(fakes)

        splits_dados = {
            "features_train_3.csv": (reais_train, fakes_train),
            "features_valid_3.csv": (reais_valid, fakes_valid),
            "features_test_3.csv": (reais_test, fakes_test)
        }

        for csv_nome, (r_lista, f_lista) in splits_dados.items():
            amostras = [(r, 0) for r in r_lista] + [(f, 1) for f in f_lista]
            random.shuffle(amostras)
            salvar_features_csv(saida_dir / csv_nome, amostras, saida_dir)

    elif dataset_id == 4:

        base_dir = PROJETO / "archive (3)" / "AI-face-detection-Dataset"
        saida_dir = PROJETO / "archive (3)"
        if not base_dir.exists():
            logger.error("Pasta do Dataset 4 nao encontrada: %s", base_dir)
            return

        reais = obter_imagens_diretorio(base_dir / "real")
        fakes = obter_imagens_diretorio(base_dir / "AI")
        
        logger.info("Dataset 4 Encontrados -> Reais: %d, Fakes: %d", len(reais), len(fakes))

        random.shuffle(reais)
        random.shuffle(fakes)


        reais = reais[:max_por_classe]
        fakes = fakes[:max_por_classe]


        def dividir_lista(lista):
            n = len(lista)
            n_train = int(n * 0.70)
            n_valid = int(n * 0.15)
            return lista[:n_train], lista[n_train:n_train + n_valid], lista[n_train + n_valid:]

        reais_train, reais_valid, reais_test = dividir_lista(reais)
        fakes_train, fakes_valid, fakes_test = dividir_lista(fakes)

        splits_dados = {
            "features_train_4.csv": (reais_train, fakes_train),
            "features_valid_4.csv": (reais_valid, fakes_valid),
            "features_test_4.csv": (reais_test, fakes_test)
        }

        for csv_nome, (r_lista, f_lista) in splits_dados.items():
            amostras = [(r, 0) for r in r_lista] + [(f, 1) for f in f_lista]
            random.shuffle(amostras)
            salvar_features_csv(saida_dir / csv_nome, amostras, saida_dir)

    logger.info("Features extraidas com sucesso para o Dataset %d.", dataset_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python extrair_features.py [1|2|3|4|todos] [limite_por_classe]")
        print("Exemplo: python extrair_features.py 3 2000")
        sys.exit(1)

    selecao = sys.argv[1]
    limite = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    if selecao == "todos":
        for i in [1, 2, 3, 4]:
            processar_dataset(i, limite)
    else:
        try:
            ds_id = int(selecao)
            if ds_id in [1, 2, 3, 4]:
                processar_dataset(ds_id, limite)
            else:
                print("ID de dataset invalido. Escolha de 1 a 4.")
        except ValueError:
            print("Selecao invalida. Use um numero de 1 a 4 ou 'todos'.")
