import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJETO = Path(__file__).parent.resolve()

COLUNAS_FEATURES = [
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

NUM_FEATURES_FORENSES = len(COLUNAS_FEATURES)
TAMANHO_IMAGEM = 224
BATCH_SIZE = 32
NUM_EPOCAS = 15
TAXA_APRENDIZADO = 0.00003

TRANSFORMACAO_TREINO = transforms.Compose([
    transforms.Resize((TAMANHO_IMAGEM, TAMANHO_IMAGEM)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORMACAO_AVALIACAO = transforms.Compose([
    transforms.Resize((TAMANHO_IMAGEM, TAMANHO_IMAGEM)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DatasetHibrido(Dataset):
    def __init__(self, conjunto, transformacao, dataset_id, normalizador=None):
        self.transformacao = transformacao
        self.normalizador = normalizador
        self.dataset_id = int(dataset_id)

        if self.dataset_id == 1:
            self.base_features = PROJETO / "archive"
            suffix = "_1"
        elif self.dataset_id == 2:
            self.base_features = PROJETO / "archive (1)"
            suffix = "_2"
        elif self.dataset_id == 3:
            self.base_features = PROJETO / "archive (2)"
            suffix = "_3"
        elif self.dataset_id == 4:
            self.base_features = PROJETO / "archive (3)"
            suffix = "_4"
        else:
            raise ValueError(f"ID de dataset invalido: {dataset_id}")

        split_map = {
            "train": "train",
            "valid": "valid",
            "validation": "valid",
            "test": "test"
        }
        split_key = split_map.get(conjunto, conjunto)
        csv_path = self.base_features / f"features_{split_key}{suffix}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Arquivo CSV nao encontrado: {csv_path}")

        self.df = pd.read_csv(csv_path)

        self.mapa_features = {}
        for _, linha in self.df.iterrows():
            rel_path = linha["arquivo"]
            valores = [float(linha[col]) for col in COLUNAS_FEATURES]
            self.mapa_features[rel_path] = np.array(valores, dtype=np.float32)

        self.amostras = []
        for _, linha in self.df.iterrows():
            rel_path = linha["arquivo"]
            rotulo = int(linha["rotulo"])
            caminho = self.base_features / rel_path
            
            if caminho.exists() and rel_path in self.mapa_features:
                self.amostras.append((caminho, rotulo, rel_path))

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, idx):
        caminho, rotulo, rel_path = self.amostras[idx]

        imagem = Image.open(caminho).convert("RGB")
        imagem = self.transformacao(imagem)

        features = self.mapa_features[rel_path].copy()

        if self.normalizador is not None:
            features = self.normalizador.transform(features.reshape(1, -1)).flatten()

        return imagem, torch.tensor(features, dtype=torch.float32), torch.tensor(rotulo, dtype=torch.float32)


class ModeloHibrido(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        if pretrained:
            self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.cnn = models.efficientnet_b0(weights=None)
        num_features_cnn = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()

        self.camada_forense = nn.Sequential(
            nn.Linear(NUM_FEATURES_FORENSES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        tamanho_fusao = num_features_cnn + 32

        self.classificador = nn.Sequential(
            nn.Linear(tamanho_fusao, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, imagem, features_forenses):
        saida_cnn = self.cnn(imagem)
        saida_forense = self.camada_forense(features_forenses)
        fusao = torch.cat([saida_cnn, saida_forense], dim=1)
        return self.classificador(fusao).squeeze(1)


def criar_normalizador(dataset_id):
    from sklearn.preprocessing import StandardScaler

    dataset_id = int(dataset_id)
    if dataset_id == 1:
        base_features = PROJETO / "archive"
        suffix = "_1"
    elif dataset_id == 2:
        base_features = PROJETO / "archive (1)"
        suffix = "_2"
    elif dataset_id == 3:
        base_features = PROJETO / "archive (2)"
        suffix = "_3"
    elif dataset_id == 4:
        base_features = PROJETO / "archive (3)"
        suffix = "_4"
    else:
        raise ValueError(f"ID de dataset invalido: {dataset_id}")

    csv_treino = base_features / f"features_train{suffix}.csv"
    if not csv_treino.exists():
        raise FileNotFoundError(f"CSV de treino nao encontrado para o normalizador {dataset_id}: {csv_treino}")

    df = pd.read_csv(csv_treino)

    normalizador = StandardScaler()
    normalizador.fit(df[COLUNAS_FEATURES].values)
    
    caminho_normalizador = PROJETO / f"normalizador_{dataset_id}.joblib"
    joblib.dump(normalizador, caminho_normalizador)

    print(f"Normalizador {dataset_id} criado com {len(df)} amostras de treino")
    return normalizador


def treinar(dataset_id):
    dataset_id = int(dataset_id)
    print(f"Dispositivo: {DISPOSITIVO}")
    print(f"Iniciando treinamento do Modelo {dataset_id}")

    normalizador = criar_normalizador(dataset_id)

    dataset_treino = DatasetHibrido("train", TRANSFORMACAO_TREINO, dataset_id, normalizador)
    dataset_valid = DatasetHibrido("valid", TRANSFORMACAO_AVALIACAO, dataset_id, normalizador)

    print(f"Treino: {len(dataset_treino)} amostras")
    print(f"Valid: {len(dataset_valid)} amostras")

    loader_treino = DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    loader_valid = DataLoader(
        dataset_valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    modelo = ModeloHibrido(pretrained=True).to(DISPOSITIVO)
    criterio = nn.BCEWithLogitsLoss()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=TAXA_APRENDIZADO, weight_decay=1e-3)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=NUM_EPOCAS)

    melhor_acuracia = 0.0
    paciencia = 7
    sem_melhora = 0

    caminho_modelo = PROJETO / f"modelo_{dataset_id}.pth"

    for epoca in range(NUM_EPOCAS):
        modelo.train()
        perda_total = 0.0
        acertos = 0
        total = 0

        barra = tqdm(loader_treino, desc=f"Modelo {dataset_id} - Epoca {epoca + 1}/{NUM_EPOCAS}")
        for imagens, features, rotulos in barra:
            imagens = imagens.to(DISPOSITIVO)
            features = features.to(DISPOSITIVO)
            rotulos = rotulos.to(DISPOSITIVO)

            otimizador.zero_grad()
            saidas = modelo(imagens, features)
            perda = criterio(saidas, rotulos)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()

            perda_total += perda.item()
            predicoes = (torch.sigmoid(saidas) > 0.5).float()
            acertos += (predicoes == rotulos).sum().item()
            total += rotulos.size(0)

            barra.set_postfix(perda=f"{perda.item():.4f}", acc=f"{acertos/total:.4f}")

        agendador.step()

        modelo.eval()
        acertos_valid = 0
        total_valid = 0

        with torch.no_grad():
            for imagens, features, rotulos in loader_valid:
                imagens = imagens.to(DISPOSITIVO)
                features = features.to(DISPOSITIVO)
                rotulos = rotulos.to(DISPOSITIVO)

                saidas = modelo(imagens, features)
                predicoes = (torch.sigmoid(saidas) > 0.5).float()
                acertos_valid += (predicoes == rotulos).sum().item()
                total_valid += rotulos.size(0)

        acc_valid = acertos_valid / total_valid if total_valid > 0 else 0
        acc_treino = acertos / total if total > 0 else 0
        print(f"Modelo {dataset_id} | Epoca {epoca + 1} | Treino: {acc_treino:.4f} | Valid: {acc_valid:.4f}")

        if acc_valid > melhor_acuracia:
            melhor_acuracia = acc_valid
            torch.save(modelo.state_dict(), caminho_modelo)
            print(f"--> Modelo {dataset_id} salvo com acuracia: {melhor_acuracia:.4f}")
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                print(f"Early stopping do Modelo {dataset_id} na epoca {epoca + 1}")
                break

    print(f"Treinamento do Modelo {dataset_id} concluido. Melhor acuracia: {melhor_acuracia:.4f}")


def testar(dataset_id):
    dataset_id = int(dataset_id)
    caminho_normalizador = PROJETO / f"normalizador_{dataset_id}.joblib"
    caminho_modelo = PROJETO / f"modelo_{dataset_id}.pth"

    if not caminho_normalizador.exists() or not caminho_modelo.exists():
        print(f"Erro: Arquivos do Modelo {dataset_id} nao encontrados.")
        return

    normalizador = joblib.load(caminho_normalizador)

    dataset_teste = DatasetHibrido("test", TRANSFORMACAO_AVALIACAO, dataset_id, normalizador)
    loader_teste = DataLoader(
        dataset_teste,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    modelo = ModeloHibrido().to(DISPOSITIVO)
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=DISPOSITIVO))
    modelo.eval()

    acertos = 0
    total = 0
    verdadeiros_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    with torch.no_grad():
        for imagens, features, rotulos in tqdm(loader_teste, desc=f"Testando Modelo {dataset_id}"):
            imagens = imagens.to(DISPOSITIVO)
            features = features.to(DISPOSITIVO)
            rotulos = rotulos.to(DISPOSITIVO)

            saidas = modelo(imagens, features)
            predicoes = (torch.sigmoid(saidas) > 0.5).float()

            acertos += (predicoes == rotulos).sum().item()
            total += rotulos.size(0)

            verdadeiros_positivos += ((predicoes == 1) & (rotulos == 1)).sum().item()
            falsos_positivos += ((predicoes == 1) & (rotulos == 0)).sum().item()
            falsos_negativos += ((predicoes == 0) & (rotulos == 1)).sum().item()

    acuracia = acertos / total if total > 0 else 0
    precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos + 1e-9)
    recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos + 1e-9)
    f1 = 2 * (precisao * recall) / (precisao + recall + 1e-9)

    print(f"\n--- Metricas do Modelo {dataset_id} ---")
    print(f"Acuracia: {acuracia:.4f}")
    print(f"Precisao: {precisao:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


def classificar_modelo(conteudo_bytes, features_forenses, dataset_id):
    dataset_id = int(dataset_id)
    caminho_normalizador = PROJETO / f"normalizador_{dataset_id}.joblib"
    caminho_modelo = PROJETO / f"modelo_{dataset_id}.pth"

    if not caminho_normalizador.exists() or not caminho_modelo.exists():
        return None

    normalizador = joblib.load(caminho_normalizador)

    features_array = np.array([features_forenses], dtype=np.float32)
    features_norm = normalizador.transform(features_array)
    features_tensor = torch.tensor(features_norm, dtype=torch.float32).to(DISPOSITIVO)

    imagem = Image.open(__import__("io").BytesIO(conteudo_bytes)).convert("RGB")
    imagem_tensor = TRANSFORMACAO_AVALIACAO(imagem).unsqueeze(0).to(DISPOSITIVO)

    modelo = ModeloHibrido().to(DISPOSITIVO)
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=DISPOSITIVO))
    modelo.eval()

    with torch.no_grad():
        saida = modelo(imagem_tensor, features_tensor)
        probabilidade = torch.sigmoid(saida).item()

    score = round(probabilidade * 100, 2)

    if score >= 70:
        nivel = "Alto"
    elif score >= 40:
        nivel = "Medio"
    else:
        nivel = "Baixo"

    return {
        "score": score,
        "nivel": nivel,
        "probabilidade": round(probabilidade, 4),
    }


def classificar_bancada(conteudo_bytes, features_forenses):
    resultados = {}
    soma_ponderada = 0.0
    soma_pesos = 0.0
    contagem_modelos = 0

    nomes_modelos = {
        1: "IA Geral",
        2: "IA Principal",
        3: "IA Multicategoria",
        4: "IA Face Detection"
    }

    pesos_modelos = {
        1: 1,
        2: 6,
        3: 1,
        4: 1
    }

    for dataset_id in range(1, 5):
        caminho_modelo = PROJETO / f"modelo_{dataset_id}.pth"
        caminho_normalizador = PROJETO / f"normalizador_{dataset_id}.joblib"

        if caminho_modelo.exists() and caminho_normalizador.exists():
            try:
                res_ind = classificar_modelo(conteudo_bytes, features_forenses, dataset_id)
                if res_ind is not None:
                    peso = pesos_modelos[dataset_id]
                    resultados[f"modelo_{dataset_id}"] = {
                        "nome": nomes_modelos[dataset_id],
                        "score": res_ind["score"],
                        "nivel": res_ind["nivel"],
                        "probabilidade": res_ind["probabilidade"],
                        "peso": peso
                    }
                    soma_ponderada += res_ind["score"] * peso
                    soma_pesos += peso
                    contagem_modelos += 1
            except Exception as e:
                logger.error(f"Erro ao classificar com Modelo {dataset_id}: {e}")

    if contagem_modelos > 0 and soma_pesos > 0:
        media_geral = round(soma_ponderada / soma_pesos, 2)
        if media_geral >= 70:
            nivel_geral = "Alto"
        elif media_geral >= 40:
            nivel_geral = "Medio"
        else:
            nivel_geral = "Baixo"
    else:
        media_geral = 0.0
        nivel_geral = "Indisponivel"

    return {
        "modelos": resultados,
        "media_geral": media_geral,
        "nivel_geral": nivel_geral,
        "total_ativos": contagem_modelos
    }


def classificar(conteudo_bytes, features_forenses):
    """
    Função legada para manter compatibilidade retroativa, usa o Modelo 1.
    """
    res = classificar_modelo(conteudo_bytes, features_forenses, 1)
    if res is None:

        return {"score": 0.0, "nivel": "Baixo", "probabilidade": 0.0}
    return res


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python modelozudo.py treinar <1|2|3|4>")
        print("  python modelozudo.py testar <1|2|3|4>")
        print("  python modelozudo.py treinar_todos")
        sys.exit(1)

    comando = sys.argv[1]

    if comando == "treinar":
        if len(sys.argv) < 3:
            print("Especifique o ID do dataset (1 a 4). Ex: python modelozudo.py treinar 2")
            sys.exit(1)
        treinar(sys.argv[2])
    elif comando == "testar":
        if len(sys.argv) < 3:
            print("Especifique o ID do dataset (1 a 4). Ex: python modelozudo.py testar 2")
            sys.exit(1)
        testar(sys.argv[2])
    elif comando == "treinar_todos":
        for idx in [2, 3, 4]:
            try:
                treinar(idx)
            except Exception as exc:
                print(f"Erro ao treinar Modelo {idx}: {exc}")
