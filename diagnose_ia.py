import sys
import os

print("--- DIAGNÓSTICO ---")
print(f"Diretório atual: {os.getcwd()}")

try:
    print("\n1. Importando bibliotecas...")
    import torch
    import numpy as np
    import joblib
    print("Sucesso! PyTorch e dependências carregadas.")
    print(f"  PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"ERRO DE IMPORTAÇÃO: {e}")
    sys.exit(1)

try:
    print("\n2. Testando importação do modelozudo...")
    import modelozudo
    print("Sucesso! modelozudo importado.")
except Exception as e:
    print(f"ERRO AO IMPORTAR modelozudo.py: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. Verificando existência dos arquivos de modelo .pth...")
    for i in range(1, 5):
        path = f"modelo_{i}.pth"
        exists = os.path.exists(path)
        print(f"  - {path}: {'EXISTE' if exists else 'NÃO ENCONTRADO'} (Tamanho: {os.path.getsize(path) if exists else 0} bytes)")
except Exception as e:
    print(f"Erro ao verificar arquivos .pth: {e}")

try:
    print("\n4. Tentando carregar normalizadores e pesos neurais como feito no modelozudo...")
    import torch.nn as nn
    

    class RedeClassificacao(nn.Module):
        def __init__(self, input_dim):
            super(RedeClassificacao, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 2)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            out = self.softmax(out)
            return out

    print("Carregando normalizador_1.joblib...")
    norm = joblib.load("normalizador_1.joblib")
    print("Sucesso! Normalizador carregado.")

    print("Carregando modelo_1.pth...")
    model = RedeClassificacao(10)

    state_dict = torch.load("modelo_1.pth", map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Sucesso! Modelo PyTorch carregado.")
except Exception as e:
    print(f"ERRO AO CARREGAR NORMALIZADOR OU PESOS: {e}")
    import traceback
    traceback.print_exc()

print("\nDiagnóstico encerrado.")
