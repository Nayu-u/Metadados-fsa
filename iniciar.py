import sys
import subprocess
import time
import webbrowser
import logging
from pathlib import Path

# Configuração de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Iniciador")

PROJETO_DIR = Path(__file__).parent.resolve()

def verificar_e_instalar_dependencias():
    logger.info("Verificando dependencias do projeto...")
    requirements_file = PROJETO_DIR / "requirements.txt"
    if not requirements_file.exists():
        logger.error("requirements.txt nao encontrado na raiz do projeto!")
        return False

    # Mapeamento simples de pacotes pip para nomes de importação para verificação rápida
    mapeamento = {
        "Flask": "flask",
        "pyexiftool": "exiftool",
        "numpy": "numpy",
        "requests": "requests",
        "Pillow": "PIL",
        "opencv-python": "cv2",
        "python-dotenv": "dotenv",
        "certifi": "certifi",
        "torch": "torch",
        "torchvision": "torchvision",
        "scikit-learn": "sklearn",
        "pandas": "pandas",
        "joblib": "joblib",
        "tqdm": "tqdm"
    }

    instalando = False
    for pacote, modulo in mapeamento.items():
        try:
            __import__(modulo)
        except ImportError:
            logger.info("Dependencia faltante detectada: %s", pacote)
            instalando = True
            break

    if instalando:
        logger.info("Instalando dependencias do requirements.txt... Isso pode levar alguns instantes.")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
            logger.info("Todas as dependencias foram instaladas com sucesso!")
        except subprocess.CalledProcessError as err:
            logger.error("Falha ao instalar dependencias: %s", err)
            logger.info("Tentando prosseguir mesmo assim...")
    else:
        logger.info("Todas as dependencias ja estao instaladas!")

    return True

def iniciar_servidor():
    server_script = PROJETO_DIR / "server.py"
    if not server_script.exists():
        logger.error("Arquivo server.py nao encontrado na raiz do projeto!")
        sys.exit(1)

    logger.info("Iniciando o servidor Flask (server.py)...")
    
    # Inicia o servidor Flask como um subprocesso
    processo = subprocess.Popen([sys.executable, str(server_script)], cwd=str(PROJETO_DIR))
    
    # Aguarda o Flask subir
    time.sleep(3)
    
    url = "http://localhost:5000"
    logger.info("Abrindo o navegador no endereço: %s", url)
    webbrowser.open(url)

    try:
        logger.info("Servidor Kingambit rodando com sucesso. Pressione Ctrl+C para encerrar.")
        processo.wait()
    except KeyboardInterrupt:
        logger.info("Encerrando o servidor Kingambit...")
        processo.terminate()
        try:
            processo.wait(timeout=5)
        except subprocess.TimeoutExpired:
            processo.kill()
        logger.info("Servidor encerrado.")

if __name__ == "__main__":
    print("==================================================")
    print("      Iniciador Automatico - Kingambit Forense    ")
    print("==================================================")
    
    verificar_e_instalar_dependencias()
    iniciar_servidor()
