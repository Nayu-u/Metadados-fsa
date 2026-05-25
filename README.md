Para rodar o programa localmente é preciso seguir os 3 passos abaixo, desde que tenha o mínimo de permissões na máquina, qualquer um consegue fazer isso


1.
Instalar o Python:

Certifique-se de ter o Python instalado.

Caso não tenha, você pode baixá-lo diretamente pelo site oficial (link: https://www.python.org/downloads/).


2.
O projeto precisa da ferramenta externa ExifTool para ler metadados. Siga o passo a passo rápido:

Baixar: Clique em Download ExifTool v13.58 (link https://sourceforge.net/projects/exiftool/files/exiftool-13.58_64.zip/download)

(o download começará automaticamente).

Esse link é para Windows 64-bits, caso o sistema operacional seja 32 ou Mac, procure o download no site (link : https://exiftool.org/)

Preparar: Extraia o .zip e renomeie o arquivo exiftool(-k).exe para exiftool.exe.

Mova o arquivo renomeado e a pasta exiftool_files no caminho:

C:\Program Files


3.
Como Executar

Com o ExifTool configurado, basta ir até a pasta do projeto e executar o arquivo:

iniciar.bat (Dê dois cliques)

O script irá instalar todas as dependências do Python necessárias automaticamente e abrirá o navegador no endereço http://localhost:5000.
