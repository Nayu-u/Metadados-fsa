import os
import sys
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
except ImportError:
    print("Erro: reportlab nao esta instalado! Instale via 'pip install reportlab' antes de rodar este script.")
    sys.exit(1)

PROJETO_DIR = Path(__file__).parent.resolve()
CAMINHO_PDF = PROJETO_DIR / "sobre.pdf"

def gerar_pdf():
    print("Iniciando compilação do manual pericial sobre.pdf do Kingambit 2.1...")
    
    doc = SimpleDocTemplate(
        str(CAMINHO_PDF),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    estilos = getSampleStyleSheet()
    
    estilo_titulo_principal = ParagraphStyle(
        'TituloPrincipal',
        parent=estilos['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#d4a520'),
        spaceAfter=8,
        alignment=1
    )
    
    estilo_sub_principal = ParagraphStyle(
        'SubPrincipal',
        parent=estilos['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#c41a35'),
        spaceAfter=15,
        alignment=1
    )
    
    estilo_h1 = ParagraphStyle(
        'Heading1_Custom',
        parent=estilos['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#d4a520'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    estilo_h2 = ParagraphStyle(
        'Heading2_Custom',
        parent=estilos['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#c41a35'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    estilo_paragrafo = ParagraphStyle(
        'CorpoTexto',
        parent=estilos['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#e2e2e9'),
        spaceAfter=6
    )
    
    estilo_aviso = ParagraphStyle(
        'CaixaAviso',
        parent=estilos['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#8888a2'),
        spaceBefore=4,
        spaceAfter=8
    )

    estilo_lista_item = ParagraphStyle(
        'ListaItem',
        parent=estilo_paragrafo,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    conteudo = []
    
    # ------------------ CABEÇALHO / APRESENTAÇÃO ------------------
    conteudo.append(Spacer(1, 5))
    conteudo.append(Paragraph("MANUAL DE ANÁLISE FORENSE DIGITAL — KINGAMBIT v2.1", estilo_titulo_principal))
    conteudo.append(Paragraph("SISTEMA DE IDENTIFICAÇÃO E VALIDAÇÃO DE IMAGENS SINTÉTICAS (IA)", estilo_sub_principal))
    
    # Linha divisória horizontal dourada
    linha_dados = [['']]
    linha_tabela = Table(linha_dados, colWidths=[504], rowHeights=[1.5])
    linha_tabela.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#d4a520')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    conteudo.append(linha_tabela)
    conteudo.append(Spacer(1, 8))
    
    # Resumo técnico sério e formal
    introducao = (
        "<b>APRESENTAÇÃO DO SISTEMA:</b> O Kingambit v2.1 é uma ferramenta de análise forense computacional "
        "desenvolvida para examinar a integridade física, estrutural e lógica de arquivos de imagem. "
        "Através da extração de metadados Exif, geração de mapas de sinais e execução de modelos neurais profundos "
        "de forma local e offline, o sistema avalia a probabilidade de manipulação digital e geração sintética de conteúdo."
    )
    conteudo.append(Paragraph(introducao, estilo_aviso))
    conteudo.append(Spacer(1, 6))
    
    # ------------------ SEÇÃO 1: JÚRI DE IAS ------------------
    conteudo.append(Paragraph("1. ENSEMBLE NEURAL (JÚRI DE IAS — COMITÊ DE MODELOS ESPECIALIZADOS)", estilo_h1))
    p1 = (
        "O veredito de probabilidade de geração sintética é calculado a partir de um ensemble de redes neurais profundas "
        "(EfficientNet) calibradas localmente. A arquitetura do sistema utiliza uma média ponderada para consolidar os scores "
        "individuais de cada classificador:"
    )
    conteudo.append(Paragraph(p1, estilo_paragrafo))
    
    conteudo.append(Paragraph("• <b>IA Principal [PESO 6]:</b> O modelo central do sistema, treinado com um amplo e diverso dataset forense. Possui a maior taxa de acerto global e atua como o pilar principal do veredito final. Por sua precisão técnica superior, possui peso de decisão 6.", estilo_lista_item))
    conteudo.append(Paragraph("• <b>IA Geral [PESO 1]:</b> Modelo voltado para a análise de inconsistências em texturas macroscópicas e padrões de compressão típicos de geradores de imagem genéricos.", estilo_lista_item))
    conteudo.append(Paragraph("• <b>IA Multicategoria [PESO 1]:</b> Modelo focado em anomalias de iluminação e consistência física em objetos, vegetação e cenários gerais.", estilo_lista_item))
    conteudo.append(Paragraph("• <b>IA Face Detection [PESO 1]:</b> Modelo altamente especializado na identificação forense de rostos sintéticos, retratos e faces hiper-realistas geradas artificialmente.", estilo_lista_item))
    
    formula_expl = (
        "<b>Lógica de Fusão de Decisão:</b> O score final (Média Geral) é uma média ponderada dinâmica. "
        "O score do Modelo Principal é multiplicado por 6, os complementares por 1, e o resultado é dividido "
        "pela soma dos pesos dos modelos atualmente ativos no sistema, garantindo resiliência operacional."
    )
    conteudo.append(Paragraph(formula_expl, estilo_paragrafo))
    
    # ------------------ SEÇÃO 2: METADADOS ------------------
    conteudo.append(Paragraph("2. EXTRAÇÃO LOGÍSTICA (METADADOS EXIF)", estilo_h1))
    p_exif = (
        "Os metadados Exif consistem em informações gravadas diretamente no cabeçalho do arquivo durante a sua "
        "captura ou exportação. O sistema divide a auditoria destas informações em três áreas técnicas:"
    )
    conteudo.append(Paragraph(p_exif, estilo_paragrafo))
    
    conteudo.append(Paragraph("• <b>Destaques de IA:</b> Varredura automatizada em busca de strings, assinaturas ou metadados específicos deixados por plataformas de geração de imagens sintéticas. A presença destas informações é uma evidência inequívoca de origem artificial.", estilo_lista_item))
    conteudo.append(Paragraph("• <b>Câmera & Equipamento:</b> Leitura de parâmetros de captura óptica física (como fabricante, modelo da câmera, parâmetros da lente, abertura, tempo de exposição e ISO). Capturas fotográficas autênticas de sensores ópticos tipicamente contêm esses registros. A ausência total de metadados físicos serve como indicador de suspeita.", estilo_lista_item))
    conteudo.append(Paragraph("• <b>Metadados Completos:</b> Dump estruturado do cabeçalho binário lido via ExifTool para auditoria pericial detalhada.", estilo_lista_item))
    
    conteudo.append(PageBreak()) # Quebra para a segunda página
    
    # ------------------ SEÇÃO 3: LUPAS MÁGICAS (FILTROS) ------------------
    conteudo.append(Paragraph("3. MÉTODOS DE EXTRAÇÃO DE SINAIS E FILTROS FÍSICOS", estilo_h1))
    p_filtros = (
        "Para fundamentar tecnicamente o parecer neural, o Kingambit processa a imagem original através de filtros de imagem "
        "específicos, revelando inconsistências físicas imperceptíveis à visão humana:"
    )
    conteudo.append(Paragraph(p_filtros, estilo_paragrafo))
    
    conteudo.append(Paragraph("3.1 Compressão (ELA - Error Level Analysis)", estilo_h2))
    p_ela = (
        "A Análise de Nível de Erro (ELA) funciona salvando temporariamente a imagem em uma taxa de compressão JPEG predefinida "
        "e medindo a diferença nos valores de pixel em relação ao arquivo original. Áreas modificadas ou coladas reagem de forma "
        "diferente à compressão e se destacam com níveis de brilho anormalmente elevados no mapa de compressão, indicando inserções "
        "locais ou edições parciais na imagem."
    )
    conteudo.append(Paragraph(p_ela, estilo_paragrafo))
    
    conteudo.append(Paragraph("3.2 Ruído SRM (Spatial Rich Models)", estilo_h2))
    p_srm = (
        "Sensores físicos de câmeras fotográficas deixam um padrão microscópico e espacialmente distribuído de ruído de alta frequência "
        "(conhecido como PRNU). O filtro SRM isola o ruído do sensor removendo o conteúdo semântico da imagem. "
        "Geradores de IA criam imagens matematicamente homogêneas que carecem deste ruído óptico físico contínuo, "
        "apresentando anomalias na variância do ruído ou regiões excessivamente suavizadas."
    )
    conteudo.append(Paragraph(p_srm, estilo_paragrafo))
    
    conteudo.append(Paragraph("3.3 Frequências (FFT - Fast Fourier Transform)", estilo_h2))
    p_fft = (
        "Modelos gerativos de imagem realizam processos de interpolação e convolução em grades matemáticas discretas. "
        "Esse método de amostragem deixa assinaturas estruturais periódicas na imagem. A Transformada Rápida de Fourier (FFT) "
        "mapeia o espectro bidimensional de frequências da imagem, onde essas estruturas periódicas artificiais "
        "aparecem como padrões altamente geométricos, grades ou picos de brilho simétricos, inexistentes na fotografia óptica convencional."
    )
    conteudo.append(Paragraph(p_fft, estilo_paragrafo))
 
    conteudo.append(Paragraph("3.4 Gradiente de Sobel e Aberração Cromática", estilo_h2))
    p_grad_aber = (
        "As lentes de vidro físicas causam distorções ópticas naturais, como o desalinhamento de canais de cor nas transições de alto contraste "
        "(aberração cromática). A ausência completa desta distorção microscópica natural é uma assinatura de imagens sintéticas perfeitamente calculadas. "
        "Simultaneamente, o cálculo de gradientes por operador Sobel mede a variação das bordas: geradores artificiais tendem a suavizar ou "
        "padronizar as transições de luz e sombra, gerando médias de gradiente excessivamente baixas."
    )
    conteudo.append(Paragraph(p_grad_aber, estilo_paragrafo))
    
    # ------------------ SEÇÃO 4: PARECER TÉCNICO ------------------
    conteudo.append(Paragraph("4. PARECER FORENSE E SÍNTESE DE DIAGNÓSTICO", estilo_h1))
    p_parecer = (
        "A consolidação dos dados forenses resulta em um Parecer Técnico automatizado, categorizado de acordo com o nível "
        "geral de anomalia estrutural e a probabilidade de geração artificial:"
    )
    conteudo.append(Paragraph(p_parecer, estilo_paragrafo))
    
    conteudo.append(Paragraph("• 🟢 <b>Autenticidade Confirmada:</b> Indica que o arquivo exibe características físicas consistentes com fotografia óptica (ruído físico contínuo, metadados de câmera válidos, ausência de assinaturas periódicas na FFT) e unanimidade favorável no ensemble neural.", estilo_lista_item))
    conteudo.append(Paragraph("• 🟡 <b>Dúvida Razoável:</b> Sinais contraditórios detectados. O arquivo pode apresentar scores moderados nos modelos neurais ou anomalias parciais em um dos filtros físicos. Recomendada auditoria pericial detalhada.", estilo_lista_item))
    conteudo.append(Paragraph("• 🔴 <b>Forte Suspeita de IA:</b> Presença de evidências inequívocas de geração sintética (metadados específicos de geradores de IA, ausência completa de ruído de sensor, padrões simétricos periódicos na FFT ou forte indicação matemática pela IA Principal).", estilo_lista_item))
    
    p_operacao = (
        "<b>Interface e Operação Técnica:</b> O carregamento do arquivo executa simultaneamente o processamento "
        "multidimensional de sinais e a inferência de aprendizado de máquina. O painel unificado exibe de forma detalhada as "
        "métricas físicas e os vereditos probabilísticos de cada modelo, fundamentando cientificamente o diagnóstico pericial."
    )
    conteudo.append(Paragraph(p_operacao, estilo_paragrafo))

    # Linha final
    conteudo.append(Spacer(1, 10))
    conteudo.append(linha_tabela)
    conteudo.append(Spacer(1, 5))
    conteudo.append(Paragraph("<font color='#8888a2'>Kingambit v2.1 — Sistema de Auditoria de Imagens local, síncrono e 100% offline.</font>", estilo_aviso))
    
    # Construir o PDF
    print("Compilando arquivo PDF...")
    
    def draw_background(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor('#07070e'))
        canvas.rect(0, 0, 612, 792, fill=True, stroke=False)
        canvas.restoreState()
    
    doc.build(conteudo, onFirstPage=draw_background, onLaterPages=draw_background)
    print("Manual sobre.pdf gerado com sucesso!")

if __name__ == "__main__":
    gerar_pdf()
