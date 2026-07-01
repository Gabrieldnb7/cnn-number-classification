# 🧠 Reconhecimento de Números Manuscritos (CNN)

## 🎯 Objetivo do Projeto

O objetivo principal deste projeto é desenvolver um sistema capaz de reconhecer e classificar dígitos numéricos escritos à mão (de 0 a 9) em tempo real, capturados através de uma webcam. O projeto utiliza uma **Rede Neural Convolucional (CNN)** treinada com o clássico banco de dados MNIST, demonstrando a aplicação prática de técnicas de *Deep Learning* e *Visão Computacional*.

## 📁 Estrutura de Arquivos

Abaixo estão os principais arquivos que compõem o repositório e suas respectivas finalidades:

* **`number_classifier.ipynb`**: Arquivo Jupyter Notebook responsável por toda a etapa de inteligência artificial. Contém o carregamento do dataset MNIST, pré-processamento dos dados, construção da arquitetura, compilação, treinamento, avaliação e exportação do modelo da CNN usando o framework TensorFlow/Keras.
* **`webcam.py`**: Script em Python que realiza o reconhecimento em tempo real. Ele utiliza a biblioteca OpenCV para acessar a webcam, capturar os quadros, aplicar pré-processamento na imagem (binarização, suavização e redimensionamento para 28x28 pixels) e realizar as predições baseando-se no modelo.
* **`modelo.h5`**: Arquivo final do modelo treinado. Ele armazena os pesos e a estrutura arquitetural da rede neural gerada no notebook. Este modelo é lido diretamente pelo `webcam.py` para fazer a inferência (não é preciso retreinar a cada execução).
* **`relatorio.md`** e **`Relatório - Projeto de Rede Neural Convolucional.pdf`**: Relatórios técnicos detalhando a fundamentação teórica, escolhas arquiteturais, processo de desenvolvimento e conclusões do projeto.

## 🚀 Como Rodar o Projeto

Siga o passo a passo abaixo para executar o projeto na sua máquina:

### 1. Pré-requisitos
Certifique-se de ter o Python (versão 3.8 a 3.11 recomendada) instalado na sua máquina. O uso de um ambiente virtual (`venv`) é recomendado.
Instale as bibliotecas necessárias executando no terminal:
```bash
pip install tensorflow opencv-python numpy
```
*(Caso queira interagir e rodar o arquivo `.ipynb`, será necessário ter instalado também o Jupyter)*

### 2. Treinamento do Modelo (Opcional)
O projeto já conta com o arquivo treinado `modelo.h5`. Porém, caso você deseje executar o treinamento do zero para testar hiperparâmetros ou inspecionar o código:
1. Inicie o Jupyter e abra o notebook `number_classifier.ipynb` (ou suba-o no Google Colab).
2. Execute todas as células sequencialmente. 
3. Ao final, o arquivo `modelo.h5` será recriado no seu diretório com o resultado deste novo treinamento.

### 3. Execução em Tempo Real pela Webcam
Para rodar a classificação ao vivo:
1. Com a sua câmera disponível, abra o terminal na pasta do projeto.
2. Execute o script Python:
   ```bash
   python webcam.py
   ```
3. Uma tela com o vídeo da câmera aparecerá, contendo um "quadrado verde" (a Área de Interesse).
4. Escreva um número bem visível de 0 a 9 em um pedaço de papel.
5. Posicione a folha de forma que o **número desenhado fique estritamente dentro do quadrado verde**.
6. Uma segunda tela mostrará como o modelo está enxergando o número (Threshold), e o valor previsto com a taxa de confiança aparecerá acima do quadrado verde no vídeo original.
7. Para interromper o programa, certifique-se de que a janela do vídeo esteja selecionada e pressione a tecla **`q`** ou **`ESC`**.
