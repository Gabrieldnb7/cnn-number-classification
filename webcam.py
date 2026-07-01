import cv2
import numpy as np
from tensorflow.keras.models import load_model

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

print("Carregando modelo.h5...")
modelo = load_model('model.h5')

cap = cv2.VideoCapture(0)

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    altura, largura = frame.shape[:2]
    tamanho_quadrado = 400
    x1 = int(largura/2 - tamanho_quadrado/2)
    y1 = int(altura/2 - tamanho_quadrado/2)
    x2 = x1 + tamanho_quadrado
    y2 = y1 + tamanho_quadrado
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    area_recortada = frame[y1:y2, x1:x2]
    
    cinza = cv2.cvtColor(area_recortada, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, threshold = cv2.threshold(suave, 100, 255, cv2.THRESH_BINARY_INV)
    imagem_redimensionada = cv2.resize(threshold, (28, 28))
    imagem_normalizada = imagem_redimensionada / 255.0
    imagem_final = np.reshape(imagem_normalizada, (1, 28, 28, 1))
    
    predicao = modelo.predict(imagem_final, verbose=0)
    indice_classe = np.argmax(predicao)
    confianca = np.max(predicao)
    
    if confianca > 0.5:
        texto = f"Numero: {classes[indice_classe]} ({confianca*100:.1f}%)"
        cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    
    altura_tela, largura_tela = frame.shape[:2]
    visao_ampliada = cv2.resize(imagem_redimensionada, (largura_tela, altura_tela), interpolation=cv2.INTER_NEAREST)

    visao_ampliada_colorida = cv2.cvtColor(visao_ampliada, cv2.COLOR_GRAY2BGR)
    cv2.putText(visao_ampliada_colorida, "O que a IA enxerga (28x28 ampliado)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Reconhecimento de Numeros ao Vivo", frame)
    cv2.imshow("Visao do Modelo", visao_ampliada_colorida)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()