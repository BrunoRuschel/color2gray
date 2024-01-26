import cv2
import math
import numpy as np

# Carregar a imagem
imagem = cv2.imread('preview.jpg')

theta = 45

# Converter a imagem para o espaço de cores L*a*b*
imagem_lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)

height, width, channals = imagem.shape
# Dividir os canais L*, a*, e b*
#L, a, b = cv2.split(imagem_lab)

pixels = imagem_lab.astype(int)

cos_theta, sin_theta = math.cos(theta), math.sin(theta)

ps = pixels.reshape((height * width, 3))
l, a, b = map(list, zip(*ps))

#normalizando as listas lab para mais fácil manipulação
l = [x * 100 / 255 for x in l]
a = [x - 128 for x in a]
b = [x - 128 for x in b]

l_media = sum(l) / len(l)

pixels = np.array(list(zip(l, a, b))).reshape((height, width, 3))

def Delta(i, j, alpha, theta):
    #calcula diferenças em L, a, b
    delta_a = i[1] - j[1]
    delta_b = i[2] - j[2]
    delta_l = i[0] - j[0]

    #norma euclidiana do vetor Cij (dAij, dBij)
    norma_c = math.sqrt(delta_a ** 2 + delta_b ** 2)

    #calcula crunch
    crunch = alpha * math.tan(norma_c / alpha)

    #retorna de acordo com a definição de Delta
    if abs(delta_l) > crunch:
        return delta_l
    if delta_a * math.cos(theta) + delta_b * math.sin(theta) >= 0:
        return crunch
    return -crunch

# -------------------calcula a vizinhanca de um pixel de acordo com o parâmetro u---------------------------
# cria uma lista 2d e inicializa com zeros
vizinhanca = np.zeros((width, height))
# garante que a vizinhança está nos limites da imagem
for i in range(0, height):
    for j in range(0, width):
        if (i - u) >= 0:
            viz_cima = i - u
        else:
            viz_cima = 0
        if (i + u) <= (height - 1):
            viz_baixo = i + u
        else:
            viz_baixo = height - 1
        if (j - u) >= 0:
            viz_esq = j - u
        else:
            viz_esq = 0
        if (j + u) <= (width - 1):
            viz_dir = j + u
        else:
            viz_dir = width - 1
        for k in range(viz_esq, viz_dir+1):
            for z in range(viz_cima, viz_baixo+1):
                # garante que o pixel central nao é contado
                if i * width + j != k * width + z:
                    vizinhanca[i][j] += 1



# Ajustar o tamanho da janela para 600x400 pixels (ou o tamanho desejado)
cv2.namedWindow('Canal L*', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Canal L*', 600, 400)

cv2.namedWindow('Canal a*', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Canal a*', 600, 400)

cv2.namedWindow('Canal b*', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Canal b*', 600, 400)

# Exibir ou salvar os canais separados, se necessário
cv2.imshow('Canal L*', L)
cv2.imshow('Canal a*', a)
cv2.imshow('Canal b*', b)

# Aguardar a tecla 'q' ser pressionada para fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
