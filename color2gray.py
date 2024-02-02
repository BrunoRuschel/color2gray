import cv2
import math
import numpy as np
from scipy.sparse.linalg import cg
from matplotlib import pyplot as plt

# Carregar a imagem
input_img = cv2.imread('mapa.png')

theta = math.radians(45)
u = 1000
alpha = 20

# Converter a imagem para o espaço de cores L*a*b*
imagem_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)

height, width, channels = input_img.shape
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

print('debug1 ')

def delta(i, j, alpha, theta):
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

print('fez delta ')

# Função para calcular os limites da vizinhança
def calcula_limites(i, j, u, height, width):
    viz_cima = max(i - u, 0)
    viz_baixo = min(i + u, height - 1)
    viz_esq = max(j - u, 0)
    viz_dir = min(j + u, width - 1)
    return viz_cima, viz_baixo, viz_esq, viz_dir

# -------------------calcula a vizinhanca de um pixel de acordo com o parâmetro u---------------------------
# cria uma lista 2d e inicializa com zeros
vizinhanca = np.zeros((height, width))
# garante que a vizinhança está nos limites da imagem
for i in range(0, height):
    for j in range(0, width):
        viz_cima, viz_baixo, viz_esq, viz_dir = calcula_limites(i, j, u, height, width)
        for k in range(viz_esq, viz_dir+1):
            for z in range(viz_cima, viz_baixo+1):
                # garante que o pixel central nao é contado
                if i * width + j != k * width + z:
                    vizinhanca[i][j] += 1

print('vizinhanca ')

# calcula target difference, deltas[i][j] = delta_ij
deltas = [[0 for _ in range(0, height * width)] for _ in range(0, height * width)]
for i in range(0, height):
    for j in range(0, width):
        viz_cima, viz_baixo, viz_esq, viz_dir = calcula_limites(i, j, u, height, width)
        for k in range(viz_cima, viz_baixo+1):
            for z in range(viz_esq, viz_dir+1):
                deltas[i * width + j][k * width + z] = delta(pixels[i][j], pixels[k][z], alpha, theta)
print('deltas ')

# Constrói a matriz A a ser resolvida no sistema linear
diag = []
for row in vizinhanca:
    for col in row:
        diag.append(2 * col)
A = np.diag(diag)
for i in range(0, height):
    for j in range(0, width):
        viz_cima, viz_baixo, viz_esq, viz_dir = calcula_limites(i, j, u, height, width)
        for k in range(viz_cima, viz_baixo+1):
            for z in range(viz_esq, viz_dir+1):
                if i * width + j != k * width + z:
                    # -2 indica que há uma influência negativa entre os pixels vizinhos na solução do sistema linear
                    A[i * width + j][k * width + z] = -2
print('matriz a ')

# Constrói o vetor B
B = np.zeros((height * width,))
for i in range(0, height):
    for j in range(0, width):
        viz_cima, viz_baixo, viz_esq, viz_dir = calcula_limites(i, j, u, height, width)
        for k in range(viz_cima, viz_baixo+1):
            for z in range(viz_esq, viz_dir+1):
                B[i * width + j] += deltas[i * width + j][k * width + z] - deltas[k * width + z][i * width + j]
print('matriz b')

g_flat = np.asarray([[pixels[row][col][0] for col in range(0,width)] for row in range(0,height)]).flatten()


plt.imshow(np.reshape(g_flat, (height, width)), cmap='gray', vmin=0, vmax=255)

# ao resolver esse sistema, a solução será um vetor que representa como os valores de
# delta devem ser ajustados para alcançar a conversão desejada de cor para escala de cinza
res, info = cg(A, B, x0=g_flat)

res = res + (l_media - res.mean())

# escala de volta
res = res * 255 / 100

out = np.reshape(res,(height,width))


plt.imshow(out, cmap='gray', vmin=0, vmax=255)
cv2.imwrite("output5.png", out)

print("finished processing image", input_img,
      "output image to", out)
cv2.imshow('output', out)


# Aguardar a tecla 'q' ser pressionada para fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
