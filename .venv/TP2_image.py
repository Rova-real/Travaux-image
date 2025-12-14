import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color


I = io.imread('.venv/astronaut.png')
I = color.rgb2gray(I)

#function histogram
def histogram(I, edges):
    #I = io.imread('Image')
    H = np.zeros(256)  # bins = 256
    if edges > 1:
        I= I.flatten() #convertit en un tableau 1D
    else :
        pass
    for i in range(len(I)):
        H[I[i]] += 1
    return H

#test de la fonction histogramme
H = histogram(I, 256)

#affichage histogramme
plt.figure()
plt.bar(range(256),H)
plt.title('Histogram avec notre fonction')
plt.show()

#comparaison avec le module histogramme de numpy
I = I.flatten()
H_py = np.histogram(I, 256)

#visualisation
plt.figure()
plt.bar(range(256),H_py[0])
plt.title('Histogram avec numpy')
plt.show()

#egalisation
def equalize(image):
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    #calcul de l'histogramme de l'image
    hist, bins = np.histogram(image.flatten(), 256)
    cumul = np.cumsum(hist, dtype= int)
    pixel = cumul[-1]
    cumul_norm = (cumul * 255 / pixel).astype(np.uint8)
    imag_egali = cumul_norm[image]  # utilisation du slicing
    '''
    im = []
    for i in image:
        for j in cumul_norm:
            '''
    return imag_egali

#affichage des deux histogramme egalisé et non égalisé

imag_eq = equalize(I)
hist_eg,_ = np.histogram(imag_eq, 256)
hist_no,_ = np.histogram(I, 256)
plt.figure(1)
plt.subplot(211)
plt.bar(range(256),hist_eg)
plt.title('Histogram egalisé')
plt.subplot(212)
plt.bar(range(256),hist_no)
plt.title('Histogram sans egalisation')
plt.tight_layout()
plt.axis('off')
plt.show()

#affichage de l'image modifié

plt.imshow(imag_eq, cmap='gray', vmin=0, vmax=255)
plt.title('Image avec histogram égalisé')
plt.axis('off')
plt.savefig('LenaEqualized.png')
plt.show()


#image quantification
def quantification(image, n_bits):
    pas = (255/(2**n_bits))
    val = (image // pas).astype(np.uint8)
    result = val * pas
    return result


def affichage(image, bits, *vmin):

    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(f' Image avec {bits} bits')
    plt.axis('off')
    plt.show()
    return 0

#test pour différents n bits
n_bits = [2,4,6,8,9,10,11]
I = I*255
for n in n_bits:
    img = quantification(I,n)
    affichage(img, n)
    print(f'')
'''à partir de 9 bits, l'image devient sombre, la quantification rend l'image sombre
'''







