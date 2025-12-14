import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color


I = io.imread('.venv/astronaut.png')
I = color.rgb2gray(I)  #convertir l'image en gris

#function histogram
def histogram(I, edges):
    H = np.zeros(edges)  # bins = 256
    I = I.flatten()  # convertit en un tableau 1D
    if np.max(I) <= 1:
        I = (I*255).astype(np.uint8)  #transformation en entier et multiplier par 255
    else :
        I = I.astype(np.uint8)
    for i in I:
        H[i] += 1
    return H

#test de la fonction histogramme
H = histogram(I, 256)

#affichage histogramme
plt.figure(1)
plt.bar(range(256),H)
plt.title('Histogramme avec notre propre fonction')
plt.savefig('image/histogram.png')
plt.show()

#comparaison avec le module histogramme de numpy
def hist_nump(I, bins):
    I = I.flatten()
    return np.histogram(I, bins=bins)

#I = I.flatten()
#H_py = np.histogram(I, 256)

H_py = hist_nump(I, 256)
#visualisation
plt.figure(2)
plt.bar(range(256),H_py[0])
plt.title('Histogram avec numpy')
plt.savefig('image/Histogram_numpy.png')

plt.show()

#egalisation
def equalize(image):
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)  # si la valeur est compris entre 0 et 1, on convertit en 0 à 255
    else:
        image = image.astype(np.uint8) # type = entier
    #calcul de l'histogramme de l'image
    hist, bins = np.histogram(image.flatten(), 256)
    cumul = np.cumsum(hist, dtype= int)  #somme cumulé de l'histogramme
    pixel = cumul[-1] #nombre de pixel
    cumul_norm = (cumul * 255 / pixel).astype(np.uint8)
    imag_egali = cumul_norm[image]  # utilisation du slicing

    return imag_egali

#affichage des deux histogramme egalisé et non égalisé

imag_eq = equalize(I)
hist_eg,_ = np.histogram(imag_eq, 256)
hist_no,_ = np.histogram(I, 256)

plt.figure(3)
plt.subplot(211)
plt.bar(range(256),hist_eg)
plt.title('Histogram egalisé')

plt.subplot(212)
plt.bar(range(256),hist_no)
plt.title('Histogram sans egalisation')

plt.tight_layout()
plt.savefig('image/histogramme égalisé.png')
plt.show()

#affichage de l'image modifié
plt.figure(4)
plt.subplot(121)
plt.imshow(imag_eq, cmap='gray', vmin=0, vmax=255)
plt.title('Image avec histogram égalisé')
plt.axis('off')
plt.subplot(122)
plt.imshow(I, cmap='gray')
plt.axis('off')
plt.title('Image avec histogram sans egalisation')

plt.tight_layout()
plt.savefig('image/LenaEqualized.png')
plt.show()


#image quantification
def quantification(image, n_bits): #quantification sur n_bits
    pas = (255/(2**n_bits)) #notre pas de quantification
    val = (image // pas).astype(np.uint8) #division entière
    result = val * pas #on applique la formule donné dans le sujet
    return result


img_quant = quantification(I, 4) #before equalization
img_quant_equal = quantification(imag_eq, 4)#after equalization

#affichage deds deux images
plt.subplot(121)
plt.imshow(img_quant, cmap='gray', vmin=0, vmax=255)
plt.title('image quantifié')
plt.axis('off')
plt.subplot(122)
plt.imshow(img_quant_equal, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('image égalisé quantifié ')
plt.tight_layout()
plt.savefig('image/comparaison_quantifier.png')
plt.show()




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
    print(f'{n}bits')
'''à partir de 9 bits, l'image devient sombre, la quantification rend l'image sombre
'''







