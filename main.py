import numpy as np
from PIL import Image
class Traitement :
    def __init__(self, image):
        self.image = image
        self.result = ""
        self.list_image = None
    def ouvrir_image(self): #FAUFAU
        pass
    def decoupe_en_pixel(self): #avec numpy MATHEO
        img = Image.open(self.image)
        pixels = np.array(img)
        return pixels
    def binarisation(self, pixels): #mettre la valeur des pixels FAUSTINE
        seuil = np.mean(pixels)
        image_binaire = np.where(pixels>seuil, 255, 0)
        return image_binaire

    def histogramme(self,image_binarisee): #entree : image binarisée, sortie : "list numpy" ELANA
        image_numpy = np.where(image_binarisee >0, 1,0)
        histogramme = np.sum(image_numpy, axis=1)
        return histogramme
    def selection_lignes(self,histogramme,image_binarisee,seuil): # MAXIME
        #le seuil désigne à partir de quel nombre de pixel on peut considérer que ça fait partie de la ligne
        #cela sert à ne pas prendre en compte les points "parasites"
        lignes={} #on met les lignes découpées dans un dico
        i=0
        j=-1
        while i<len(histogramme): #on s'assure de bien regarder chaque ligne
            while i<len(histogramme) and histogramme[i]<seuil:
                i+=1 #on détecte la première ligne de pixel à partir de laquelle il y a un texte
            j+=1 #on a détecté une nouvelle ligne à prendre en compte --> on met dans l'image
            k=i #on regarde le nombre de ligne de pixels qui composent cette ligne de texte
            #on va prendre les lignes de i à là où s'arrête la ligne
            while i<len(histogramme) and histogramme[i]>=seuil:
                i+=1
            lignes[j]=image_binarisee[k:i]
        return lignes #(à ajouter --> exceptions --> par exemple lignes d'un seul pixel)

    def histogrammes_colonnes(self, ligne): #on fait de même, un histogramme, mais dans l'autre sens MAXIME
        liste_colonnes=ligne.T
        image_numpy = np.where(liste_colonnes > 0, 1, 0)
        histogramme_colonnes = np.sum(image_numpy, axis=1)
        return histogramme_colonnes

    def selection_colonnes(self, histogramme_colonnes, image_binarisee,seuil): #on fait de même que pour
        #les lignes mais avec les colonnes afin de détacher les lettres. Il faudra retransposer
        #pour avoir les images dans le bon sens
        lettres={}
        i = 0
        j = -1
        while i < len(histogramme_colonnes):
            while i < len(histogramme_colonnes) and histogramme_colonnes[i] < seuil:
                i += 1
            j += 1
            k = i
            while i < len(histogramme_colonnes) and histogramme_colonnes[i] >= seuil:
                i += 1
            lettres[j] = image_binarisee[k:i].T #on transpose pour remettre l'image droite
        return lettres

    def correction2pente(self): #inutile c'est carré dans l'axe PERSONNE
        pass
    def correction_inclinaison(self): #inutile c'est carré dans l'axe PERSONNE
        pass
class Reseau2Neurone :
    def forward(self): #PIERRE
        pass
    def backward(self): # Oksana
        pass
class Entrainement :
    pass


#### CODE SOURCE ####
image = ""
Reseau = Reseau2Neurone
file = Traitement(image)
