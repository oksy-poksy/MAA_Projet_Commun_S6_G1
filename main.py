import numpy as np

class Traitement :
    def __init__(self, image):
        self.image = image
        self.result = ""
        self.list_image = None
    def ouvrir_image(self): #FAUFAU
        pass
    def decoupe_en_pixel(self): #avec numpy MATHEO
        pass
    def binarisation(self, pixels): #mettre la valeur des pixels FAUSTINE
        seuil = np.mean(pixels)
        image_binaire = np.where(pixels>seuil, 255, 0)
        return image_binaire

    def histogramme(self,image_binarisee): #entree : image binarisée, sortie : "list numpy" ELANA
        image_numpy = np.where(image_binarisee >0, 1,0)
        histogramme = np.sum(image_numpy, axis=1)
        return histogramme
    def selection_lignes(self): # MAXIME
        pass
    def selection_colonnes(self): #on sait pas encore comment on va faire MAXIME
        pass
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
