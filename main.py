import numpy as np
from PIL import Image



class Traitement :
    def __init__(self, image):
        self.image = image
        self.result = ""
        self.list_image = None
    def decoupe_en_pixel(self): #avec numpy MATHEO
        img = Image.open(self.image)
        pixels = np.array(img)
        return pixels
    def binarisation(self, pixels): #mettre la valeur des pixels FAUSTINE
        pass
    def histogramme(self): #entree : image binarisée, sortie : "list numpy" ELANA
        pass
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
    def backward(self): #
        pass
class Entrainement :
    pass


#### CODE SOURCE ####

