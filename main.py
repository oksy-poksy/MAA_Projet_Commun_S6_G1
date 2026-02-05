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

    def backward(self): # Oksana
        pass
class Entrainement :
    pass


#### CODE SOURCE ####
image = ""
Reseau = Reseau2Neurone
file = Traitement(image)
