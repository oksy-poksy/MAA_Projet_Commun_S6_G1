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
class Reseau2Neurone :
    def __init__(self, nb_couche, neurones_couche, taux_apprentissage):
        self.activation = {}
        self.nb_couche = nb_couche
        self.neurones_couche = neurones_couche
        self.taux_apprentissage = taux_apprentissage
        for couche in range(1, self.nb_couche):
            nb_de_colonnes = neurones_couche[couche - 1] + 1
            nb_de_lignes = self.neurones_couche[couche]
            self.reseau_poids[couche] = np.random.randn(nb_de_lignes,nb_de_colonnes) * 0.01

    def forward(self, image): #PIERRE
        self.activation[0] = image
        for couche in range(1, self.nb_couche):
            activation_avec_biais = np.append(self.activation[couche - 1], 1)
            self.sommes[couche] = np.dot(self.reseau_poids[couche], activation_avec_biais)
            self.activation[couche] = np.where(self.sommes[couche] > 0, self.sommes[couche], 0)
    def backward(self): # Oksana
        pass
class Entrainement :
    pass


#### CODE SOURCE ####
image = ""
Reseau = Reseau2Neurone
file = Traitement(image)
