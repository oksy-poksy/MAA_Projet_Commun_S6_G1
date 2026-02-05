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

    def affiche_image(self, pixels):
        img = Image.fromarray(pixels.astype('uint8'))
        img.show()

    def binarisation(self, pixels): #mettre la valeur des pixels FAUSTINE
        gris = np.mean(pixels, axis=2)
        seuil = np.mean(pixels)*0.75
        image_binaire = np.where(gris > seuil, 255, 0).astype('uint8')
        return image_binaire

    def histogramme(self,image_binarisee): #entree : image binarisée, sortie : "list numpy" ELANA
        image_numpy = np.where(image_binarisee==0, 1,0)
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

    def redimensionner_image(self,image,largeur,hauteur):
        image=Image.fromarray(image)
        image_redimensionnee=image.resize((largeur,hauteur))
        image_redimensionnee=np.array(image_redimensionnee)
        return image_redimensionnee

    def correction2pente(self): #inutile c'est carré dans l'axe PERSONNE
        pass

    def correction_inclinaison(self): #inutile c'est carré dans l'axe PERSONNE
        pass

class Reseau2Neurone :
    def __init__(self, nb_couche, neurones_couche, taux_apprentissage):

        self.nb_couche = nb_couche
        self.neurones_couche = neurones_couche
        self.taux_apprentissage = taux_apprentissage

        self.sommes = {}
        self.reseau_poids={}
        self.activation = {}

        # Oksana
        self.erreurs = {}
        self.gradients = {}

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

    def backward_propagation(self, label):
        vecteur_attendu = np.zeros(self.neurones_couche[-1])
        vecteur_attendu[label] = 1

        L = self.nb_couche

        self.erreurs[L - 1] = self.activation[L - 1] - vecteur_attendu

        # Propagation de l'erreur vers les couches cachées (en arrière)
        for couche in range(L - 2, 0, -1):
            poids_sans_biais = self.reseau_poids[couche + 1][:, :-1]
            erreur_propagee = np.dot(poids_sans_biais.T, self.erreurs[couche + 1])
            derivee_activation = np.where(self.sommes[couche] > 0, 1, 0)
            self.erreurs[couche] = erreur_propagee * derivee_activation

        # Mise à jour des poids (Descente de gradient)
        for couche in range(L - 1, 0, -1):
            activation_prec_avec_biais = np.append(self.activation[couche - 1], 1)
            self.gradients[couche] = np.outer(self.erreurs[couche], activation_prec_avec_biais)
            self.reseau_poids[couche] -= self.taux_apprentissage * self.gradients[couche]
    
class Entrainement :
    pass

#### CODE SOURCE ####
matheo = Traitement("mat.png")
p = matheo.binarisation(matheo.decoupe_en_pixel())
print(matheo.histogramme(p))
matheo.affiche_image(p)
