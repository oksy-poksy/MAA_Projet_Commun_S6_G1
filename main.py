import numpy as np
from PIL import Image
from ReadingEMNIST import *
import pickle

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
        nouvelles_lignes = {} #lignes corrigés
        i=0
        j=-1
        while i<len(histogramme): #on s'assure de bien regarder chaque ligne
            while i<len(histogramme) and histogramme[i]<seuil:
                i+=1 #on détecte la première ligne de pixel à partir de laquelle il y a un texte
            j+=1 #on a détecté une nouvelle ligne à prendre en compte --> on met dans l'image
            k=i #on regarde le nombre de ligne de pixels qui composent cette ligne de texte
            #on va prendre les lignes de i à là où s'arrête la ligne
            deuxieme_seuil=seuil
            while i<len(histogramme) and histogramme[i]>=deuxieme_seuil:
                deuxieme_seuil=1 #permet de prendre en compte la barre du p par exemple --> les éléments fins
            #qui font partie de la lettre. On pourra enlever cela si on le juge peu important
                i+=1
            lignes[j]=image_binarisee[k:i]

        hauteurs = []
        for i in range(len(lignes)):
            hauteurs.append(len(lignes[i]))
        hauteurs = np.array(hauteurs)
        hauteur_moyenne = np.mean(hauteurs)
        i = 0
        while i < len(lignes)-1: #on recolle les points des i
            if len(lignes[i]) < hauteur_moyenne * 0.5:
                nouvelles_lignes[i] = np.vstack((lignes[i],lignes[i + 1]))
                i += 2
            else:
                nouvelles_lignes[i] = lignes[i]
                i += 1
        return nouvelles_lignes #(à ajouter --> exceptions --> par exemple lignes d'un seul pixel)

    def histogrammes_colonnes(self, ligne): #on fait de même, un histogramme, mais dans l'autre sens MAXIME
        liste_colonnes=ligne.T
        image_numpy = np.where(liste_colonnes== 0, 1, 0)
        histogramme_colonnes = np.sum(image_numpy, axis=1)
        return histogramme_colonnes

    def selection_colonnes(self, histogramme_colonnes, ligne,seuil): #on fait de même que pour
        #les lignes mais avec les colonnes afin de détacher les lettres.
        lettres={}
        espaces={}#on compare les espaces pour savoir s'il s'agit ou non seulement d'espace entre les lettres
        #comme on saura les espaces on saura quels ensembles de lettres sont des mots
        i = 0
        j = -1
        while i < len(histogramme_colonnes):
            while i < len(histogramme_colonnes) and histogramme_colonnes[i] < seuil:
                i += 1
            j += 1
            k = i
            while i < len(histogramme_colonnes) and histogramme_colonnes[i] >= seuil:
                i += 1
            lettres[j] = ligne[:,k:i] #on sélectionne les colonnes correspondantes,
            # quelque soit la ligne
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
    def entrainement_consecutif(self,nb_essais, nb_couche, neurones_couche, taux_apprentissage):
        meilleur_modele = {}
        # {"Reseau":None,"sommes":None,"activation":None,"meilleure_precision":None,
        #  "evolution_perte_moyenne":None,"evolution_precision":None}
        # on ne retiendra que ce qui est utilisé dans la forward (afin de prédire)
        meilleure_precision = 0
        reseau = Reseau2Neurone(nb_couche, neurones_couche, taux_apprentissage)
        images = MnistDataloader()
        (x_train, y_train), (x_test, y_test) = images.load_data()
        nb_it = len(x_train)
        cpt = 0
        evolution_precision = np.array([])
        evolution_perte_moyenne = np.array([])
        rang = 0
        total_loss = 0
        for i in range(nb_it):
            picture = np.array(x_train[i]) / 255.0
            picture_a = picture.flatten()
            reseau.forward(picture_a)
            prediction = reseau.activation[reseau.nb_couche - 1]
            pred = np.argmax(prediction)
            perte = 0.5 * np.sum((prediction - [1 if j == y_train[i]-1 else 0 for j in range(26)]) ** 2)
            total_loss += perte
            result = np.argmax([1 if j == y_train[i]-1 else 0 for j in range(26)])
            if pred == result:
                cpt += 1
            reseau.backward_propagation(y_train[i]-1)
            rang += 1
            evolution_precision = np.append(evolution_precision, cpt / rang)
            evolution_perte_moyenne = np.append(evolution_perte_moyenne, total_loss / rang)
        precision = cpt / nb_it
        meilleur_modele["meilleure_precision"] = precision
        meilleur_modele["Reseau"] = reseau.reseau_poids
        meilleur_modele["evolution_precision"] = evolution_precision
        meilleur_modele["evolution_perte_moyenne"] = evolution_perte_moyenne
        meilleur_modele["sommes"] = reseau.sommes
        meilleur_modele["activation"] = reseau.activation
        meilleur_modele["erreurs"] = reseau.erreurs
        # Visualisation
        plt.figure(figsize=(15, 6))
        # x = les étapes (1, 2, 3, ...)
        x = range(1, len(evolution_precision) + 1)

        # Scatter plot + ligne pour voir l'évolution
        plt.plot(x, evolution_precision, linewidth=1, alpha=0.7, color='blue', label='Précision')
        plt.plot(x, evolution_perte_moyenne, linewidth=2, alpha=0.7, color='black', label='Perte moyenne')
        plt.scatter(x, evolution_precision, s=2, alpha=0.3, color='red')

        plt.title(f"Résultat du 1er entrainement sur ({len(evolution_precision)} images)")
        plt.xlabel("Image traitée")
        plt.ylabel("Précision cumulée")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Ajouter la précision finale comme ligne horizontale
        plt.axhline(y=precision, color='green', linestyle='--', alpha=0.5,
                    label=f'Précision finale: {precision:.4f}')
        plt.legend()
        print("1ère simulation effectuée")
        for j in range(2, nb_essais + 1):
            images = MnistDataloader()
            (x_train, y_train), (x_test, y_test) = images.load_data()
            nb_it = len(x_train)
            cpt = 0
            evolution_precision = np.array([])
            evolution_perte_moyenne = np.array([])
            rang = 0
            total_loss = 0
            for i in range(nb_it):
                picture = np.array(x_train[i]) / 255.0
                picture_a = picture.reshape(len(x_train[i]), -1)
                reseau.forward(picture_a)
                prediction = reseau.activation[reseau.nb_couche - 1]
                pred = np.argmax(prediction)
                perte = 0.5 * np.sum((prediction - [1 if j == y_train[i]-1 else 0 for j in range(26)]) ** 2)
                total_loss += perte
                result = np.argmax([1 if j == y_train[i]-1 else 0 for j in range(26)])
                if pred == result:
                    cpt += 1
                reseau.backward_propagation(y_train[i]-1)
                rang += 1
                evolution_precision = np.append(evolution_precision, cpt / rang)
                evolution_perte_moyenne = np.append(evolution_perte_moyenne, total_loss / rang)
            precision = cpt / nb_it
            meilleur_modele["meilleure_precision"] = precision
            meilleur_modele["Reseau"] = reseau.reseau_poids
            meilleur_modele["evolution_precision"] = evolution_precision
            meilleur_modele["evolution_perte_moyenne"] = evolution_perte_moyenne
            meilleur_modele["sommes"] = reseau.sommes
            meilleur_modele["activation"] = reseau.activation
            meilleur_modele["erreurs"] = reseau.erreurs
            # Visualisation
            plt.figure(figsize=(15, 6))
            # x = les étapes (1, 2, 3, ...)
            x = range(1, len(evolution_precision) + 1)

            # Scatter plot + ligne pour voir l'évolution
            plt.plot(x, evolution_precision, linewidth=1, alpha=0.7, color='blue', label='Précision')
            plt.plot(x, evolution_perte_moyenne, linewidth=2, alpha=0.7, color='black', label='Perte moyenne')
            plt.scatter(x, evolution_precision, s=2, alpha=0.3, color='red')

            plt.title(f"Résultat du {j}ème entrainement sur ({len(evolution_precision)} images)")
            plt.xlabel("Image traitée")
            plt.ylabel("Précision cumulée")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Ajouter la précision finale comme ligne horizontale
            plt.axhline(y=precision, color='green', linestyle='--', alpha=0.5,
                        label=f'Précision finale: {precision:.4f}')
            plt.legend()
            print(f"{j}eme simulation effectuée")
            reseau.taux_apprentissage = reseau.taux_apprentissage / 2  # on diminue le taux d'apprentissage apr_s chaque itération
        return meilleur_modele
    def test(self, nb_couche, neurones_couche, taux_apprentissage,meilleur_modele):
        # TEST
        images = MnistDataloader()
        (x_train, y_train), (x_test, y_test) = images.load_data()
        nb_it = len(x_test)
        cpt = 0
        evolution_precision = np.array([])
        evolution_perte_moyenne = np.array([])
        rang = 0
        total_loss = 0
        reseau = Reseau2Neurone(nb_couche, neurones_couche, taux_apprentissage)
        reseau.reseau_poids = meilleur_modele["Reseau"]
        reseau.sommes = meilleur_modele["sommes"]
        reseau.activation = meilleur_modele["activation"]
        for i in range(nb_it):
            picture = np.array(x_test[i]) / 255.0
            picture_a = picture.reshape(len(x_test[i]), -1)
            reseau.forward(picture_a)
            prediction = reseau.activation[reseau.nb_couche - 1]
            pred = np.argmax(prediction)
            perte = 0.5 * np.sum((prediction - [1 if j == y_test[i]-1 else 0 for j in range(26)]) ** 2)
            total_loss += perte
            result = np.argmax([1 if j == y_test[i]-1 else 0 for j in range(26)])
            if pred == result:
                cpt += 1
            rang += 1
            evolution_precision = np.append(evolution_precision, cpt / rang)
            evolution_perte_moyenne = np.append(evolution_perte_moyenne, total_loss / rang)
        precision = cpt / nb_it

        #### VISUALISATION ####
        plt.figure(figsize=(15, 6))

        x = range(1, len(evolution_precision) + 1)

        plt.plot(x, evolution_precision, linewidth=1, alpha=0.7, color='blue', label='Précision')
        plt.plot(x, evolution_perte_moyenne, linewidth=2, alpha=0.7, color='black', label='Perte moyenne')
        plt.scatter(x, evolution_precision, s=2, alpha=0.3, color='red')

        plt.title(f"Résultat des tests sur ({len(evolution_precision)} images)")
        plt.xlabel("Image traitée")
        plt.ylabel("Précision cumulée")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.axhline(y=precision, color='green', linestyle='--', alpha=0.5,
                    label=f'Précision finale: {precision:.4f}')
        plt.legend()

        plt.tight_layout()

        print(
            f'Précision finale du test : {precision}, Paramètres utilisés : learning rate de {taux_apprentissage}, nombre de neurones (par couche):{neurones_couche}')

#### CODE SOURCE ####
matheo = Traitement("mat.png")
p = matheo.decoupe_en_pixel()
p2 = matheo.binarisation(p)
h = matheo.histogramme(p2)
lignes=matheo.selection_lignes(h, p2, 50)
matheo.affiche_image(lignes[0])
hc=matheo.histogrammes_colonnes(lignes[0])
lettres=matheo.selection_colonnes(hc,lignes[0],20)

#Code pour charger les paramètres du réseau de neurone :

with open("mon_modele_ocr.pkl", "rb") as fichier:
    modele_charge = pickle.load(fichier)

poids_entraines = modele_charge["Reseau"]

nb_couche = modele_charge["nb_couche"]
neurones_couche = modele_charge["neurones_couche"]
taux_apprentissage = modele_charge["taux_apprentissage"]
mon_reseau = Reseau2Neurone(nb_couche, neurones_couche, taux_apprentissage)
mon_reseau.reseau_poids = poids_entraines
###