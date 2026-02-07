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
        seuil = np.mean(pixels)*0.7
        image_binaire = np.where(gris > seuil, 0, 255).astype('uint8')
        return image_binaire

    def histogramme(self,image_binarisee): #entree : image binarisée, sortie : "list numpy" ELANA
        image_numpy = np.where(image_binarisee==255, 1,0)
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
            if len(lignes[i]) < hauteur_moyenne * 0.25:
                nouvelles_lignes[i] = np.vstack((lignes[i],lignes[i + 1]))
                i += 2
            else:
                nouvelles_lignes[i] = lignes[i]
                i += 1
        return nouvelles_lignes #(à ajouter --> exceptions --> par exemple lignes d'un seul pixel)

    def histogrammes_colonnes(self, ligne): #on fait de même, un histogramme, mais dans l'autre sens MAXIME
        liste_colonnes=ligne.T
        image_numpy = np.where(liste_colonnes== 255, 1, 0)
        histogramme_colonnes = np.sum(image_numpy, axis=1)
        return histogramme_colonnes

    def selection_colonnes(self, hist, ligne, seuil):
        lettres = []
        espaces = []
        blocs = []

        i = 0
        n = len(hist)

        # --- 1. Découpage brut ---
        while i < n:
            # espace
            if hist[i] < seuil:
                start = i
                while i < n and hist[i] < seuil:
                    i += 1
                espaces.append((start, i))
                blocs.append(("espace", ligne[:, start:i]))

            # lettre
            else:
                start = i
                while i < n and hist[i] >= seuil:
                    i += 1
                lettres.append((start, i))
                blocs.append(("lettre", ligne[:, start:i]))

        # --- 2. Filtrage des petits espaces ---
        largeurs_espaces = [esp[1] - esp[0] for esp in espaces]
        if len(largeurs_espaces) > 0:
            moyenne = np.mean(largeurs_espaces)
        else:
            moyenne = 0

        blocs_filtres = []
        for typ, img in blocs:
            if typ == "espace":
                if img.shape[1] >= moyenne * 0.6:  # seuil plus souple
                    blocs_filtres.append((typ, img))
            else:
                blocs_filtres.append((typ, img))

        return blocs_filtres

    def redimensionner_image(self, image, largeur=28,hauteur=28):

        h, w = image.shape
        pixels_blancs = np.where(image > 128)
        if len(pixels_blancs[0]) == 0:
            return np.zeros((largeur,hauteur))

        haut, bas = pixels_blancs[0].min(), pixels_blancs[0].max()
        gauche, droite = pixels_blancs[1].min(), pixels_blancs[1].max()
        marge = 2
        haut = max(0, haut - marge)
        bas = min(h - 1, bas + marge)
        gauche = max(0, gauche - marge)
        droite = min(w - 1, droite + marge)
        lettre_cadree = image[haut:bas + 1, gauche:droite + 1]
        img_pil = Image.fromarray(lettre_cadree)
        img_pil.thumbnail((largeur - 4, hauteur - 4)) #on redimensionne la taille , mais on met une marge
        img_finale = Image.new('L', (largeur, hauteur), 0) #on crée une image 28*28 sur fond noir
        x = (largeur - img_pil.width) // 2 #on centre la lettre
        y = (hauteur - img_pil.height) // 2
        img_finale.paste(img_pil, (x, y))

        return np.array(img_finale)


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
            for i in random.sample(range(nb_it),nb_it):
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

#### Lecture d'une image :###
#Code pour charger les paramètres du réseau de neurone :

with open("parametres.pkl", "rb") as fichier:
    parametres_reseau = pickle.load(fichier)

def lire_image(image,seuil_ligne,seuil_colonne,parametres_reseau):
    ##on met les bons paramètres au réseau de neurone
    reseau=Reseau2Neurone(3,[784, 128, 26],0.01)
    reseau.reseau_poids = parametres_reseau["Reseau"]
    reseau.sommes = parametres_reseau["sommes"]
    reseau.activation = parametres_reseau["activation"]
    ##phase de prétraitement
    traitement=Traitement(image)
    p=traitement.decoupe_en_pixel()
    p2=traitement.binarisation(p)
    traitement.affiche_image(p2)
    h=traitement.histogramme(p2)
    ##sélection des lignes
    lignes=traitement.selection_lignes(h,p2,seuil_ligne)
    nb_lignes = len(lignes)
    print("Nb lignes trouvées :", nb_lignes)
    texte=""
    alphabet= ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    ##selection lettres et espaces
    for i in range(nb_lignes):

        hc=traitement.histogrammes_colonnes(lignes[i])
        blocs = traitement.selection_colonnes(hc, lignes[i], seuil_colonne)

        for typ, img in blocs:
            if typ == "espace":
                if img.shape[1]>6: #on ne compte les espaces que si l'image est assez large
                    texte += " "
            else:
                caractere = traitement.redimensionner_image(img, 28, 28)
                caractere = caractere.T
                caractere = caractere.flatten()/255
                reseau.forward(caractere)
                prediction = reseau.activation[reseau.nb_couche - 1]
                pred = np.argmax(prediction)
                texte += alphabet[pred]

    print(texte)

lire_image("image4.png",50,1,parametres_reseau)



#nb_essais=4
#nb_couche=3
#neurones_couche=[784, 128, 26]
#taux_apprentissage=0.01
#entrainement=Entrainement()
#meilleur_modele=entrainement.entrainement_consecutif(nb_essais,nb_couche,neurones_couche,taux_apprentissage)
#entrainement.test(nb_couche,neurones_couche,taux_apprentissage,meilleur_modele)

#with open("parametres2.pkl", 'wb') as f:
#    pickle.dump(meilleur_modele, f)
#print("fichier créé")
#plt.show()

