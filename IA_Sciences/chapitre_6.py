#!/usr/bin/python
# -*- coding: utf-8 -*-

import imutils
import time
import cv2
import freenect
import numpy as np
import os
from clarifai.rest import ClarifaiApp
import sys

### La fonction pour récupérer une frame vidéo depuis le kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array


### La classe de travail
class Detection:

    def __init__(self, cb_get_frame, 
                 min_object_area = 200, 
                 object_movement_threshold = 10, 
                 archive_folder = "/tmp/",
                 do_recognize = False,                
                 clarifai_client_id = None,
                 clarifai_client_secret = None):
        """ Constructeur dans lequel on va initialiser les variables.
            Paramètres : 
            @cb_get_frame : callback vers la fonction qui récupère la frame courante du flux vidéo.
            @min_object_area : aire minimale en pixels des objets détectés dans l'image.
            @object_movement_threshold : seuil de tolérance pour le mouvement des objets .
            @archive_folder : dossier qui sera utilisé pour stocker des images sous forme de fichiers.
            @do_recognize : True/False : activer (True) ou non l'identification des objets détectés avec Clarifai.
            @clarifai_client_id : client id pour utiliser l'API de Clarifai.
            @clarifai_client_secret : clé du client pour utiliser l'API de Clarifai.
        """
      
        # On affecte les valeurs des paramètres à des variables de la classe
        self.get_frame = cb_get_frame
        self.min_object_area = min_object_area
        self.object_movement_threshold = object_movement_threshold
        self.archive_folder = archive_folder
        self.do_recognize = do_recognize
        self.clarifai_client_id = clarifai_client_id
        self.clarifai_client_secret = clarifai_client_secret

        # On initialise la frame pour l'affichage de l'image
        self.frame = None

        # On initialise une frame 'outil' pour afficher les objets détectés hors de l'image.
        # La taille de cette frame (500px de haut pour 1500px de large) sera à adapter en fonction de l'usage. 
        # Dans le cadre de cet article, elle permet d'afficher une dizaine d'objets détectés.
        self.frame_found_objects = np.zeros((500, 1500, 3), np.uint8)

        # On intialise la liste des objets détectés. 
        # Elle contiendra cette structure : 
        #   { "id1" : 
        #       {
        #          "x" : coordonnées
        #          "y" : coordonnées
        #          "w" : coordonnées
        #          "h" : coordonnées
        #          "img" : l'image de l'objet lors de sa première apparition
        #          "recognition" : une liste de tags qui décrivent l'objet
        #          "first_seen" : le moment où a été vu l'objet la première fois
        #          "last_seen" : le dernier moment où a été vu l'objet
        #       },
        #     "id2" : 
        #       { ... },
        #     ...
        #  }
        self.objects = {}

        # On initialise un compteur qui contient l'id du dernier objet trouvé.
        # Il sera incrémenté au prochain objet trouvé.
        self.last_object_id = 0

        # Si on souhaite activer l'identification des objets
        if self.do_recognize:
            # Initialisation de Clarifai avec l'identifiant et sa clé.
            app = ClarifaiApp(self.clarifai_client_id, self.clarifai_client_secret)
            # Création du model Clarifai. Ici on utilise le modèle général.
            self.model = app.models.get("general-v1.3")




    def recognize_object(self, obj_id, filepath):
        """ Fonction qui va appeler Clarifai pour identifier ce que contient l'image.
            Paramètres :
            @obj_id : identifiant de l'objet concerné. Il va servir à compléter la liste des tags 'recognition' de l'objet.
            @filepath : chemin vers le fichier qui contient l'image de l'objet à identifier.
        """
        # On appelle Clarifai en lui donnant notre fichier image
        result = self.model.predict_by_filename(filename=filepath)
        
        # On teste le code de statut. Si il est différent de 10 000, il y a eu une erreur.
        # En cas d'erreur, on arrête là.
        if result['status']['code'] != 10000:
            print("Erreur lors de l'appel à Clarifai")
            return
        
        # Pour un résultat correct, on va parcourir les différents concepts identifiés sur l'image.
        # On ne prend que les 5 premiers : les premiers sont les plus pertinents.
        idx = 0
        for concept in result["outputs"][0]["data"]["concepts"]:
            # On ajoute le concept à la liste des tags de l'objet.
            self.objects[obj_id]["recognition"].append(concept["name"])
            # On gère le compteur. Si on a eu 5 tags, on quitte la boucle et la fonction.
            idx += 1
            if idx > 5:
                break




    def detect_objects(self):
        """ Fonction à appeler pour lancer la détection des objets.
        """
        # On va commencer par initialiser à None la frame de référence : il s'agira de la première frame capturée.
        frame_reference = None

        # Pour chaque frame du flux vidéo...
        while True:

            # On récupère une frame du flux vidéo en appelant la fonction de callback
            self.frame = self.get_frame()
         
            # On diminiue la taille de la frame.
            # Si l'image est trop grosse, les calculs sont plus longs et consomment plus de ressources.
            # Comme nous recherchons de gros éléments, nous pouvons redimensionner l'image dans une petite taille (500px de large)
            self.frame = imutils.resize(self.frame, width=500)

            # On réalise une copie de l'image qui ne sera pas modifiée.
            # En effet, l'image originale sera agrémentée de rectangles autour des objets et d'autres informations.
            self.frame_raw = self.frame.copy()  

            # On copie la frame dans une autre frame en la passant en niveaux de gris. 
            # Passer en niveau de gris est nécessaire pour réaliser des opérations de soustraction qui ont du sens sur les images.
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # On copie la frame en niveaux de gris dans une autre frame sur laquelle sera appliqué un flou gaussien.
            # L'application du flou gaussien permet de lisser des imperfections de l'image (mouche qui vole, petits bruits, ...).
            frame_gray_blurred = cv2.GaussianBlur(frame_gray, (11, 11), 0)

            # Pour finir, nous copions la frame grise et floutée dans une frame de "travail".           
            frame_work = frame_gray_blurred

            # Si la frame de référence n'est pas définie, on lui assigne la frame de travail et on arrête le traitement de cette itération de la boucle.
            if frame_reference is None:
                frame_reference = frame_work
                continue
        
            # On calcule la différence absolue entre la frame en cours et la frame de référence.
            # Ceci va donner une frame avec un fond noir et en niveaux de gris les pixels qui ont changés.
            frame_delta = cv2.absdiff(frame_reference, frame_work)
        
            # On transforme tous les niveaux de gris supérieurs à 45 (sur une échelle allant de 0 à 255) en blanc.
            # Ceci permet de supprimer des ombres légères de la détection, mais aussi de se débarasser de bruits.
            frame_threshold = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

            # On réalise une dilation de l'image pour combler les défauts des formes blanches (pour faire simple, on tente de remplir les trous).
            frame_dilated = cv2.dilate(frame_threshold, None, iterations=2)
        
            # Et on trouve les boîtes qui encadrent les formes blanches.
            (contours, _) = cv2.findContours(frame_dilated.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
         
            # On parcourt la liste des contours trouvés
            for c in contours:
                # On ignore les formes de taille négligeables
                if cv2.contourArea(c) < self.min_object_area:
                    continue
         
                # Pour les formes non négligeables, nous récupérons leurs coordonnées,
                # puis on dessine un rectangle de couleur verte (0, 255, 0) et d'épaisseur 2 autour.
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # On appelle la fonction qui maintient à jour la liste des objets.
                # C'est cette fonction qui va voir si l'objet est déjà présent dans la liste des objets trouvés ou non.
                # On passe en arguments les coordonnées de l'objet dans l'image.
                self.update_objects_list(x, y, w, h)


            # On appelle la fonction qui va analyser tous les objets qui ont été ajoutés à la liste pour cette image, 
            # mais aussi pour ceux qui étaient déjà présents dans la liste auparavant.
            self.analyse_objects()
         
            # On affiche les différentes fenêtres.
            # Dans la pratique, 3 fenêtres suffiraient : 
            # - l'image de base avec les cadres autour des objets, 
            # - l'image de base brute (pour une meilleure lecture sans les cadres),
            # - l'image qui affiche la liste des objets trouvés.
            # Pour l'exercice, il est intéressant d'afficher chacune des frames afin de mieux comprendre le processus de recherche.
            cv2.imshow("frame", self.frame)
            cv2.moveWindow("frame", 0, 0)
            cv2.imshow("gray", frame_gray)
            cv2.moveWindow("gray", 500, 0)
            cv2.imshow("gray blurred", frame_gray_blurred)
            cv2.moveWindow("gray blurred", 1000, 0)
            cv2.imshow("delta", frame_delta)
            cv2.moveWindow("delta", 0, 400)
            cv2.imshow("threshold", frame_threshold)
            cv2.moveWindow("threshold", 500, 400)
            cv2.imshow("dilated", frame_dilated)
            cv2.moveWindow("dilated", 1000, 400)

            cv2.imshow("reference", frame_reference)
            cv2.moveWindow("reference", 1400, 800)
        
            cv2.imshow("found objects", self.frame_found_objects)
            cv2.moveWindow("found objects", 0, 1000)

            # On va gérer ici la pression de la touche 'q' pour quitter proprement le programme.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                # On supprime les fenêtres 
                cv2.destroyAllWindows()
                # On quitte la boucle infinie
                break



    def update_objects_list(self, x, y, w, h):
        """ Fonction qui ajoute (ou pas) un objet à la liste des objets trouvés.
            Paramètres : 
            @x, @y : coordonnées de l'objet
            @w, @h : largeur et hauteur de l'objet

            Le principe est assez basique : on considère qu'un objet est déjà présent dans la liste si on a déjà un objet à peu près au même endroit et à peu près de la même taille dans la liste.
        """
        # On initialise un flag comme quoi l'objet n'est pas déjà présent dans la liste.
        found = False
        
        # On parcourt la liste des objets
        for id in self.objects:
            # On récupère l'objet
            obj = self.objects[id]
            # Et on regarde si la différence entre les coordonnées et les dimensions est inférieure au seuil que nous avons défini.
            # Ce seuil de tolérance est nécessaire car à cause du bruit, de petites ombres et autres détail, un objet détecté peu bouger de quelques pixels en plus en haut, en bas, à gauche ou à droite.
            if abs(obj["x"] - x) <= self.object_movement_threshold and \
               abs(obj["y"] - y) <= self.object_movement_threshold and \
               abs(obj["w"] - w) <= self.object_movement_threshold and \
               abs(obj["h"] - h) <= self.object_movement_threshold:
                # Si c'est le cas, on avait déjà identifié cet objet.
                found = True
                # Nous mettons à jour le moment de la dernière visualisation de l'objet.
                obj["last_seen"] = time.time()

        # Si l'objet n'a pas été trouvé dans la liste, il s'agit d'un nouvel objet
        if not found:
            # On incrémente l'identifiant du dernier objet trouvé 
            self.last_object_id += 1
            # Et on ajoute à la liste l'objet avec ses informations. L'image est initialisée à None, elle sera remplie un peu plus tard.
            self.objects[self.last_object_id] = {
                                                  "img" : None,
                                                  "recognition" : [],
                                                  "x" : x,
                                                  "y" : y,
                                                  "w" : w,
                                                  "h" : h,
                                                  "first_seen" : time.time(),
                                                  "last_seen" : time.time()}


    def analyse_objects(self):
        """ Fonction qui analyse la liste des objets trouvés pour voir si :
            - l'objet est trop récent pour remonter comme objet trouvé, auquel cas on le garde pour le moment
            - l'objet a disparu. Si oui, on analyse depuis combien de temps et si on le considère comme disparut ou pas
            - l'objet devient considéré comme trouvé et on l'identifie comme tel
        """
        # On initialise la liste des objets à supprimer à l'issue du parcours de la liste des objets
        objects_to_del = []

        # On réinitialise la frame qui affiche les objets détectés
        self.frame_found_objects = np.zeros((500, 1500, 3), np.uint8)
  
        # On initialise à zéro l'offset d'affichage des objets détectés dans la frame concernée.
        offset_display = 0

        # On initialise l'espace qui sera mis entre l'affichage des objets détectés dans la frame concernée.
        padding = 10


        # On parcourt la liste des objets connus
        for id in self.objects:

            # Histoire de simplifier l'écriture du code, on copie l'objet dans une variable
            obj = self.objects[id]

            # On calcule le temps pendant lequel l'objet a été vu pour la première et la dernière fois
            # Ce temps inclut les disparitions de l'objet (par exemple si il a été caché par un groupe de personnes)
            time_seen = int(obj["last_seen"] - obj["first_seen"])

            # On calcule le temps depuis la première apparition de l'objet à maintenant
            time_since_first_seen = int(time.time() - obj["first_seen"])

            # On calcule le temps depuis la dernière apparition de l'objet à maintenant
            time_since_last_seen = int(time.time() - obj["last_seen"])

            # Début de la phase de filtrage...
            # Dans toute cette phase, les timings (en secondes) sont volontairement courts. 
            # En 'production', il faudra les allonger et adapter leur valeur avec l'expérience
            OBJECT_TOO_YOUNG_TIME_MIN = 5        # secondes
            OBJECT_TOO_YOUNG_TIME_MAX = 10       # secondes
            OBJET_HIDDEN_FACTOR = 2              # ratio de multiplication
        
            # Objets perdus de vue il y a peu et restés peu de temps à l'écran : on les supprime
            if time_since_last_seen > OBJECT_TOO_YOUNG_TIME_MIN and time_seen <= OBJECT_TOO_YOUNG_TIME_MAX:
                print(u"Suppression de l'objet (disparu et trop jeune) : {0}".format(id))
                objects_to_del.append(id)

            # Objets restés un peu à l'écran mais perdus de vue depuis 2 fois leur temps d'apparition : on les supprime
            # Ici on considère qu'un objet a le droit de disparaître  pendant au moins 2 fois son temps d'apparition.
            if time_seen > OBJECT_TOO_YOUNG_TIME_MAX and time_since_last_seen > OBJET_HIDDEN_FACTOR*time_seen:
                print(u"Suppression de l'objet (disparu mais ayant été un objet détecté pendant un moment) : {0}".format(id))
                objects_to_del.append(id)

            # Objets présents à l'écran depuis un moment et candidats à être de potentiels objets perdus
            if time_seen > OBJECT_TOO_YOUNG_TIME_MAX:
                # Quelques raccourcis pour simplifier l'écriture du code...
                ix = obj["x"] 
                iy = obj["y"] 
                iw = obj["w"] 
                ih = obj["h"] 

                # Si on n'a pas déjà une image pour l'objet dans la liste des objets :
                # 1. on la stocke. Ceci signifie qu'on gardera en image de référence la première apparition de l'objet.
                # 2. si on a choisit de requêter le service Clarifai pour identifier l'objet, on appelle Clarifai.
                if obj["img"] is None:
                    # On stocke l'image
                    print(u"Nouvel objet immobile identifié (id={0})".format(id))
                    obj["img"] = self.frame_raw[iy:iy+ih,ix:ix+iw]

                    # Si on souhaite appeler Clarifai...
                    if self.do_recognize:
                        # On va sauvegarder l'image sur le disque dans un fichier, car Clarifai attend un fichier.
                        print(u"Appel de clarifai pour identification...")
                        filepath = os.path.join(self.archive_folder,  "object_found_{0}.jpg".format(id))
                        cv2.imwrite(filepath, obj["img"])

                        # Et on appelle Clarifai
                        self.recognize_object(id, filepath)


                # Maintenant on repasse aux opérations effectuées sur les objets détectés à chaque passage dans la fonction...
                # Tout d'abord, on affiche sur l'image des objets trouvés l'image stockée de l'objet et son id en texte.
                # Notez l'usage de l'offset horizontal offset_display
                self.frame_found_objects[30:30+ih,offset_display:offset_display+iw] = obj["img"]
                cv2.putText(self.frame_found_objects, 
                            "Id={0}".format(id), 
                            (offset_display, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2)

                # Ensuite, on va afficher la liste des labels récupérés lors de la phase d'identification du contenu de l'image.
                # Les labels seront affichés les uns sous les autres, nous déclarons donc un nouvel offset vertical pour les labels.
                offset_label = 40 + ih
                for label in obj["recognition"]:
                    cv2.putText(self.frame_found_objects, 
                                label,
                                (offset_display, offset_label), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 255, 255), 1)
                    offset_label += 30
                    
                # On décale l'offset vertical pour l'objet suivant.
                # On considère que nous avons besoin d'une largeur minimale de 100 pixels pour pouvoir afficher correctement les labels.
                if iw < 100:
                    offset_display += 10 + 100
                else:
                    offset_display += 10 + iw

                # Par simplicité, on définit une couleur pour les manipulations qui suivent.
                color = (255, 255, 0)
                
                # On dessine le cadre autour de l'objet
                cv2.rectangle(self.frame, (obj["x"], obj["y"]), (obj["x"] + obj["w"], obj["y"] + obj["h"]), color, 2)
    
                # On affiche l'id de l'objet et le temps écoulé depuis que l'objet est découvert (et le temps depuis la dernière fois qu'on l'a vu)
                cv2.putText(self.frame, "Id={0} / {1}s / {2}s".format(id, time_seen, time_since_last_seen), (obj["x"], obj["y"]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)   

        # Suppression des objets marqués à supprimer
        for id in objects_to_del:
            del self.objects[id]



if __name__ == "__main__":
    # On définit les paramètres d'identification de l'api Clarifai
    client_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    client_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    # On initialise la détection
    w = Detection(get_video, 
                  min_object_area = 200, 
                  object_movement_threshold = 20, 
                  archive_folder = "/tmp", 
                  do_recognize = True, 
                  clarifai_client_id = client_id,
                  clarifai_client_secret = client_secret)
    # On lance la détection
    w.detect_objects()
