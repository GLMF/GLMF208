#!/usr/bin/python
# -*- coding: utf-8 -*-

# On importe les librairies nécessaires
import freenect
import cv2
import numpy as np
import time
import sys

# Cette fonction va récupérer une frame depuis le flux vidéo.
# La frame récupérée sera celle de l'instant T
def get_image():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

# On gère un éventuel paramètre optionnel pour construire le nom du fichier cible
if len(sys.argv) == 2:
    # Si le paramètre est présent, ce sera le nom du fichier.
    img_file = sys.argv[1]
else:
    # Sinon on construit le nom du fichier depuis le timestamp courant.
    img_file = "image_{0}.png".format(time.time())

# On appelle la fonction qui récupère la frame et on affiche l'image capturée dans une fenêtre.
frame = get_image()
cv2.imshow("image", frame)

# On sauvegarde l'image dans le fichier
cv2.imwrite(img_file, frame)

# On attend que l'utilisateur appuie sur une touche et on ferme la fenêtre
cv2.waitKey(0)
cv2.destroyAllWindows()
