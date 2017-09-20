#!/usr/bin/python
# -*- coding: utf-8 -*-

# On importe les dépendances nécessaires
import freenect
import cv2
import numpy as np

# Cette fonction va récupérer une frame depuis le flux vidéo.
# La frame récupérée sera cette de l'instant T
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

# La partie du code appelée quand on lance le script python
if __name__ == "__main__":
    # Nous capturons le flux indéfiniment
    while 1:
        # Appel de la fonction qui récupère la frame
        frame = get_video()
        # Affichage de la frame dans une fenêtre nommée "Flux vidéo"
        cv2.imshow('Flux vidéo',frame)

        # Pour permettre une sortie propre de la boucle, si on appuie 
        # sur la touche "Esc", on quitte la boucle infinie
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    # Suppression de la fenêtre
    cv2.destroyAllWindows()
