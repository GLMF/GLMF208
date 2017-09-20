#!/usr/bin/python
# -*- coding: utf-8 -*-

import imutils
import time
import cv2
import numpy as np
import sys

if __name__ == "__main__":
    # On teste le nombre de paramètres
    if len(sys.argv) < 3:
        print("Usage : {0} <image 1> <image 2>".format(sys.argv[0]))
        sys.exit(1)

    # On charge l'image n°1
    img1 = cv2.imread(sys.argv[1])
    cv2.imshow("img1", img1)

    # On charge l'image n°2
    img2 = cv2.imread(sys.argv[2])
    cv2.imshow("img2", img2)

    # On fait la différence entre les deux images en couleur.
    # On le fait dans les 2 sens : image 1 - image 2 et image 2 - image 1
    img_delta1 = cv2.subtract(img1, img2)
    cv2.imshow("delta 1 - 2", img_delta1)
    img_delta2 = cv2.subtract(img2, img1)
    cv2.imshow("delta 2 - 1", img_delta2)

    # On fait la différence absolue entre les deux images en couleur.
    img_delta_abs = cv2.absdiff(img1, img2)
    cv2.imshow("delta abs" , img_delta_abs)

    ### chapitre 4.3 ###

    # On convertit les 2 images en niveaux de gris.
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    cv2.imshow("gray1" , gray1)
    cv2.imshow("gray2" , gray2)
    
    # On fait la différence entre les deux images en gris.
    # On le fait dans les 2 sens : 1-2 et 2-1
    gray_delta1 = cv2.subtract(gray1, gray2)
    cv2.imshow("delta gris", gray_delta1)
    gray_delta2 = cv2.subtract(gray2, gray1)
    cv2.imshow("delta gris 2", gray_delta2)

    # On fait la différence absolue entre les deux images en gris.
    gray_delta_abs = cv2.absdiff(gray1, gray2)
    cv2.imshow("abs delta gris" , gray_delta_abs)

    ### chapitre 4.4 ###

    # On applique un seuil
    for threshold in (5, 25, 45, 65):
        threshold_img = cv2.threshold(gray_delta_abs, threshold, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("threshold {0}".format(threshold), threshold_img)

    ### chapitre 4.6 ###

    # On applique différents flous sur une image en niveaux de gris
    for blur in (11, 21, 31, 41):
        blur1 = cv2.GaussianBlur(gray1, (blur, blur), 0)
        cv2.imshow("blur1 {0}".format(blur), blur1)

    # On applique un flou sur les deux images en niveaux de gris et on fait la différence
    for blur in (11, 21, 31, 41):
        blur1 = cv2.GaussianBlur(gray1, (blur, blur), 0)
        blur2 = cv2.GaussianBlur(gray2, (blur, blur), 0)
        blur_diff = cv2.absdiff(blur1, blur2)
        cv2.imshow("blur diff {0}".format(blur), blur_diff)

    ### chapitre 4.7 ###

   # On réalise une détection de contour sur la différence d'images en niveaux de gris avec un flou de 11 et un seuil de 45 (threshold_img)
    idx = 0
    for method in (cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE):
        idx += 1
        img = threshold_img.copy()
        (contours, _) = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                         method)
        print("Dimension des contours :")
        for c in contours:
            print(len(c))
        cv2.drawContours(img, contours, -1, (255,255,255), 1)
        cv2.imshow("contours, method {0}".format(idx), img)




        # On dessine un cadre autour de chaque contour
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
        cv2.imshow("bounding boxes, method {0}".format(idx), img)


