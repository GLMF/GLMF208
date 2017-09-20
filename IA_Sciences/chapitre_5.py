#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import des dépendances
from clarifai.rest import ClarifaiApp
import sys

# Initialisation de Clarifai avec les identifiants fournis lors de l’inscription
app = ClarifaiApp("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# Chargement du modèle "général"
model = app.models.get("general-v1.3")

# Lancement de la prédiction sur une image
result = model.predict_by_filename(filename='sacs1.jpg')

# On teste le code retour. 10 000 = OK.
# Pourquoi 10 000... bonne question!    
if result['status']['code'] != 10000:
    print("Erreur")
    sys.exit(1)
    
# On affiche chaque concept retourné par Clarifai concernant notre image
for concept in result["outputs"][0]["data"]["concepts"]:
    print(concept)
