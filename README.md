# Mnistify

![App screenshot](public/og-image.png)

Live demo: https://mnistify.vercel.app

## Aperçu

Bienvenue sur Mnistify, un petit projet qui rend le machine learning amusant et accessible directement dans ton navigateur. Le but était de créer une expérience interactive où tu peux jouer avec la reconnaissance de chiffres (dataset MNIST) en utilisant différents modèles de deep learning.

Tout se passe côté client grâce à WebGPU - donc pas de serveur coûteux, juste ton GPU qui travaille !

## Fonctionnalités

🎨 **Interface intuitive**

- Dessine tes propres chiffres
- Visualisation en temps réel des prédictions

📊 **Visualisation des performances**

- Graphiques interactifs des prédictions
- Comparaison des temps d'inférence entre modèles

🔄 **Modèles disponibles**

- MLP (Multi-Layer Perceptron) : rapide et léger
- CNN (Convolutional Neural Network) : plus précis

## 🚀 Tech Stack

### 🎯 Frontend

- **[React](https://react.dev/)** – Bibliothèque JavaScript.
- **[Next.js](https://nextjs.org/)** – Framework React pour applications web.
- **[shadcn/ui](https://ui.shadcn.com/)** – Bibliothèque de composants UI.
- **[Tailwind CSS](https://tailwindcss.com/)** – Framework CSS.
- **[Recharts](https://recharts.org/)** – Bibliothèque de visualisation de données basée sur React.

### 🧠 Machine Learning

- **[Python 3.13.4](https://docs.python.org/3.13/)**
- **[TinyGrad](https://github.com/tinygrad/tinygrad)** – Framework de deep learning.
- **[WebGPU](https://www.w3.org/TR/webgpu/)** – API pour inférence côté client.
- **[SafeTensors](https://huggingface.co/docs/safetensors/index)** – Format sécurisé pour le partage de modèles.

## Résumé des modèles

### Meilleur MLP

| Type de couche | Détails                            |
| -------------- | ---------------------------------- |
| Entrée         | 784 neurones (image aplatie en 1D) |
| Couche dense 1 | 512 neurones avec activation SiLU  |
| Couche dense 2 | 512 neurones avec activation SiLU  |
| Sortie         | 10 neurones                        |

Précision finale (test): 94.49%

### Meilleur CNN

| Type de couche | Détails                                   |
| -------------- | ----------------------------------------- |
| Entrée         | Image 1 canal (28×28 pixels)              |
| Convolution 1  | 32 filtres de taille 5×5, activation SiLU |
| Convolution 2  | 32 filtres de taille 5×5, activation SiLU |
| Normalisation  | Normalisation par lots (32 canaux)        |
| Pooling        | Max-pooling (réduction de taille)         |
| Convolution 3  | 64 filtres de taille 3×3, activation SiLU |
| Convolution 4  | 64 filtres de taille 3×3, activation SiLU |
| Normalisation  | Normalisation par lots (64 canaux)        |
| Pooling        | Max-pooling (réduction de taille)         |
| Aplatissement  | Conversion en vecteur 1D                  |
| Couche dense   | 576 neurones vers 10 neurones             |

Précision finale (test): 98.22%

## Installation & exécution locale

Pré-requis

- Node.js (recommandé >= 18)
- Python 3.9+ (si vous voulez ré-entraîner les modèles)
- Un environnement WebGPU compatible (navigateur récent Chrome/Edge/Firefox Nightly avec drapeau WebGPU si nécessaire)

Frontend

```bash
# à la racine du projet
npm install
npm run dev
```

Ouvrez http://localhost:3000 pour voir l'application.

Entraînements des modèles

```bash
cd python
python -m pip install -r requirements.txt
python model_training/mlp.py
python model_training/cnn.py
```

## Journal d'hyperparamètres

Voir [`HYPERPARAMETERS.md`](/HYPERPARAMETERS.md).

## Rétrospective de projet

Ajoutez ici 3-6 phrases sur les défis techniques rencontrés et les apprentissages (ex : limitations de tinygrad, adaptation des modèles pour WebGPU, compromis quant à la taille du modèle vs latence, etc.).
