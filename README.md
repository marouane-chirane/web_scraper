# Agent de Scraping Web avec LangChain

Cet agent permet de scraper un site web, de traiter son contenu et de le stocker dans une base de données vectorielle (FAISS ou Chroma) pour des recherches sémantiques ultérieures.

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Modifiez le fichier `web_scraper_agent.py` pour spécifier l'URL du site que vous souhaitez scraper :
```python
url = "https://votre-site.com"  # Remplacez par l'URL de votre choix
```

2. Choisissez le type de vector store que vous souhaitez utiliser (FAISS ou Chroma) :
```python
agent = WebScraperAgent(url, vector_store_type="faiss")  # ou "chroma"
```

3. Exécutez le script :
```bash
python web_scraper_agent.py
```

## Fonctionnalités

- Scraping automatique du contenu web
- Découpage du texte en chunks pour un meilleur traitement
- Création d'une base de données vectorielle pour la recherche sémantique
- Support pour FAISS et Chroma comme bases de données vectorielles
- Utilisation de modèles d'embeddings de Hugging Face

## Personnalisation

Vous pouvez personnaliser les paramètres suivants dans le code :
- Taille des chunks (`chunk_size`)
- Chevauchement des chunks (`chunk_overlap`)
- Modèle d'embeddings utilisé
- Type de vector store (FAISS ou Chroma)