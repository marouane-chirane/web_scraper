# Web Scraping Agent with LangChain - Version Améliorée

Un agent de web scraping puissant qui utilise LangChain pour crawler des sites web, extraire du contenu structuré et créer une base de données vectorielle pour la recherche sémantique. Cette version améliorée inclut un support spécialisé pour les sites d'emploi comme jobup.ch.

## 🚀 Nouvelles Fonctionnalités

### **Scraping d'Offres d'Emploi (jobup.ch)**
- **Extraction spécialisée** des offres d'emploi avec pagination automatique
- **Support Selenium** pour les sites dynamiques
- **Extraction de données structurées** :
  - Titre du poste
  - Nom de l'entreprise
  - Localisation
  - Type de contrat
  - Taux d'activité
  - Salaire
  - Date de publication
- **Gestion intelligente de la pagination** pour récupérer toutes les offres
- **Sauvegarde automatique** en JSON avec statistiques détaillées

### **Fonctionnalités Générales**
- **Web Crawling Récursif** : Suit automatiquement et scrape les pages liées jusqu'à une profondeur spécifiée
- **Extraction de Contenu Structuré** :
  - En-têtes (H1, H2, H3)
  - Paragraphes
  - Listes (ordonnées et non ordonnées)
  - Tableaux
  - Images avec texte alternatif
  - Liens
  - Meta descriptions
- **Intégration Base de Données Vectorielle** :
  - Support pour FAISS et Chroma vector stores
  - Capacités de recherche sémantique
  - Chunking de texte efficace
- **Traitement Intelligent du Contenu** :
  - Chunking automatique du contenu
  - Normalisation du texte
  - Prévention des doublons
  - Validation et normalisation des URLs
- **Gestion Robuste des Erreurs** :
  - Retry automatique pour les requêtes échouées
  - Rate limiting
  - Rapports d'erreur complets
- **Export de Données** :
  - Export JSON de tout le contenu scrapé
  - Format de données structuré
  - Intégration facile avec d'autres outils

## 📦 Installation

1. Cloner le repository:
```bash
git clone <repository-url>
cd web-scraper
```

2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

3. Installer Chrome/Chromium (pour Selenium):
```bash
# Ubuntu/Debian
sudo apt-get install chromium-browser

# macOS
brew install --cask google-chrome

# Windows: Télécharger depuis https://www.google.com/chrome/
```

## 🚀 Utilisation

### **1. Scraping d'Offres d'Emploi (jobup.ch)**

#### Utilisation Simple:
```python
from web_scraper_agent import scrape_jobup_developer_offers

# Scraper toutes les offres de développeur
offers, vector_store = scrape_jobup_developer_offers()

# Scraper avec limitation de pages
offers, vector_store = scrape_jobup_developer_offers(max_pages=10)
```

#### Utilisation Avancée:
```python
from web_scraper_agent import WebScraperAgent

# Configuration pour jobup.ch
agent = WebScraperAgent(
    url="https://www.jobup.ch/fr/emplois/?term=developer",
    vector_store_type="faiss",
    use_selenium=True,  # Pour les sites dynamiques
    headless=True,      # Mode headless
    max_pages=20        # Limiter le nombre de pages
)

# Lancer le scraping
vector_store = agent.process_website()

# Accéder aux offres scrapées
offers = agent.job_offers
print(f"Nombre d'offres trouvées: {len(offers)}")
```

#### Script en Ligne de Commande:
```bash
# Scraper avec paramètres par défaut
python scrape_jobup.py

# Scraper avec options personnalisées
python scrape_jobup.py --pages 10 --term "python" --output "offres_python.json"

# Scraper sans Selenium (mode requests uniquement)
python scrape_jobup.py --no-selenium --pages 5
```

### **2. Scraping Standard (Autres Sites)**

```python
from web_scraper_agent import WebScraperAgent

# Scraper un site standard
agent = WebScraperAgent(
    url="https://example.com",
    vector_store_type="faiss",
    max_depth=2
)

# Lancer le scraping
vector_store = agent.process_website()

# Recherche sémantique
results = vector_store.similarity_search("Votre question ici")
```

### **3. Test du Scraper**

```bash
# Tester le scraper jobup.ch
python test_jobup.py
```

## Configuration

The agent can be configured with the following parameters:

- `url`: The starting URL for scraping
- `vector_store_type`: Choose between "faiss" or "chroma"
- `max_depth`: Maximum depth for recursive crawling
- `chunk_size`: Size of text chunks for vector store (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

## Output

The scraper generates:
1. A detailed console output showing:
   - Crawling statistics
   - Content structure
   - Extracted information
2. A JSON file containing all scraped data
3. A vector store for semantic search

## Dependencies

- langchain
- beautifulsoup4
- requests
- faiss-cpu
- chromadb
- python-dotenv
- tiktoken
- sentence-transformers

## Best Practices

1. **Rate Limiting**: The scraper includes built-in delays between requests to avoid overwhelming servers
2. **Error Handling**: Automatic retries and comprehensive error reporting
3. **Content Validation**: Checks for valid content before processing
4. **Memory Management**: Efficient handling of large websites
5. **Data Organization**: Structured storage of scraped content

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.