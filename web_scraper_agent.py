from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
from typing import List, Union, Set, Dict
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import time
from urllib.parse import urljoin, urlparse
import re

# Charger les variables d'environnement
load_dotenv()

class WebScraperAgent:
    def __init__(self, url: str, vector_store_type: str = "faiss", max_depth: int = 2):
        self.base_url = url
        self.vector_store_type = vector_store_type
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
        }
        self.visited_urls: Set[str] = set()
        self.max_depth = max_depth
        self.all_data: Dict[str, dict] = {}
        
    def is_valid_url(self, url: str) -> bool:
        """Vérifie si l'URL est valide et appartient au même domaine"""
        try:
            parsed_base = urlparse(self.base_url)
            parsed_url = urlparse(url)
            return (
                parsed_url.netloc == parsed_base.netloc and
                not url.endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.doc', '.docx')) and
                not any(ext in url.lower() for ext in ['mailto:', 'tel:', 'javascript:', '#'])
            )
        except:
            return False

    def normalize_url(self, url: str) -> str:
        """Normalise l'URL en URL absolue"""
        return urljoin(self.base_url, url)

    def extract_content(self, soup: BeautifulSoup, url: str) -> dict:
        """Extrait le contenu structuré d'une page"""
        return {
            "url": url,
            "title": soup.title.string if soup.title else "Pas de titre",
            "meta_description": soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else "Pas de description",
            "headers": {
                "h1": [h1.text.strip() for h1 in soup.find_all('h1')],
                "h2": [h2.text.strip() for h2 in soup.find_all('h2')],
                "h3": [h3.text.strip() for h3 in soup.find_all('h3')]
            },
            "paragraphs": [p.text.strip() for p in soup.find_all('p') if p.text.strip()],
            "lists": {
                "ul": [ul.text.strip() for ul in soup.find_all('ul')],
                "ol": [ol.text.strip() for ol in soup.find_all('ol')]
            },
            "tables": [table.text.strip() for table in soup.find_all('table')],
            "images": [img.get('alt', '') for img in soup.find_all('img') if img.get('alt')],
            "links": [a.get('href') for a in soup.find_all('a', href=True)],
            "raw_text": soup.get_text(separator=' ', strip=True),
            "scraping_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def scrape_page(self, url: str, depth: int = 0) -> dict:
        """Scrape une page spécifique avec gestion des erreurs et retries"""
        if depth > self.max_depth or url in self.visited_urls:
            return {}

        self.visited_urls.add(url)
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                data = self.extract_content(soup, url)
                self.all_data[url] = data
                
                if depth < self.max_depth:
                    # Récupérer tous les liens de la page
                    links = [self.normalize_url(link) for link in data['links']]
                    valid_links = [link for link in links if self.is_valid_url(link) and link not in self.visited_urls]
                    
                    # Scraper récursivement les liens valides
                    for link in valid_links:
                        time.sleep(1)  # Délai entre les requêtes
                        self.scrape_page(link, depth + 1)
                
                return data
                
            except requests.exceptions.RequestException as e:
                print(f"Tentative {attempt + 1}/{max_retries} échouée pour {url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"Impossible de scraper {url} après {max_retries} tentatives")
                    return {}
            except Exception as e:
                print(f"Erreur inattendue pour {url}: {str(e)}")
                return {}

    def display_scraping_results(self):
        """Affiche les résultats du scraping de manière structurée"""
        if not self.all_data:
            print("\n❌ ERREUR: Aucun contenu n'a été scrapé")
            return

        print("\n" + "="*80)
        print(f"RÉSULTATS DU SCRAPING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        total_pages = len(self.all_data)
        print(f"\n📊 STATISTIQUES GLOBALES")
        print("-"*40)
        print(f"Nombre total de pages scrapées: {total_pages}")
        print(f"URLs visitées: {len(self.visited_urls)}")
        
        for url, data in self.all_data.items():
            print(f"\n\n📄 PAGE: {url}")
            print("-"*40)
            print(f"Titre: {data.get('title', 'N/A')}")
            print(f"Description: {data.get('meta_description', 'N/A')}")
            
            print("\n📑 STRUCTURE DU CONTENU")
            for header_type, headers in data.get('headers', {}).items():
                if headers:
                    print(f"\n{header_type.upper()}:")
                    for i, header in enumerate(headers, 1):
                        print(f"  {i}. {header}")
            
            print("\n📝 PARAGRAPHES")
            for i, para in enumerate(data.get('paragraphs', []), 1):
                if len(para) > 100:
                    print(f"{i}. {para[:100]}...")
                else:
                    print(f"{i}. {para}")
            
            print("\n📋 LISTES")
            for list_type, lists in data.get('lists', {}).items():
                if lists:
                    print(f"\n{list_type.upper()}:")
                    for i, list_content in enumerate(lists, 1):
                        print(f"  {i}. {list_content[:100]}...")
            
            print("\n📊 TABLEAUX")
            for i, table in enumerate(data.get('tables', []), 1):
                print(f"{i}. {table[:100]}...")
            
            print("\n🖼️ IMAGES")
            for i, alt in enumerate(data.get('images', []), 1):
                print(f"{i}. {alt}")
        
        # Sauvegarder les résultats dans un fichier JSON
        filename = f"scraping_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Résultats sauvegardés dans: {filename}")

    def create_chunks(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Divise le texte en chunks"""
        if not text:
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_text(text)

    def create_vector_store(self, chunks: List[str]):
        """Crée et retourne le vector store"""
        if not chunks:
            raise ValueError("Aucun contenu à indexer dans le vector store")
            
        if self.vector_store_type.lower() == "faiss":
            return FAISS.from_texts(chunks, self.embeddings)
        elif self.vector_store_type.lower() == "chroma":
            return Chroma.from_texts(chunks, self.embeddings)
        else:
            raise ValueError("Type de vector store non supporté. Utilisez 'faiss' ou 'chroma'")

    def process_website(self):
        """Processus complet de scraping et de création du vector store"""
        # Scraping récursif du site
        self.scrape_page(self.base_url)
        
        # Afficher les résultats du scraping
        self.display_scraping_results()
        
        if not self.all_data:
            raise ValueError("Impossible de créer le vector store car aucun contenu n'a été scrapé")
        
        # Création des chunks à partir de tout le contenu
        all_text = " ".join([data.get('raw_text', '') for data in self.all_data.values()])
        chunks = self.create_chunks(all_text)
        
        if not chunks:
            raise ValueError("Aucun chunk n'a pu être créé à partir du contenu")
        
        # Création du vector store
        vector_store = self.create_vector_store(chunks)
        
        return vector_store

def main():
    try:
        # Exemple d'utilisation
        url = "https://www.algerietelecom.dz"  # Remplacez par l'URL de votre choix
        agent = WebScraperAgent(url, vector_store_type="faiss", max_depth=2)
        vector_store = agent.process_website()
        
        # Exemple de recherche sémantique
        print("\n🔍 RECHERCHE SÉMANTIQUE")
        print("-"*40)
        query = "Quelles sont les offres internet disponibles ?"
        print(f"Question: {query}")
        print("\nRésultats:")
        docs = vector_store.similarity_search(query)
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")

if __name__ == "__main__":
    main() 