from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
from typing import List, Union, Set, Dict, Optional
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import time
from urllib.parse import urljoin, urlparse, parse_qs
import re
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebScraperAgent:
    def __init__(self, url: str, vector_store_type: str = "faiss", max_depth: int = 2, 
                 use_selenium: bool = False, headless: bool = True, max_pages: int = None):
        self.base_url = url
        self.vector_store_type = vector_store_type
        # Utilisation du meilleur modèle d'embedding multilingue
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.visited_urls: Set[str] = set()
        self.max_depth = max_depth
        self.all_data: Dict[str, dict] = {}
        self.use_selenium = use_selenium
        self.headless = headless
        self.max_pages = max_pages
        self.driver = None
        self.job_offers: List[Dict] = []
        
        # Configuration spécifique pour jobup.ch
        self.is_jobup = 'jobup.ch' in url
        if self.is_jobup:
            self.jobup_config = {
                'base_search_url': 'https://www.jobup.ch/fr/emplois/',
                'job_selector': 'div.d_flex.bg-c_white.bdr_r16, div.d_flex.bg-c_brand\\.01.bdr_r16',
                'title_selector': 'h2, h3, .job-title, [data-testid="job-title"]',
                'company_selector': '.company-name, [data-testid="company-name"]',
                'location_selector': '.location, [data-testid="location"]',
                'date_selector': '.date, .published, [data-testid="date"]',
                'link_selector': 'a[href*="/emploi/"]',
                'pagination_selector': '.pagination a, .page-link',
                'next_page_selector': '.pagination .next, .page-next'
            }
        
    def setup_selenium(self):
        """Configure et initialise Selenium WebDriver"""
        if not self.use_selenium:
            return
            
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            logger.info("Selenium WebDriver initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Selenium: {e}")
            self.use_selenium = False

    def close_selenium(self):
        """Ferme le driver Selenium"""
        if self.driver:
            self.driver.quit()
            self.driver = None

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

    def extract_job_offer(self, job_element, page_url: str) -> Dict:
        """Extrait les données d'une offre d'emploi spécifique"""
        job_data = {
            'url': '',
            'title': '',
            'company': '',
            'location': '',
            'date_published': '',
            'contract_type': '',
            'activity_rate': '',
            'description': '',
            'full_description': '',
            'requirements': '',
            'benefits': '',
            'salary': '',
            'mission': '',
            'profile': '',
            'company_info': '',
            'scraping_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_page': page_url
        }
        
        try:
            text_content = job_element.get_text()
            
            # Pour jobup.ch, parser le texte structuré
            if self.is_jobup:
                # Le texte est collé, on doit le parser différemment
                text = text_content.strip()
                
                # Extraire la date
                date_match = re.search(r'Il y a \d+ jours?', text)
                if date_match:
                    job_data['date_published'] = date_match.group(0)
                
                # Extraire le titre (entre la date et "Lieu de travail")
                title_match = re.search(r'Il y a \d+ jours?(.*?)Lieu de travail:', text, re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()
                    job_data['title'] = title
                
                # Extraire la localisation
                location_match = re.search(r'Lieu de travail:([^T]+)', text)
                if location_match:
                    job_data['location'] = location_match.group(1).strip()
                
                # Extraire le taux d'activité
                rate_match = re.search(r'Taux d\'activité:([^T]+)', text)
                if rate_match:
                    job_data['activity_rate'] = rate_match.group(1).strip()
                
                # Extraire le type de contrat
                contract_match = re.search(r'Type de contrat:([^G]+)', text)
                if contract_match:
                    job_data['contract_type'] = contract_match.group(1).strip()
                
                # Extraire l'entreprise (après "Type de contrat:")
                company_match = re.search(r'Type de contrat:[^G]+([A-Z][^C]+)', text)
                if company_match:
                    company = company_match.group(1).strip()
                    # Nettoyer le nom de l'entreprise
                    company = re.sub(r'Candidature.*$', '', company).strip()
                    job_data['company'] = company
                
                # Extraire le salaire
                salary_match = re.search(r'CHF\s*[\d\s\-]+', text)
                if salary_match:
                    job_data['salary'] = salary_match.group(0).strip()
                
                # Chercher un lien vers l'offre dans les parents
                link_elem = None
                parent = job_element.parent
                while parent and parent.name != 'body':
                    link_elem = parent.find('a', href=True)
                    if link_elem:
                        href = link_elem['href']
                        # Vérifier que c'est un lien vers une offre détaillée
                        if '/emplois/detail/' in href:
                            job_data['url'] = self.normalize_url(href)
                            break
                    parent = parent.parent
                
            else:
                # Mode standard pour autres sites
                # Titre de l'offre
                title_elem = job_element.find('h2') or job_element.find('h3') or job_element.find('.job-title')
                if title_elem:
                    job_data['title'] = title_elem.get_text(strip=True)
                
                # Lien vers l'offre
                link_elem = job_element.find('a', href=True)
                if link_elem:
                    job_data['url'] = self.normalize_url(link_elem['href'])
                
                # Entreprise
                company_elem = job_element.find(class_=re.compile(r'company|employer', re.I))
                if company_elem:
                    job_data['company'] = company_elem.get_text(strip=True)
                
                # Localisation
                location_elem = job_element.find(class_=re.compile(r'location|place|lieu', re.I))
                if location_elem:
                    job_data['location'] = location_elem.get_text(strip=True)
                
                # Date de publication
                date_elem = job_element.find(class_=re.compile(r'date|published|time', re.I))
                if date_elem:
                    job_data['date_published'] = date_elem.get_text(strip=True)
                
                # Type de contrat et taux d'activité
                if 'Durée indéterminée' in text_content:
                    job_data['contract_type'] = 'Durée indéterminée'
                elif 'Temporaire' in text_content:
                    job_data['contract_type'] = 'Temporaire'
                elif 'CDI' in text_content:
                    job_data['contract_type'] = 'CDI'
                elif 'CDD' in text_content:
                    job_data['contract_type'] = 'CDD'
                
                # Taux d'activité
                rate_match = re.search(r'(\d+%)', text_content)
                if rate_match:
                    job_data['activity_rate'] = rate_match.group(1)
                
                # Salaire
                salary_match = re.search(r'CHF\s*(\d+[\s\d]*)\s*[/-]?\s*(\d+[\s\d]*)?', text_content)
                if salary_match:
                    job_data['salary'] = salary_match.group(0)
            
            # Description (texte complet de l'élément)
            job_data['description'] = text_content.strip()
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction d'une offre: {e}")
        
        return job_data

    def scrape_job_detail(self, job_url: str) -> Dict:
        """Scrape le contenu détaillé d'une offre d'emploi"""
        detail_data = {
            'full_description': '',
            'mission': '',
            'profile': '',
            'benefits': '',
            'company_info': '',
            'requirements': ''
        }
        
        try:
            if self.use_selenium and self.driver:
                self.driver.get(job_url)
                time.sleep(3)  # Attendre le chargement
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            else:
                response = requests.get(job_url, headers=self.headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraire le contenu principal
            main_content = soup.find('main') or soup.find('div', class_=lambda x: x and any(word in str(x).lower() for word in ['content', 'main', 'job', 'offer']))
            
            if main_content:
                full_text = main_content.get_text(separator='\n', strip=True)
                detail_data['full_description'] = full_text
                
                # Essayer d'extraire des sections spécifiques avec des patterns plus précis
                lines = full_text.split('\n')
                
                # Mission/Responsabilités
                mission_keywords = ['your responsibilities', 'mission', 'aufgabe', 'tâche', 'responsabilité', 'responsibilities']
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in mission_keywords):
                        section_content = []
                        for j in range(i, min(i + 15, len(lines))):
                            if lines[j].strip() and not any(stop_word in lines[j].lower() for stop_word in ['your profile', 'profil', 'qualification', 'exigence']):
                                section_content.append(lines[j].strip())
                            elif lines[j].strip() == '' and section_content:
                                break
                        if section_content:
                            detail_data['mission'] = '\n'.join(section_content)
                            break
                
                # Profil/Qualifications
                profile_keywords = ['your profile', 'profil', 'qualification', 'exigence', 'requis', 'requirements', 'qualifications']
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in profile_keywords):
                        section_content = []
                        for j in range(i, min(i + 15, len(lines))):
                            if lines[j].strip() and not any(stop_word in lines[j].lower() for stop_word in ['we offer', 'wir bieten', 'nous offrons', 'avantage']):
                                section_content.append(lines[j].strip())
                            elif lines[j].strip() == '' and section_content:
                                break
                        if section_content:
                            detail_data['profile'] = '\n'.join(section_content)
                            break
                
                # Avantages/Offres
                benefits_keywords = ['we offer', 'wir bieten', 'nous offrons', 'avantage', 'bénéfice', 'benefits', 'perks']
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in benefits_keywords):
                        section_content = []
                        for j in range(i, min(i + 15, len(lines))):
                            if lines[j].strip() and not any(stop_word in lines[j].lower() for stop_word in ['contact', 'à propos', 'about', 'company']):
                                section_content.append(lines[j].strip())
                            elif lines[j].strip() == '' and section_content:
                                break
                        if section_content:
                            detail_data['benefits'] = '\n'.join(section_content)
                            break
                
                # Informations entreprise
                company_keywords = ['à propos', 'about', 'company', 'entreprise', 'über uns', 'bobst', 'btg']
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in company_keywords):
                        section_content = []
                        for j in range(i, min(i + 10, len(lines))):
                            if lines[j].strip():
                                section_content.append(lines[j].strip())
                            elif lines[j].strip() == '' and section_content:
                                break
                        if section_content:
                            detail_data['company_info'] = '\n'.join(section_content)
                            break
            
        except Exception as e:
            logger.warning(f"Erreur lors du scraping détaillé de {job_url}: {e}")
        
        return detail_data

    def scrape_jobup_page(self, url: str) -> List[Dict]:
        """Scrape une page de jobup.ch pour extraire les offres d'emploi"""
        offers = []
        
        try:
            if self.use_selenium and self.driver:
                self.driver.get(url)
                time.sleep(3)  # Attendre le chargement
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            else:
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
            
            # Rechercher les offres d'emploi avec plusieurs sélecteurs possibles
            if self.is_jobup:
                # Sélecteurs spécifiques pour jobup.ch
                job_selectors = [
                    'div[class*="d_flex"][class*="bdr_r16"]'
                ]
            else:
                # Sélecteurs génériques
                job_selectors = [
                    'div[data-testid="job-item"]',
                    '.job-item',
                    '.job-card',
                    '.job-listing',
                    'article',
                    '.search-result-item'
                ]
            
            job_elements = []
            for selector in job_selectors:
                elements = soup.select(selector)
                if elements:
                    job_elements = elements
                    logger.info(f"Trouvé {len(elements)} offres avec le sélecteur: {selector}")
                    break
            
            if not job_elements and self.is_jobup:
                # Fallback spécifique pour jobup.ch: chercher les divs avec les bonnes classes
                job_elements = soup.find_all('div', class_=lambda x: x and 'd_flex' in x and 'bdr_r16' in x and 'p_s16' in x)
                logger.info(f"Fallback jobup.ch: trouvé {len(job_elements)} éléments avec classes spécifiques")
            
            if not job_elements:
                # Fallback général: chercher tous les éléments contenant des mots-clés d'emploi
                job_elements = soup.find_all('div', string=re.compile(r'(Developer|Développeur|Ingénieur|Engineer)', re.I))
                logger.info(f"Fallback général: trouvé {len(job_elements)} éléments avec mots-clés")
            
            for job_element in job_elements:
                job_data = self.extract_job_offer(job_element, url)
                if job_data['title']:  # Seulement si on a au moins un titre
                    # Si on a une URL vers l'offre détaillée, scraper le contenu complet
                    if job_data['url'] and self.is_jobup:
                        logger.info(f"Scraping détaillé de: {job_data['url']}")
                        detail_data = self.scrape_job_detail(job_data['url'])
                        job_data.update(detail_data)
                        time.sleep(1)  # Délai entre les requêtes détaillées
                    
                    offers.append(job_data)
            
            logger.info(f"Extrait {len(offers)} offres de la page {url}")
            
        except Exception as e:
            logger.error(f"Erreur lors du scraping de la page {url}: {e}")
        
        return offers

    def get_pagination_links(self, soup: BeautifulSoup) -> List[str]:
        """Récupère les liens de pagination"""
        pagination_links = []
        
        try:
            # Chercher les liens de pagination
            pagination_selectors = [
                '.pagination a',
                '.page-link',
                '.pager a',
                'a[href*="page="]',
                'a[href*="p="]'
            ]
            
            for selector in pagination_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href and 'page' in href.lower():
                        full_url = self.normalize_url(href)
                        if full_url not in pagination_links:
                            pagination_links.append(full_url)
            
            # Extraire les numéros de page des liens
            page_numbers = []
            for link in pagination_links:
                page_match = re.search(r'page[=_](\d+)', link)
                if page_match:
                    page_numbers.append(int(page_match.group(1)))
            
            # Générer les URLs de pagination si nécessaire
            if page_numbers:
                max_page = max(page_numbers)
                base_url = self.base_url.split('?')[0]
                for page_num in range(2, min(max_page + 1, self.max_pages or max_page + 1)):
                    pag_url = f"{base_url}?term=developer&page={page_num}"
                    if pag_url not in pagination_links:
                        pagination_links.append(pag_url)
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction des liens de pagination: {e}")
        
        return pagination_links

    def scrape_all_jobup_offers(self) -> List[Dict]:
        """Scrape toutes les offres d'emploi de jobup.ch avec pagination"""
        all_offers = []
        
        if not self.is_jobup:
            logger.warning("Cette méthode est spécifique à jobup.ch")
            return all_offers
        
        try:
            # Initialiser Selenium si nécessaire
            if self.use_selenium:
                self.setup_selenium()
            
            # Scraper la première page
            logger.info(f"Scraping de la page principale: {self.base_url}")
            first_page_offers = self.scrape_jobup_page(self.base_url)
            all_offers.extend(first_page_offers)
            
            # Obtenir les liens de pagination
            if self.use_selenium and self.driver:
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            else:
                response = requests.get(self.base_url, headers=self.headers, timeout=15)
                soup = BeautifulSoup(response.text, 'html.parser')
            
            pagination_links = self.get_pagination_links(soup)
            logger.info(f"Trouvé {len(pagination_links)} liens de pagination")
            
            # Scraper les pages suivantes
            for i, page_url in enumerate(pagination_links[:self.max_pages or len(pagination_links)]):
                logger.info(f"Scraping page {i+2}: {page_url}")
                page_offers = self.scrape_jobup_page(page_url)
                all_offers.extend(page_offers)
                
                # Délai entre les pages
                time.sleep(2)
                
                if self.max_pages and len(pagination_links) >= self.max_pages:
                    break
            
            logger.info(f"Total d'offres scrapées: {len(all_offers)}")
            
        except Exception as e:
            logger.error(f"Erreur lors du scraping de toutes les offres: {e}")
        finally:
            if self.use_selenium:
                self.close_selenium()
        
        return all_offers

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

    def save_job_offers(self, offers: List[Dict], filename: str = None) -> str:
        """Sauvegarde les offres d'emploi dans un fichier JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"job_offers_{timestamp}.json"
        
        # Créer le dossier results s'il n'existe pas
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(offers, f, ensure_ascii=False, indent=2)
            logger.info(f"Offres d'emploi sauvegardées dans: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return ""

    def display_job_offers_summary(self, offers: List[Dict]):
        """Affiche un résumé des offres d'emploi scrapées"""
        if not offers:
            print("\n❌ Aucune offre d'emploi trouvée")
            return
        
        print("\n" + "="*80)
        print(f"RÉSUMÉ DES OFFRES D'EMPLOI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        print(f"\n📊 STATISTIQUES")
        print("-"*40)
        print(f"Nombre total d'offres: {len(offers)}")
        
        # Compter par entreprise
        companies = {}
        locations = {}
        contract_types = {}
        
        for offer in offers:
            company = offer.get('company', 'Non spécifié')
            location = offer.get('location', 'Non spécifié')
            contract = offer.get('contract_type', 'Non spécifié')
            
            companies[company] = companies.get(company, 0) + 1
            locations[location] = locations.get(location, 0) + 1
            contract_types[contract] = contract_types.get(contract, 0) + 1
        
        print(f"\n🏢 TOP 10 ENTREPRISES")
        print("-"*40)
        for company, count in sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{company}: {count} offres")
        
        print(f"\n📍 TOP 10 LOCALISATIONS")
        print("-"*40)
        for location, count in sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{location}: {count} offres")
        
        print(f"\n📋 TYPES DE CONTRAT")
        print("-"*40)
        for contract, count in sorted(contract_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{contract}: {count} offres")
        
        print(f"\n💼 ÉCHANTILLON D'OFFRES")
        print("-"*40)
        for i, offer in enumerate(offers[:5], 1):
            print(f"\n{i}. {offer.get('title', 'N/A')}")
            print(f"   Entreprise: {offer.get('company', 'N/A')}")
            print(f"   Localisation: {offer.get('location', 'N/A')}")
            print(f"   Type: {offer.get('contract_type', 'N/A')}")
            print(f"   Taux: {offer.get('activity_rate', 'N/A')}")
            if offer.get('salary'):
                print(f"   Salaire: {offer.get('salary')}")

    def process_website(self):
        """Processus complet de scraping et de création du vector store"""
        if self.is_jobup:
            # Mode spécialisé pour jobup.ch
            logger.info("Mode jobup.ch détecté - scraping des offres d'emploi")
            offers = self.scrape_all_jobup_offers()
            
            if offers:
                self.job_offers = offers
                self.display_job_offers_summary(offers)
                
                # Sauvegarder les offres
                filename = self.save_job_offers(offers)
                
                # Créer des chunks pour la recherche sémantique
                all_text = " ".join([offer.get('description', '') + " " + offer.get('title', '') for offer in offers])
                chunks = self.create_chunks(all_text)
                
                if chunks:
                    vector_store = self.create_vector_store(chunks)
                    return vector_store
                else:
                    logger.warning("Impossible de créer des chunks pour la recherche sémantique")
                    return None
            else:
                logger.error("Aucune offre d'emploi trouvée")
                return None
        else:
            # Mode standard pour autres sites
            self.scrape_page(self.base_url)
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
    """Fonction principale avec exemples d'utilisation"""
    print("🚀 Web Scraper Agent - Version Améliorée")
    print("="*50)
    
    # Exemple 1: Scraping de jobup.ch
    print("\n📋 EXEMPLE 1: Scraping des offres d'emploi jobup.ch")
    print("-"*50)
    
    try:
        # Configuration pour jobup.ch
        jobup_url = "https://www.jobup.ch/fr/emplois/?term=developer"
        
        # Agent avec Selenium pour les sites dynamiques
        agent = WebScraperAgent(
            url=jobup_url,
            vector_store_type="faiss",
            use_selenium=True,  # Activer Selenium pour jobup.ch
            headless=True,      # Mode headless
            max_pages=5         # Limiter à 5 pages pour l'exemple
        )
        
        print(f"Scraping de: {jobup_url}")
        vector_store = agent.process_website()
        
        if vector_store:
            # Recherche sémantique sur les offres
            print("\n🔍 RECHERCHE SÉMANTIQUE SUR LES OFFRES")
            print("-"*40)
            queries = [
                "Développeur Python",
                "Ingénieur logiciel",
                "Travail à distance",
                "Salaire élevé"
            ]
            
            for query in queries:
                print(f"\nQuestion: {query}")
                docs = vector_store.similarity_search(query, k=3)
                for i, doc in enumerate(docs, 1):
                    print(f"  {i}. {doc.page_content[:150]}...")
        
    except Exception as e:
        logger.error(f"Erreur lors du scraping jobup.ch: {e}")
    
    # Exemple 2: Scraping standard d'un autre site
    print("\n\n🌐 EXEMPLE 2: Scraping standard")
    print("-"*50)
    
    try:
        # Exemple avec un site standard
        standard_url = "https://www.exemple.com"
        agent_standard = WebScraperAgent(
            url=standard_url,
            vector_store_type="faiss",
            max_depth=2
        )
        
        print(f"Scraping de: {standard_url}")
        # vector_store = agent_standard.process_website()
        print("Mode standard disponible pour d'autres sites")
        
    except Exception as e:
        logger.error(f"Erreur lors du scraping standard: {e}")

def scrape_jobup_developer_offers(max_pages: int = None, use_selenium: bool = True):
    """Fonction spécialisée pour scraper les offres de développeur sur jobup.ch"""
    url = "https://www.jobup.ch/fr/emplois/?term=developer"
    
    agent = WebScraperAgent(
        url=url,
        vector_store_type="faiss",
        use_selenium=use_selenium,
        headless=True,
        max_pages=max_pages
    )
    
    print(f"🔍 Scraping des offres de développeur sur jobup.ch")
    print(f"📄 Pages maximum: {max_pages or 'Toutes'}")
    print(f"🤖 Selenium: {'Activé' if use_selenium else 'Désactivé'}")
    
    vector_store = agent.process_website()
    return agent.job_offers, vector_store

if __name__ == "__main__":
    # Exécuter les exemples
    main()
    
    # Exemple d'utilisation directe
    print("\n" + "="*80)
    print("💡 UTILISATION DIRECTE")
    print("="*80)
    print("Pour scraper jobup.ch directement:")
    print("offers, vector_store = scrape_jobup_developer_offers(max_pages=10)")
    print("")
    print("Pour un scraping complet:")
    print("offers, vector_store = scrape_jobup_developer_offers()") 
