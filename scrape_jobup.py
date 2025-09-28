#!/usr/bin/env python3
"""
Script spécialisé pour scraper les offres d'emploi de jobup.ch
Usage: python scrape_jobup.py [--pages N] [--no-selenium] [--headless]
"""

import argparse
import sys
import os
from web_scraper_agent import WebScraperAgent, scrape_jobup_developer_offers

def main():
    parser = argparse.ArgumentParser(description='Scraper spécialisé pour jobup.ch')
    parser.add_argument('--pages', type=int, default=None, 
                       help='Nombre maximum de pages à scraper (défaut: toutes)')
    parser.add_argument('--no-selenium', action='store_true',
                       help='Désactiver Selenium (utilise seulement requests)')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Mode headless pour Selenium (défaut: True)')
    parser.add_argument('--term', type=str, default='developer',
                       help='Terme de recherche (défaut: developer)')
    parser.add_argument('--output', type=str, default=None,
                       help='Nom du fichier de sortie (défaut: auto-généré)')
    
    args = parser.parse_args()
    
    print("🚀 JobUp.ch Scraper - Version Améliorée")
    print("="*50)
    print(f"🔍 Terme de recherche: {args.term}")
    print(f"📄 Pages maximum: {args.pages or 'Toutes'}")
    print(f"🤖 Selenium: {'Désactivé' if args.no_selenium else 'Activé'}")
    print(f"👻 Mode headless: {args.headless}")
    print("-"*50)
    
    try:
        # Construire l'URL de recherche
        base_url = f"https://www.jobup.ch/fr/emplois/?term={args.term}"
        
        # Créer l'agent
        agent = WebScraperAgent(
            url=base_url,
            vector_store_type="faiss",
            use_selenium=not args.no_selenium,
            headless=args.headless,
            max_pages=args.pages
        )
        
        # Lancer le scraping
        print(f"\n🔄 Démarrage du scraping...")
        vector_store = agent.process_website()
        
        if agent.job_offers:
            print(f"\n✅ Scraping terminé avec succès!")
            print(f"📊 Total d'offres trouvées: {len(agent.job_offers)}")
            
            # Sauvegarder avec nom personnalisé si fourni
            if args.output:
                filename = agent.save_job_offers(agent.job_offers, args.output)
            else:
                filename = agent.save_job_offers(agent.job_offers)
            
            if filename:
                print(f"💾 Données sauvegardées dans: {filename}")
            
            # Recherche sémantique si vector store disponible
            if vector_store:
                print(f"\n🔍 Test de recherche sémantique...")
                test_queries = [
                    "Développeur Python",
                    "Ingénieur logiciel",
                    "Travail à distance",
                    "Salaire élevé"
                ]
                
                for query in test_queries:
                    print(f"\nQuestion: {query}")
                    docs = vector_store.similarity_search(query, k=2)
                    for i, doc in enumerate(docs, 1):
                        print(f"  {i}. {doc.page_content[:100]}...")
        else:
            print("❌ Aucune offre d'emploi trouvée")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Scraping interrompu par l'utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
