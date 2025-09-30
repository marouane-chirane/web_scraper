#!/usr/bin/env python3
"""
Script de test pour le scraping détaillé des offres d'emploi
"""

from web_scraper_agent import WebScraperAgent

def test_detailed_scraping():
    """Test du scraping détaillé"""
    print("🔍 Test du scraping détaillé des offres d'emploi")
    print("="*60)
    
    # Configuration pour jobup.ch avec scraping détaillé
    agent = WebScraperAgent(
        url='https://www.jobup.ch/fr/emplois/?term=developer',
        use_selenium=True,
        headless=True,
        max_pages=1  # Une seule page pour le test
    )
    
    print("🚀 Démarrage du scraping...")
    offers = agent.scrape_all_jobup_offers()
    
    print(f"\n✅ Offres trouvées: {len(offers)}")
    
    if offers:
        print("\n📋 Détails de la première offre:")
        offer = offers[0]
        
        print(f"Titre: {offer.get('title', 'N/A')}")
        print(f"Entreprise: {offer.get('company', 'N/A')}")
        print(f"Localisation: {offer.get('location', 'N/A')}")
        print(f"URL: {offer.get('url', 'N/A')}")
        
        print(f"\n📄 Description complète:")
        full_desc = offer.get('full_description', '')
        if full_desc:
            print(f"Longueur: {len(full_desc)} caractères")
            print(f"Extrait: {full_desc[:500]}...")
        else:
            print("Aucune description détaillée trouvée")
        
        print(f"\n🎯 Mission:")
        mission = offer.get('mission', '')
        if mission:
            print(mission[:300] + "..." if len(mission) > 300 else mission)
        else:
            print("Aucune mission trouvée")
        
        print(f"\n👤 Profil recherché:")
        profile = offer.get('profile', '')
        if profile:
            print(profile[:300] + "..." if len(profile) > 300 else profile)
        else:
            print("Aucun profil trouvé")
        
        print(f"\n💼 Avantages:")
        benefits = offer.get('benefits', '')
        if benefits:
            print(benefits[:300] + "..." if len(benefits) > 300 else benefits)
        else:
            print("Aucun avantage trouvé")
    
    return offers

if __name__ == "__main__":
    test_detailed_scraping()
