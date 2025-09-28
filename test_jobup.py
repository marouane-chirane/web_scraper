#!/usr/bin/env python3
"""
Script de test pour le scraper jobup.ch
Teste le scraping avec et sans Selenium
"""

import sys
import os
from web_scraper_agent import WebScraperAgent

def test_without_selenium():
    """Test sans Selenium (mode requests uniquement)"""
    print("🧪 TEST 1: Mode requests uniquement")
    print("-"*40)
    
    try:
        agent = WebScraperAgent(
            url="https://www.jobup.ch/fr/emplois/?term=developer",
            use_selenium=False,
            max_pages=2  # Limiter pour le test
        )
        
        offers = agent.scrape_all_jobup_offers()
        print(f"✅ Offres trouvées (sans Selenium): {len(offers)}")
        
        if offers:
            print("\n📋 Échantillon d'offres:")
            for i, offer in enumerate(offers[:3], 1):
                print(f"  {i}. {offer.get('title', 'N/A')}")
                print(f"     Entreprise: {offer.get('company', 'N/A')}")
                print(f"     Localisation: {offer.get('location', 'N/A')}")
        
        return len(offers) > 0
        
    except Exception as e:
        print(f"❌ Erreur (sans Selenium): {e}")
        return False

def test_with_selenium():
    """Test avec Selenium"""
    print("\n🧪 TEST 2: Mode Selenium")
    print("-"*40)
    
    try:
        agent = WebScraperAgent(
            url="https://www.jobup.ch/fr/emplois/?term=developer",
            use_selenium=True,
            headless=True,
            max_pages=2  # Limiter pour le test
        )
        
        offers = agent.scrape_all_jobup_offers()
        print(f"✅ Offres trouvées (avec Selenium): {len(offers)}")
        
        if offers:
            print("\n📋 Échantillon d'offres:")
            for i, offer in enumerate(offers[:3], 1):
                print(f"  {i}. {offer.get('title', 'N/A')}")
                print(f"     Entreprise: {offer.get('company', 'N/A')}")
                print(f"     Localisation: {offer.get('location', 'N/A')}")
        
        return len(offers) > 0
        
    except Exception as e:
        print(f"❌ Erreur (avec Selenium): {e}")
        return False

def main():
    print("🚀 Test du Scraper JobUp.ch")
    print("="*50)
    
    # Test 1: Sans Selenium
    test1_success = test_without_selenium()
    
    # Test 2: Avec Selenium
    test2_success = test_with_selenium()
    
    # Résumé
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*50)
    print(f"Mode requests: {'✅ Réussi' if test1_success else '❌ Échec'}")
    print(f"Mode Selenium: {'✅ Réussi' if test2_success else '❌ Échec'}")
    
    if test1_success or test2_success:
        print("\n🎉 Au moins un mode fonctionne correctement!")
        print("💡 Recommandation: Utilisez le mode qui fonctionne le mieux")
    else:
        print("\n❌ Aucun mode ne fonctionne. Vérifiez:")
        print("   - Connexion internet")
        print("   - Installation des dépendances")
        print("   - Chrome/Chromium installé (pour Selenium)")
    
    return 0 if (test1_success or test2_success) else 1

if __name__ == "__main__":
    sys.exit(main())
