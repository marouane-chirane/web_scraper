#!/usr/bin/env python3
"""
Script pour tester le scraper avec différents modèles d'embedding
"""

import os
import time
from web_scraper_agent import WebScraperAgent

def test_scraper_with_embedding_model(model_key: str, test_url: str = "https://example.com"):
    """Teste le scraper avec un modèle d'embedding spécifique"""
    print(f"\n🔍 Test avec le modèle: {model_key}")
    print("-" * 50)
    
    # Définir le modèle via variable d'environnement
    os.environ["EMBEDDING_MODEL"] = model_key
    
    try:
        start_time = time.time()
        
        # Créer l'agent
        agent = WebScraperAgent(
            url=test_url,
            use_selenium=False,  # Mode simple pour le test
            max_pages=1
        )
        
        load_time = time.time() - start_time
        print(f"⏱️  Temps de chargement du modèle: {load_time:.2f}s")
        print(f"📊 Modèle utilisé: {agent.embeddings.model_name}")
        
        # Test d'embedding simple
        test_texts = [
            "Développeur Python",
            "Python developer",
            "Ingénieur logiciel",
            "Software engineer"
        ]
        
        start_embed_time = time.time()
        embeddings = agent.embeddings.embed_documents(test_texts)
        embed_time = time.time() - start_embed_time
        
        print(f"⏱️  Temps d'embedding (4 textes): {embed_time:.2f}s")
        print(f"📈 Dimensions des embeddings: {len(embeddings[0])}")
        
        # Test de similarité
        if len(embeddings) >= 2:
            import numpy as np
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            print(f"🔗 Similarité (français-anglais): {similarity:.4f}")
        
        return {
            'model_key': model_key,
            'model_name': agent.embeddings.model_name,
            'load_time': load_time,
            'embed_time': embed_time,
            'dimensions': len(embeddings[0]),
            'similarity': similarity if len(embeddings) >= 2 else None
        }
        
    except Exception as e:
        print(f"❌ Erreur avec {model_key}: {e}")
        return None

def main():
    print("🚀 Test des Modèles d'Embedding avec le Scraper")
    print("=" * 60)
    
    # Modèles à tester
    models_to_test = [
        "mini-lm",           # Modèle actuel (rapide)
        "all-mpnet-base-v2", # Recommandé
        "multi-qa-mpnet",    # Spécialisé recherche
        "multilingual-e5-large", # Meilleur multilingue
    ]
    
    results = []
    
    for model_key in models_to_test:
        result = test_scraper_with_embedding_model(model_key)
        if result:
            results.append(result)
    
    # Afficher le résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    print(f"{'Modèle':<20} {'Dimensions':<10} {'Chargement (s)':<15} {'Embedding (s)':<15}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['model_key']:<20} {result['dimensions']:<10} {result['load_time']:<15.2f} {result['embed_time']:<15.2f}")
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS POUR VOTRE SCRAPER:")
    print("-" * 45)
    
    if results:
        # Meilleur pour la recherche sémantique
        best_search = None
        for result in results:
            if 'multi-qa' in result['model_key']:
                best_search = result
                break
        
        if best_search:
            print(f"🔍 Pour la recherche sémantique: {best_search['model_key']}")
        
        # Meilleur multilingue
        best_multilingual = None
        for result in results:
            if 'multilingual' in result['model_key']:
                best_multilingual = result
                break
        
        if best_multilingual:
            print(f"🌍 Pour le multilingue: {best_multilingual['model_key']}")
        
        # Meilleur compromis
        best_balanced = min(results, key=lambda x: x['load_time'] + x['embed_time'])
        print(f"⚖️  Meilleur compromis: {best_balanced['model_key']}")
        
        # Plus rapide
        fastest = min(results, key=lambda x: x['embed_time'])
        print(f"⚡ Plus rapide: {fastest['model_key']}")

if __name__ == "__main__":
    main()
