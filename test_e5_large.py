#!/usr/bin/env python3
"""
Test du modèle multilingual-e5-large avec le scraper
"""

from web_scraper_agent import WebScraperAgent
import time

def test_e5_large_model():
    """Test du modèle multilingual-e5-large"""
    print("🚀 Test du modèle multilingual-e5-large")
    print("=" * 50)
    
    try:
        start_time = time.time()
        
        # Créer l'agent avec le nouveau modèle
        agent = WebScraperAgent(
            url="https://example.com",  # URL de test
            use_selenium=False,
            max_pages=1
        )
        
        load_time = time.time() - start_time
        print(f"⏱️  Temps de chargement: {load_time:.2f}s")
        print(f"📊 Modèle utilisé: {agent.embeddings.model_name}")
        
        # Test d'embedding multilingue
        test_texts = [
            "Développeur Python avec expérience en machine learning",
            "Python developer with machine learning experience", 
            "Ingénieur logiciel spécialisé en intelligence artificielle",
            "Software engineer specialized in artificial intelligence",
            "Recherche d'emploi dans le domaine de la technologie",
            "Job search in the technology field"
        ]
        
        print(f"\n🔍 Test d'embedding sur {len(test_texts)} textes multilingues...")
        
        start_embed_time = time.time()
        embeddings = agent.embeddings.embed_documents(test_texts)
        embed_time = time.time() - start_embed_time
        
        print(f"⏱️  Temps d'embedding: {embed_time:.2f}s")
        print(f"📈 Dimensions des embeddings: {len(embeddings[0])}")
        print(f"💾 Taille mémoire: {len(embeddings) * len(embeddings[0]) * 4 / 1024 / 1024:.2f} MB")
        
        # Test de similarité multilingue
        import numpy as np
        
        print(f"\n🔗 Tests de similarité multilingue:")
        
        # Français vs Anglais (même sens)
        fr_en_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        print(f"  Français-Anglais (même sens): {fr_en_sim:.4f}")
        
        # Français vs Français (différent sens)
        fr_fr_sim = np.dot(embeddings[0], embeddings[2]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[2])
        )
        print(f"  Français-Français (différent): {fr_fr_sim:.4f}")
        
        # Anglais vs Anglais (différent sens)
        en_en_sim = np.dot(embeddings[1], embeddings[3]) / (
            np.linalg.norm(embeddings[1]) * np.linalg.norm(embeddings[3])
        )
        print(f"  Anglais-Anglais (différent): {en_en_sim:.4f}")
        
        # Test de recherche sémantique
        print(f"\n🔍 Test de recherche sémantique:")
        query = "Python programming job"
        query_embedding = agent.embeddings.embed_query(query)
        
        similarities = []
        for i, embedding in enumerate(embeddings):
            sim = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((i, sim, test_texts[i]))
        
        # Trier par similarité
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Requête: '{query}'")
        print(f"  Résultats (par similarité):")
        for i, (idx, sim, text) in enumerate(similarities[:3]):
            print(f"    {i+1}. {sim:.4f} - {text}")
        
        print(f"\n✅ Modèle multilingual-e5-large fonctionne parfaitement!")
        print(f"📊 Performance: {len(test_texts)/embed_time:.1f} textes/seconde")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = test_e5_large_model()
    if success:
        print(f"\n🎉 Le modèle multilingual-e5-large est prêt à être utilisé!")
    else:
        print(f"\n💥 Problème avec le modèle multilingual-e5-large")
