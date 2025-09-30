#!/usr/bin/env python3
"""
Script de test pour comparer les performances des modèles d'embedding
"""

import os
import time
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

def test_embedding_model(model_name: str, test_sentences: list):
    """Teste un modèle d'embedding et retourne les métriques"""
    print(f"\n🔍 Test du modèle: {model_name}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Charger le modèle
        model = SentenceTransformer(model_name)
        
        load_time = time.time() - start_time
        print(f"⏱️  Temps de chargement: {load_time:.2f}s")
        
        # Tester l'embedding
        start_embed_time = time.time()
        embeddings = model.encode(test_sentences)
        embed_time = time.time() - start_embed_time
        
        print(f"⏱️  Temps d'embedding: {embed_time:.2f}s")
        print(f"📊 Dimensions: {embeddings.shape[1]}")
        print(f"📈 Taille mémoire: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        # Tester la similarité
        if len(test_sentences) >= 2:
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            print(f"🔗 Similarité (sentence 1-2): {similarity:.4f}")
        
        return {
            'model': model_name,
            'load_time': load_time,
            'embed_time': embed_time,
            'dimensions': embeddings.shape[1],
            'memory_mb': embeddings.nbytes / 1024 / 1024,
            'similarity': similarity if len(test_sentences) >= 2 else None
        }
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def main():
    print("🚀 Comparaison des Modèles d'Embedding")
    print("=" * 60)
    
    # Phrases de test multilingues
    test_sentences = [
        "Développeur Python avec expérience en machine learning",
        "Python developer with machine learning experience",
        "Ingénieur logiciel spécialisé en intelligence artificielle",
        "Software engineer specialized in artificial intelligence",
        "Recherche d'emploi dans le domaine de la technologie",
        "Job search in the technology field"
    ]
    
    # Modèles à tester
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",  # Modèle actuel
        "sentence-transformers/all-mpnet-base-v2",  # Recommandé
        "intfloat/multilingual-e5-base",  # Multilingue (version base)
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # Spécialisé recherche
    ]
    
    results = []
    
    for model_name in models_to_test:
        result = test_embedding_model(model_name, test_sentences)
        if result:
            results.append(result)
    
    # Afficher le résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ COMPARATIF")
    print("=" * 60)
    
    print(f"{'Modèle':<40} {'Dimensions':<10} {'Temps (s)':<10} {'Mémoire (MB)':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model']:<40} {result['dimensions']:<10} {result['embed_time']:<10.2f} {result['memory_mb']:<12.2f}")
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS:")
    print("-" * 30)
    
    if results:
        # Meilleur compromis performance/ressources
        best_balanced = min(results, key=lambda x: x['embed_time'] + x['memory_mb'])
        print(f"🏆 Meilleur compromis: {best_balanced['model']}")
        
        # Plus rapide
        fastest = min(results, key=lambda x: x['embed_time'])
        print(f"⚡ Plus rapide: {fastest['model']}")
        
        # Plus de dimensions (généralement plus précis)
        most_dims = max(results, key=lambda x: x['dimensions'])
        print(f"🎯 Plus précis: {most_dims['model']}")

if __name__ == "__main__":
    main()
