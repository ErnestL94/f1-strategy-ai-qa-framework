"""
Ingest golden scenarios into vector database.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.vectordb import VectorDatabase


def main():
    print("=" * 60)
    print("INGESTING GOLDEN SCENARIOS INTO VECTOR DATABASE")
    print("=" * 60)

    # Initialize database
    db = VectorDatabase()

    # Clear existing data
    print("\nClearing existing data...")
    db.clear()

    # Ingest golden scenarios
    print("\nIngesting golden scenarios...")
    golden_dir = Path("datasets/golden")
    count = db.ingest_from_directory(golden_dir)

    # Show stats
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    stats = db.stats()
    print(f"Total scenarios: {stats['total_scenarios']}")
    print(f"Tracks: {', '.join(stats['tracks'])}")
    print(f"Embedding dimension: {stats['embedding_dim']}")

    # Test search
    print("\n" + "=" * 60)
    print("TEST SEARCH")
    print("=" * 60)

    test_scenario = {
        "lap": 30,
        "tires": {"compound": "MEDIUM", "age_laps": 30},
        "position": 2,
        "weather": {"condition": "dry"},
    }

    print("\nQuery: Lap 30, MEDIUM tires (30 laps old), P2, dry")
    results = db.search(test_scenario, k=3)

    print(f"\nTop 3 similar scenarios:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['id']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Track: {result['metadata']['track']}")
        print(
            f"   Lap {result['metadata']['lap']}, "
            f"{result['metadata']['tire_compound']} "
            f"({result['metadata']['tire_age']} laps old)"
        )

    print("\n" + "=" * 60)
    print("✓ Vector database ready for RAG!")
    print("=" * 60)


if __name__ == "__main__":
    main()
