"""
Quick test: Verify your data can be loaded
"""
from pathlib import Path
import sys

# Add parent directory to path so we can import financial_kg
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("TESTING YOUR DATA ACCESS")
print("=" * 80)

# Test 1: NIFTY-50 Data
print("\n1. Testing NIFTY-50 Stock Data...")
try:
    import pandas as pd
    from financial_kg.data_loaders.stock_loader import StockDataLoader
    
    loader = StockDataLoader(
        data_dir=r"g:\projects\NLP\D\D\kaggle\NIFTY-50 Stock Market Data (2000 - 2021)"
    )
    
    # Test loading RELIANCE
    df = loader.load_stock_csv('RELIANCE')
    print(f"   ✓ Loaded RELIANCE: {len(df)} rows")
    print(f"   ✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   ✓ Latest close: ₹{df.iloc[-1]['Close']}")
    
    # Test loading all
    all_stocks = loader.load_all_nifty50()
    print(f"   ✓ Found {len(all_stocks)} stock CSV files")
    
    print("   ✅ NIFTY-50 data: WORKING!")
    
except Exception as e:
    print(f"   ✗ NIFTY-50 test failed: {e}")

# Test 2: IN-FINews Data
print("\n2. Testing IN-FINews Data...")
try:
    import pandas as pd
    
    df = pd.read_csv(
        r"g:\projects\NLP\D\D\zenodo\IN-FINews Dataset.csv",
        nrows=5
    )
    print(f"   ✓ Loaded {len(df)} articles")
    print(f"   ✓ Columns: {df.columns.tolist()}")
    print(f"   ✓ Sample title: {df.iloc[0]['Title'][:60]}...")
    
    print("   ✅ IN-FINews data: WORKING!")
    
except Exception as e:
    print(f"   ✗ IN-FINews test failed: {e}")

# Test 3: Gemini API (if configured)
print("\n3. Testing Gemini API...")
try:
    from financial_kg.utils.gemini_client import get_gemini_client
    
    client = get_gemini_client()
    print("   ✓ Gemini client initialized")
    print("   ⚠️  Skipping API test (to save costs)")
    print("   ✅ Gemini: CONFIGURED!")
    
except Exception as e:
    print(f"   ✗ Gemini test failed: {e}")
    print("   ℹ️  Make sure GEMINI_API_KEY is set in config/.env")

# Test 4: Neo4j (if configured)
print("\n4. Testing Neo4j Connection...")
try:
    from financial_kg.storage import Neo4jStorage
    
    storage = Neo4jStorage()
    stats = storage.get_graph_stats()
    print(f"   ✓ Connected to Neo4j")
    print(f"   ✓ Current nodes: {stats['total_nodes']}")
    print(f"   ✓ Current relationships: {stats['total_relationships']}")
    storage.close()
    
    print("   ✅ Neo4j: CONNECTED!")
    
except Exception as e:
    print(f"   ✗ Neo4j test failed: {e}")
    print("   ℹ️  Make sure Neo4j credentials are in config/.env")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("\nIf all tests passed:")
print("  ✓ Your data is ready to process!")
print("  ✓ Run: python scripts\\process_my_data.py")
print("\nIf any test failed:")
print("  • Check file paths in the test")
print("  • Check config/.env for API keys")
print("  • See DATA_COMPATIBILITY_ANALYSIS.md for details")
print("=" * 80)
