#!/usr/bin/env python3
"""Test that API keys are stored in .env, not in database."""
import sqlite3
import os
from pathlib import Path

def check_api_key_storage():
    """Verify API keys are NOT in database and ARE in .env file."""
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "data" / "stock_analyzer.db"
    env_path = base_dir / ".env"

    print("=" * 50)
    print("API Key Storage Verification")
    print("=" * 50)

    # Check database
    print("\n1. Checking database for API keys...")
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        api_key_found_in_db = False
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            for row in rows:
                for value in row:
                    if isinstance(value, str) and ('test-key' in value.lower() or 'api_key' in value.lower()):
                        api_key_found_in_db = True
                        print(f"   WARNING: Found API key in {table_name}: {row}")

        if not api_key_found_in_db:
            print("   PASS: No API keys found in database")
        else:
            print("   FAIL: API keys found in database!")

        conn.close()
    else:
        print("   Database does not exist yet")

    # Check .env file
    print("\n2. Checking .env file for API keys...")
    if env_path.exists():
        with open(env_path, 'r') as f:
            content = f.read()

        if 'OPENAI_API_KEY' in content:
            print("   PASS: OPENAI_API_KEY found in .env")
        else:
            print("   FAIL: OPENAI_API_KEY not in .env")

        if 'GEMINI_API_KEY' in content:
            print("   PASS: GEMINI_API_KEY found in .env")
        else:
            print("   FAIL: GEMINI_API_KEY not in .env")

        # Check if test key was saved
        if 'test-key-12345678' in content:
            print("   PASS: Test API key was saved to .env")
        else:
            print("   INFO: Test key not found (may not have been set)")
    else:
        print("   FAIL: .env file does not exist")

    print("\n" + "=" * 50)
    print("Verification Complete")
    print("=" * 50)

if __name__ == "__main__":
    check_api_key_storage()
