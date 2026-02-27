import secrets
import hashlib
import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from logo_similarity.config import paths

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            key_prefix  TEXT    NOT NULL,
            key_hash    TEXT    NOT NULL UNIQUE,
            owner_name  TEXT    NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active   BOOLEAN DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()

def generate_key(owner: str, db_path: Path):
    # Generate 32 bytes of secure entropy (base64-url style for cleanliness)
    raw_key = f"l3d_{secrets.token_urlsafe(32)}"
    prefix = raw_key[:12] # l3d_ + start of entropy
    khash = hash_key(raw_key)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO api_keys (key_prefix, key_hash, owner_name) VALUES (?, ?, ?)",
            (prefix, khash, owner)
        )
        conn.commit()
        print(f"\n🚀 New API Key generated for: {owner}")
        print(f"🔑 Key: {raw_key}")
        print("⚠️  Store this safely! It will NOT be shown again.")
        print(f"📝 Identification Prefix: {prefix}")
    except sqlite3.IntegrityError:
        print("❌ Error: Hash collision or duplicate key. Try again.")
    finally:
        conn.close()

def list_keys(db_path: Path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, key_prefix, owner_name, created_at, is_active FROM api_keys")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No API keys found.")
        return
        
    print(f"\n{'ID':<5} | {'Prefix':<15} | {'Owner':<15} | {'Created':<25} | {'Status'}")
    print("-" * 75)
    for r in rows:
        status = "ACTIVE" if r[4] else "REVOKED"
        print(f"{r[0]:<5} | {r[1]:<15} | {r[2]:<15} | {r[3]:<25} | {status}")

def revoke_key(key_id: int, db_path: Path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,))
    if cursor.rowcount:
        conn.commit()
        print(f"✅ Key {key_id} revoked.")
    else:
        print(f"❌ Key ID {key_id} not found.")
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="L3D API Key Management Utility")
    subparsers = parser.add_subparsers(dest="command")
    
    # Create
    create_p = subparsers.add_parser("create", help="Generate a new API key")
    create_p.add_argument("owner", help="Name/Description of the key owner")
    
    # List
    subparsers.add_parser("list", help="List all API keys")
    
    # Revoke
    revoke_p = subparsers.add_parser("revoke", help="Revoke an API key by ID")
    revoke_p.add_argument("id", type=int, help="Database ID of the key to revoke")
    
    args = parser.parse_args()
    
    db_path = paths.MASTER_METADATA_DB
    if not db_path.exists():
        print(f"❌ Metadata DB not found at {db_path}. Run 01b_eda_and_migrate.py first.")
        return

    init_db(db_path)
    
    if args.command == "create":
        generate_key(args.owner, db_path)
    elif args.command == "list":
        list_keys(db_path)
    elif args.command == "revoke":
        revoke_key(args.id, db_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
