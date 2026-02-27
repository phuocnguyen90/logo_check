import sqlite3
import os
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def create_rich_metadata_db(vps_json_path, master_db_path, output_db_path):
    """
    Creates a model-specific SQLite database mapping FAISS IDs to rich metadata.
    Uses metadata_v2.db as the source of truth.
    """
    if not os.path.exists(vps_json_path):
        logger.error(f"VPS ID Map not found at {vps_json_path}")
        return
    
    if not os.path.exists(master_db_path):
        logger.error(f"Master metadata DB not found at {master_db_path}")
        return

    # 1. Load FAISS ID Order (JSON list of filenames)
    logger.info(f"Loading FAISS ID map from {vps_json_path}...")
    with open(vps_json_path, 'r') as f:
        vps_filenames = json.load(f)

    # 2. Connect to Master DB
    logger.info(f"Connecting to master DB: {master_db_path}...")
    master_conn = sqlite3.connect(master_db_path)
    master_cursor = master_conn.cursor()

    # 3. Create Output DB
    if os.path.exists(output_db_path):
        os.remove(output_db_path)
    
    out_conn = sqlite3.connect(output_db_path)
    out_cursor = out_conn.cursor()
    out_cursor.execute('''
        CREATE TABLE id_map (
            id           INTEGER PRIMARY KEY,
            filename     TEXT NOT NULL,
            tm_text      TEXT,
            vienna_codes TEXT,
            year         INTEGER
        )
    ''')
    
    # 4. Batch query master DB for metadata
    logger.info("Fetching metadata from master DB and building ID map...")
    
    # Fetch all metadata at once to avoid 700k individual queries
    # Optimized: Filename -> (text, year, [codes])
    metadata_lookup = {}
    
    logger.info("  Loading all filenames and basic data...")
    master_cursor.execute("SELECT filename, tm_text, year, id FROM trademarks")
    tm_rows = master_cursor.fetchall()
    tm_id_map = {row[0]: (row[1], row[2], row[3]) for row in tm_rows}

    logger.info("  Loading Vienna code mappings...")
    master_cursor.execute("""
        SELECT tv.trademark_id, vc.code 
        FROM trademark_vienna tv
        JOIN vienna_codes vc ON vc.id = tv.vienna_code_id
    """)
    vienna_rows = master_cursor.fetchall()
    
    vienna_lookup = {}
    for tm_id, code in vienna_rows:
        if tm_id not in vienna_lookup:
            vienna_lookup[tm_id] = []
        vienna_lookup[tm_id].append(code)

    # 5. Preparing data for batch insert
    insert_data = []
    for i, filename in enumerate(tqdm(vps_filenames, desc="Preparing Output DB")):
        # Handle case mismatches if necessary (metadata is unified, but vps_filenames might have variant casing)
        meta = tm_id_map.get(filename)
        if not meta:
            # Try lower case lookup as fallback
            # (In metadata_v2.db, we tried to keep unique filenames)
            # This is just a safeguard
            meta = tm_id_map.get(filename.lower())
            
        if meta:
            text, year, tm_id = meta
            codes = vienna_lookup.get(tm_id, [])
            vienna_str = ",".join(codes)
            
            insert_data.append((
                i,
                filename,
                text,
                vienna_str,
                year
            ))
        else:
            # Minimal entry if missing from master
            insert_data.append((i, filename, None, None, None))
    
    # 6. Batch Insert
    logger.info(f"Inserting {len(insert_data):,} records into {output_db_path}...")
    for i in range(0, len(insert_data), 10000):
        batch = insert_data[i:i+10000]
        out_cursor.executemany('INSERT INTO id_map VALUES (?, ?, ?, ?, ?)', batch)
        out_conn.commit()
    
    master_conn.close()
    out_conn.close()
    logger.success("🚀 Rich Metadata Database Built from Master DB!")

if __name__ == "__main__":
    vps_map = "models/onnx/best_model/vps_id_map.json"
    master_db = "data/metadata_v2.db"
    db_out = "models/onnx/best_model/vps_id_map.db"
    
    create_rich_metadata_db(vps_map, master_db, db_out)
