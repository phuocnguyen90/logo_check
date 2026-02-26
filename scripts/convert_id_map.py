import json
import sqlite3
import os
from pathlib import Path

def convert_to_sqlite(json_path, db_path):
    if not os.path.exists(json_path):
        print(f"File {json_path} not found")
        return
        
    with open(json_path, 'r') as f:
        id_map = json.load(f)
        
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE id_map (id INTEGER PRIMARY KEY, filename TEXT)')
    
    # Inserting in batches for speed
    data = [(i, filename) for i, filename in enumerate(id_map)]
    cursor.executemany('INSERT INTO id_map VALUES (?, ?)', data)
    
    conn.commit()
    conn.close()
    print(f"Converted {len(id_map)} entries to {db_path}")

if __name__ == "__main__":
    json_path = "models/onnx/best_model/vps_id_map.json"
    db_path = "models/onnx/best_model/vps_id_map.db"
    convert_to_sqlite(json_path, db_path)
