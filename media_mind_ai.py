import os

try:
    from dotenv import load_dotenv
    
    env_path = '.env'
    if os.path.exists(env_path):
        for enc in['utf-8', 'cp1251', 'utf-8-sig', 'utf-16']:
            try:
                load_dotenv(dotenv_path=env_path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    else:
        load_dotenv()

except ImportError:
    pass

try:
    from send2trash import send2trash
except ImportError:
    print("⚠️ Библиотека send2trash не найдена. Установите: pip install send2trash")
    # Фолбэк на безвозвратное удаление, если библиотеки нет
    send2trash = os.remove

try:
    import imagehash
except ImportError:
    print("⚠️ Библиотека imagehash не найдена. Установите: pip install ImageHash")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Библиотека scikit-learn не найдена. Установите: pip install scikit-learn")

import argparse
import sys
import asyncio
import gc
import time
import json
import shutil
import hashlib
import urllib.parse
import io
import sqlite3
import datetime
import concurrent.futures
from collections import defaultdict
from pathlib import Path
import subprocess

# Внешние библиотеки
import cv2
import av
import numpy as np
import torch

import torch.nn.functional as F

# --- ПРОБРОС SAGE ATTENTION (РУЧНОЙ MONKEY PATCH) ---
try:
    from sageattention import sageattn
    
    original_sdpa = F.scaled_dot_product_attention

    def sage_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if attn_mask is not None:
            return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        return sageattn(query, key, value, is_causal=is_causal)

    F.scaled_dot_product_attention = sage_wrapper
    # print("✅ SageAttention успешно активирован и подменил SDPA!")

except Exception as e:
    print(f"⚠️ SageAttention недоступен. Используем стандартный PyTorch SDPA.")
    print(f"   (Детали: {e})")

# --- ИНТЕГРАЦИЯ INSIGHTFACE ---
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace недоступен. Для поиска по лицу установите insightface и onnxruntime-gpu.")

from PIL import Image, ImageFile
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoImageProcessor, AutoModelForImageClassification, SiglipForImageClassification
from nicegui import app, ui, run
from fastapi.responses import FileResponse, Response

# Локальная модель
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

ImageFile.LOAD_TRUNCATED_IMAGES = True

current_dir = os.path.dirname(os.path.abspath(__file__))
# Перенаправляем загрузки моделей HF и PyTorch в папку "models"
os.environ["HF_HOME"] = os.path.join(current_dir, "models")
os.environ["TORCH_HOME"] = os.path.join(current_dir, "models")
CONFIG_FILE = os.path.join(current_dir, 'config.json')
THUMB_CACHE_DIR = os.path.join(current_dir, ".thumbs")
os.makedirs(THUMB_CACHE_DIR, exist_ok=True)

SUPPORTED_IMAGES = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
SUPPORTED_VIDEOS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
SUPPORTED_TEXTS  = ('.txt', '.md', '.json', '.csv')
ITEMS_PER_PAGE = 50

# ==========================================
# МЕНЕДЖМЕНТ КОНФИГУРАЦИЙ
# ==========================================
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception: pass
    return {}

def save_config(updates):
    config = load_config()
    config.update(updates)
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Ошибка сохранения конфига: {e}")

# ==========================================
# FastAPI РОУТИНГ ДЛЯ ПЛЕЕРА И МИНИАТЮР
# ==========================================
@app.get('/media/{file_path:path}')
def read_media(file_path: str):
    clean_path = urllib.parse.unquote(file_path)
    return FileResponse(clean_path)

@app.get('/thumb/{file_path:path}')
def read_thumb(file_path: str):
    clean_path = urllib.parse.unquote(file_path)
    path_hash = hashlib.md5(clean_path.encode('utf-8')).hexdigest()
    thumb_path = os.path.join(THUMB_CACHE_DIR, f"{path_hash}.jpg")

    if os.path.exists(thumb_path):
        return FileResponse(thumb_path)

    ext = os.path.splitext(clean_path)[1].lower()
    try:
        if ext in SUPPORTED_IMAGES:
            with Image.open(clean_path) as img:
                img.thumbnail((300, 300))
                img.convert('RGB').save(thumb_path, format="JPEG", quality=80)
        elif ext in SUPPORTED_VIDEOS:
            with av.open(clean_path) as container:
                for frame in container.decode(video=0):
                    img = frame.to_image()
                    img.thumbnail((300, 300))
                    img.convert('RGB').save(thumb_path, format="JPEG", quality=80)
                    break
        if os.path.exists(thumb_path):
            return FileResponse(thumb_path)
    except: pass
    return Response(status_code=404)

# ==========================================
# 1. БАЗЫ ДАННЫХ И КЭШ
# ==========================================
def get_fast_hash(file_path):
    """Сверхбыстрое хеширование: Размер + MD5(первый 1 МБ) + MD5(последний 1 МБ)"""
    try:
        size = os.path.getsize(file_path)
        if size == 0:
            return "empty_" + hashlib.md5(file_path.encode('utf-8')).hexdigest()
        
        with open(file_path, 'rb') as f:
            first_mb = f.read(1024 * 1024)
            if size > 1024 * 1024:
                f.seek(max(0, size - 1024 * 1024))
                last_mb = f.read(1024 * 1024)
            else:
                last_mb = b""
        
        h1 = hashlib.md5(first_mb).hexdigest()
        h2 = hashlib.md5(last_mb).hexdigest()
        return f"{size}_{h1}_{h2}"
    except Exception:
        # Fallback если файл заблокирован
        return hashlib.md5(file_path.encode('utf-8')).hexdigest()

class DatabaseCache:
    def __init__(self, db_path='image_cache.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL") 
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-262144") 
        self.conn.execute("PRAGMA mmap_size=2147483648") 
        self.conn.execute("PRAGMA temp_store=MEMORY")

        self._migrate_if_needed()
        self._init_tables()

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS files (hash TEXT, path TEXT, size_mb REAL, width INTEGER, height INTEGER, PRIMARY KEY (hash, path))''')
        c.execute('''CREATE TABLE IF NOT EXISTS phash_cache (hash TEXT PRIMARY KEY, phash TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS emb_cache (model TEXT, hash TEXT, features BLOB, PRIMARY KEY (model, hash))''')
        c.execute('''CREATE TABLE IF NOT EXISTS rerank_cache_v2 (model TEXT, query TEXT, hash TEXT, score REAL, PRIMARY KEY (model, query, hash))''')
        c.execute('''CREATE TABLE IF NOT EXISTS aes_cache (model TEXT, hash TEXT, avg_score REAL, max_score REAL, PRIMARY KEY (model, hash))''')
        c.execute('''CREATE TABLE IF NOT EXISTS sim_cache (model TEXT, query TEXT, hash TEXT, score REAL, PRIMARY KEY (model, query, hash))''')
        c.execute('''CREATE TABLE IF NOT EXISTS nsfw_cache (model TEXT, hash TEXT, top_label TEXT, danger_score REAL, details TEXT, PRIMARY KEY (model, hash))''')
        c.execute('''CREATE TABLE IF NOT EXISTS face_cache (hash TEXT, face_idx INTEGER, embedding BLOB, PRIMARY KEY (hash, face_idx))''')
        c.execute('''CREATE TABLE IF NOT EXISTS tags_cache (model TEXT, hash TEXT, tags TEXT, PRIMARY KEY (model, hash))''')
        
        c.execute('CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_nsfw_hash ON nsfw_cache(hash)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_emb_hash ON emb_cache(hash)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_face_hash ON face_cache(hash)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_tags_hash ON tags_cache(hash)')
        self.conn.commit()

    def _migrate_if_needed(self):
        c = self.conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='emb_cache'")
        if c.fetchone():
            c.execute("PRAGMA table_info(emb_cache)")
            cols = [row[1] for row in c.fetchall()]
            if 'path' in cols:
                print("🚀 ВНИМАНИЕ: Найдена старая структура БД! Запускаем авто-миграцию на хеши (ЭТАП 1). Это займет некоторое время...")
                tables_to_migrate =['emb_cache', 'rerank_cache_v2', 'aes_cache', 'sim_cache', 'nsfw_cache', 'face_cache', 'tags_cache']
                for t in tables_to_migrate:
                    try:
                        c.execute(f"ALTER TABLE {t} RENAME TO old_{t}")
                    except Exception: pass
                self.conn.commit()
                self._init_tables()
                self._run_migration()

    def _run_migration(self):
        c = self.conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'old_%'")
        old_tables = [r[0] for r in c.fetchall()]
        
        unique_paths = set()
        for t in old_tables:
            try:
                c.execute(f"SELECT DISTINCT path FROM {t}")
                for row in c.fetchall(): unique_paths.add(row[0])
            except Exception: pass
        
        print(f"📦 Найдено {len(unique_paths)} уникальных файлов в старом кэше. Хешируем...")
        path_to_hash = {}
        insert_files =[]
        
        for i, p in enumerate(unique_paths):
            if i > 0 and i % 500 == 0: print(f"⏳ Хеширование для миграции: {i}/{len(unique_paths)}")
            if os.path.exists(p):
                h = get_fast_hash(p)
                path_to_hash[p] = h
                try: size_mb = os.path.getsize(p) / (1024*1024)
                except: size_mb = 0.0
                insert_files.append((h, p, size_mb, None, None))
        
        if insert_files:
            c.executemany("INSERT OR IGNORE INTO files (hash, path, size_mb, width, height) VALUES (?, ?, ?, ?, ?)", insert_files)
        
        print("🔄 Перенос данных ИИ в новые таблицы...")
        try:
            c.execute("SELECT model, path, features FROM old_emb_cache")
            c.executemany("INSERT OR IGNORE INTO emb_cache (model, hash, features) VALUES (?, ?, ?)", [(r[0], path_to_hash[r[1]], r[2]) for r in c.fetchall() if r[1] in path_to_hash])
            
            c.execute("SELECT model, path, avg_score, max_score FROM old_aes_cache")
            c.executemany("INSERT OR IGNORE INTO aes_cache (model, hash, avg_score, max_score) VALUES (?, ?, ?, ?)", [(r[0], path_to_hash[r[1]], r[2], r[3]) for r in c.fetchall() if r[1] in path_to_hash])
            
            c.execute("SELECT model, path, top_label, danger_score, details FROM old_nsfw_cache")
            c.executemany("INSERT OR IGNORE INTO nsfw_cache (model, hash, top_label, danger_score, details) VALUES (?, ?, ?, ?, ?)", [(r[0], path_to_hash[r[1]], r[2], r[3], r[4]) for r in c.fetchall() if r[1] in path_to_hash])
            
            c.execute("SELECT path, face_idx, embedding FROM old_face_cache")
            c.executemany("INSERT OR IGNORE INTO face_cache (hash, face_idx, embedding) VALUES (?, ?, ?)", [(path_to_hash[r[0]], r[1], r[2]) for r in c.fetchall() if r[0] in path_to_hash])
            
            c.execute("SELECT model, path, tags FROM old_tags_cache")
            c.executemany("INSERT OR IGNORE INTO tags_cache (model, hash, tags) VALUES (?, ?, ?)",[(r[0], path_to_hash[r[1]], r[2]) for r in c.fetchall() if r[1] in path_to_hash])
        except Exception as e: print(f"⚠️ Ошибка миграции некоторых таблиц: {e}")

        for t in old_tables: c.execute(f"DROP TABLE {t}")
            
        self.conn.commit()
        self.conn.execute("VACUUM")
        print("✅ База данных успешно обновлена до версии с HASH ключами!")

    def get_or_create_hashes(self, paths):
        c = self.conn.cursor()
        result = {}
        chunk_size = 900
        for i in range(0, len(paths), chunk_size):
            chunk = paths[i:i+chunk_size]
            ph = ','.join(['?'] * len(chunk))
            c.execute(f"SELECT path, hash FROM files WHERE path IN ({ph})", chunk)
            for row in c.fetchall():
                result[row[0]] = row[1]
                
        insert_data =[]
        for p in paths:
            if p not in result and os.path.exists(p):
                h = get_fast_hash(p)
                result[p] = h
                try: 
                    size_mb = os.path.getsize(p) / (1024 * 1024)
                    w, h_dim = 0, 0
                    ext = os.path.splitext(p)[1].lower()
                    if ext in SUPPORTED_IMAGES:
                        with Image.open(p) as img:
                            w, h_dim = img.size
                    elif ext in SUPPORTED_VIDEOS:
                        with av.open(p) as container:
                            stream = container.streams.video[0]
                            w, h_dim = stream.width, stream.height
                except Exception:
                    size_mb, w, h_dim = 0.0, 0, 0
                    
                insert_data.append((h, p, size_mb, w, h_dim))
                
        if insert_data:
            c.executemany("INSERT OR IGNORE INTO files (hash, path, size_mb, width, height) VALUES (?, ?, ?, ?, ?)", insert_data)
            self.conn.commit()
        return result

    def get_hash_by_path(self, path):
        c = self.conn.cursor()
        c.execute("SELECT hash FROM files WHERE path=?", (path,))
        res = c.fetchone()
        return res[0] if res else None

    # --- Danbooru Tags ---
    def get_tags(self, model_name, file_hash):
        c = self.conn.cursor()
        c.execute("SELECT tags FROM tags_cache WHERE model=? AND hash=?", (model_name, file_hash))
        row = c.fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def save_tags(self, model_name, file_hash, tags_dict):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO tags_cache (model, hash, tags) VALUES (?, ?, ?)", (model_name, file_hash, json.dumps(tags_dict)))
        self.conn.commit()

    def save_tags_batch(self, batch_data):
        if not batch_data: return
        c = self.conn.cursor()
        c.executemany("INSERT OR REPLACE INTO tags_cache (model, hash, tags) VALUES (?, ?, ?)",[(m, h, json.dumps(t)) for m, h, t in batch_data])
        self.conn.commit()

    # --- Face ---
    def get_face_embeddings(self, file_hash):
        c = self.conn.cursor()
        c.execute("SELECT embedding FROM face_cache WHERE hash=?", (file_hash,))
        rows = c.fetchall()
        if not rows: return None
        return [np.frombuffer(r[0], dtype=np.float32) for r in rows if len(r[0]) > 0]

    def save_face_embeddings_batch(self, batch_data):
        if not batch_data: return
        c = self.conn.cursor()
        insert_data =[]
        for h, embs in batch_data:
            if not embs: 
                insert_data.append((h, -1, b''))
            else:
                for i, emb in enumerate(embs): 
                    insert_data.append((h, i, emb.tobytes()))
        c.executemany("INSERT OR REPLACE INTO face_cache (hash, face_idx, embedding) VALUES (?, ?, ?)", insert_data)
        self.conn.commit()

    # --- NSFW ---
    def get_nsfw_score(self, model_name, file_hash):
        c = self.conn.cursor()
        c.execute("SELECT top_label, danger_score, details FROM nsfw_cache WHERE model=? AND hash=?", (model_name, file_hash))
        return c.fetchone()

    def save_nsfw_score(self, model_name, file_hash, top_label, danger_score, details):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO nsfw_cache (model, hash, top_label, danger_score, details) VALUES (?, ?, ?, ?, ?)", (model_name, file_hash, top_label, danger_score, json.dumps(details)))
        self.conn.commit()

    # --- Общие ---
    def get_query_sims(self, model_name, query):
        c = self.conn.cursor()
        c.execute("SELECT hash, score FROM sim_cache WHERE model=? AND query=?", (model_name, query))
        return {row[0]: row[1] for row in c.fetchall()}

    def save_query_sims(self, model_name, query, hashes, scores):
        c = self.conn.cursor()
        data =[(model_name, query, h, s) for h, s in zip(hashes, scores)]
        c.executemany("INSERT OR REPLACE INTO sim_cache (model, query, hash, score) VALUES (?, ?, ?, ?)", data)
        self.conn.commit()

    def get_aesthetic_score(self, model_name, file_hash):
        c = self.conn.cursor()
        c.execute("SELECT avg_score, max_score FROM aes_cache WHERE model=? AND hash=?", (model_name, file_hash))
        return c.fetchone()

    def save_aesthetic_score(self, model_name, file_hash, avg_score, max_score):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO aes_cache (model, hash, avg_score, max_score) VALUES (?, ?, ?, ?)", (model_name, file_hash, avg_score, max_score))
        self.conn.commit()

    def get_image_features(self, model_name, file_hash):
        c = self.conn.cursor()
        c.execute("SELECT features FROM emb_cache WHERE model=? AND hash=?", (model_name, file_hash))
        result = c.fetchone()
        if result is not None: return torch.load(io.BytesIO(result[0]), weights_only=False)
        return None

    def save_image_features(self, model_name, file_hash, features):
        c = self.conn.cursor()
        features_bytes = io.BytesIO()
        torch.save(features, features_bytes)
        c.execute("INSERT OR REPLACE INTO emb_cache (model, hash, features) VALUES (?, ?, ?)", (model_name, file_hash, features_bytes.getvalue()))
        self.conn.commit()

    def get_rerank_score(self, model_name, query, file_hash):
        c = self.conn.cursor()
        c.execute("SELECT score FROM rerank_cache_v2 WHERE model=? AND query=? AND hash=?", (model_name, query, file_hash))
        result = c.fetchone()
        return result[0] if result is not None else None

    def save_rerank_score(self, model_name, query, file_hash, score):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO rerank_cache_v2 (model, query, hash, score) VALUES (?, ?, ?, ?)", (model_name, query, file_hash, score))
        self.conn.commit()

    def get_max_danger_score(self, path):
        h = self.get_hash_by_path(path)
        if not h: return -1.0
        c = self.conn.cursor()
        c.execute("SELECT MAX(danger_score) FROM nsfw_cache WHERE hash=?", (h,))
        res = c.fetchone()
        return res[0] if res and res[0] is not None else -1.0

    def get_all_models(self):
        c = self.conn.cursor()
        models = set()
        for table in['emb_cache', 'rerank_cache_v2', 'aes_cache', 'sim_cache', 'nsfw_cache']:
            try:
                c.execute(f"SELECT DISTINCT model FROM {table}")
                models.update([r[0] for r in c.fetchall() if r[0]])
            except: pass
        try:
            c.execute("SELECT 1 FROM face_cache LIMIT 1")
            if c.fetchone() is not None: models.add("InsightFace (Лица)")
        except: pass
        return list(models)

    def clear_model_cache(self, model_name=None):
        c = self.conn.cursor()
        tables =['emb_cache', 'rerank_cache_v2', 'aes_cache', 'sim_cache', 'nsfw_cache', 'face_cache', 'tags_cache']
        if model_name:
            if model_name == "InsightFace (Лица)": c.execute("DELETE FROM face_cache")
            else:
                for table in tables:
                    if table == 'face_cache': continue 
                    c.execute(f"DELETE FROM {table} WHERE model=?", (model_name,))
        else:
            for table in tables: c.execute(f"DELETE FROM {table}")
            c.execute("DELETE FROM phash_cache") # Теперь полное удаление сносит и pHash тоже
            c.execute("DELETE FROM files")
        self.conn.commit()
        self.conn.execute("VACUUM")

    def clear_specific_cache(self, cache_type):
        """Точечная очистка конкретных таблиц"""
        c = self.conn.cursor()
        if cache_type == 'phash':
            c.execute("DELETE FROM phash_cache")
        elif cache_type == 'tags':
            c.execute("DELETE FROM tags_cache")
        elif cache_type == 'nsfw_aes':
            c.execute("DELETE FROM nsfw_cache")
            c.execute("DELETE FROM aes_cache")
        elif cache_type == 'search_history':
            c.execute("DELETE FROM sim_cache")
            c.execute("DELETE FROM rerank_cache_v2")
        elif cache_type == 'faces':
            c.execute("DELETE FROM face_cache")
        self.conn.commit()

    def get_all_paths(self):
        c = self.conn.cursor()
        c.execute("SELECT DISTINCT path FROM files")
        return[r[0] for r in c.fetchall() if r[0]]

    def remove_paths(self, paths_to_remove):
        if not paths_to_remove: return
        c = self.conn.cursor()
        chunk_size = 900
        for i in range(0, len(paths_to_remove), chunk_size):
            chunk = paths_to_remove[i:i+chunk_size]
            placeholders = ','.join(['?'] * len(chunk))
            c.execute(f"DELETE FROM files WHERE path IN ({placeholders})", chunk)
        self.conn.commit()

    def close(self): self.conn.close()

class FilesCache:
    FILE_NAME = 'dir_cache.json'
    def __init__(self):
        self._data = self._load_cache()
    
    def _load_cache(self):
        if not os.path.isfile(self.FILE_NAME): return {}
        try:
            with open(self.FILE_NAME, 'r', encoding='utf-8') as f: return json.load(f)
        except json.JSONDecodeError: return {}

    def save_cache(self):
        with open(self.FILE_NAME, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=4)

    def list_files(self, directory):
        return self._data.get(directory, None)

class MediaCache:
    def __init__(self):
        self.enabled = False
        self.compress = False
        self.cache = {}

    def clear(self):
        self.cache.clear()
        gc.collect()

    def _compress_img(self, img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _decompress_img(self, bytes_data):
        return Image.open(io.BytesIO(bytes_data)).convert("RGB")

    def _get_bucket_size(self, w, h, max_dim, patch_size=28):
        scale = min(max_dim / w, max_dim / h)
        if scale > 1.0: scale = 1.0
        new_w = max(patch_size, int(round((w * scale) / patch_size) * patch_size))
        new_h = max(patch_size, int(round((h * scale) / patch_size) * patch_size))
        return new_w, new_h

    def get_image(self, path, max_dim):
        cache_key = (path, max_dim)
        if self.enabled and cache_key in self.cache:
            cached = self.cache[cache_key]
            return self._decompress_img(cached) if self.compress else cached
        try:
            image = Image.open(path).convert("RGB")
            new_w, new_h = self._get_bucket_size(image.width, image.height, max_dim)
            resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            if self.enabled:
                self.cache[cache_key] = self._compress_img(resized) if self.compress else resized
            return resized
        except Exception:
            return None

    def get_video_frames(self, path, max_dim, video_frames):
        cache_key = (path, max_dim, video_frames)
        if self.enabled and cache_key in self.cache:
            cached = self.cache[cache_key]
            return[self._decompress_img(b) for b in cached] if self.compress else cached
        try:
            frames =[]
            with av.open(path) as container:
                stream = container.streams.video[0]
                total_frames = stream.frames or 100
                num_extract = max(1, video_frames)
                step = max(1, total_frames // num_extract)
                target_indices = {min(i * step, total_frames - 1) for i in range(num_extract)}
                
                extracted =[]
                for i, frame in enumerate(container.decode(video=0)):
                    if i in target_indices:
                        extracted.append(frame.to_image().convert("RGB"))
                        target_indices.remove(i)
                    if not target_indices: break
                    
            if not extracted: return None
            while len(extracted) < num_extract: extracted.append(extracted[-1])
            new_w, new_h = self._get_bucket_size(extracted[0].width, extracted[0].height, max_dim)
            resized =[img.resize((new_w, new_h), Image.Resampling.BILINEAR) for img in extracted]
            if self.enabled:
                self.cache[cache_key] =[self._compress_img(img) for img in resized] if self.compress else resized
            return resized
        except Exception:
            return None

media_cache = MediaCache()

# ==========================================
# 2. ДВИЖОК ПОИСКА
# ==========================================
class SearchEngine:
    def __init__(self, log_callback, progress_callback):
        self.log = log_callback
        self.progress = progress_callback
        self.files_cache = FilesCache()
        self.db_cache = DatabaseCache()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"} if self.device == "cuda" else {}
        self.embedding_model = None
        self.current_emb_model_state = None
        self.emb_size = 512
        self.rerank_size = 800
        self.video_frames = 4
        self.quant_mode = "None"
        self.cancel_flag = False

    def cancel(self): self.cancel_flag = True

    def _download_model(self, model_name):
        local_dir = os.path.join(current_dir, "models", model_name.replace("/", "_"))
        if not os.path.exists(local_dir) or not os.listdir(local_dir):
            self.log(f"Скачивание модели {model_name}...")
            snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
        return local_dir

    def _unload_embedding_model(self):
        if self.embedding_model is not None:
            self.log(f"Выгрузка модели эмбеддингов из VRAM...")
            del self.embedding_model
            self.embedding_model = None
            self.current_emb_model_state = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _apply_quantization(self, kwargs):
        if self.quant_mode != "None" and self.device == "cuda":
            kwargs["device_map"] = {"": self.device}
            try:
                from transformers import BitsAndBytesConfig
                if self.quant_mode == "8-bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                elif self.quant_mode == "4-bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
            except ImportError:
                self.log("⚠️ ОШИБКА: Для квантования установите пакеты: pip install bitsandbytes accelerate")
        return kwargs

    def _get_embedding_model(self, model_name):
        current_state = f"{model_name}_{self.quant_mode}"
        if self.current_emb_model_state != current_state or self.embedding_model is None:
            self._unload_embedding_model()
            kwargs = dict(self.model_kwargs)
            kwargs = self._apply_quantization(kwargs)
            
            local_model_path = self._download_model(model_name)
            self.log(f"Загрузка модели {model_name} в VRAM...")
            self.embedding_model = SentenceTransformer(local_model_path, device=self.device, model_kwargs=kwargs, trust_remote_code=True)
            self.current_emb_model_state = current_state
        return self.embedding_model

    def _gather_files(self, dir_paths, allowed_exts):
        if isinstance(dir_paths, str):
            dirs =[d.strip() for d in dir_paths.replace('\r', '\n').split('\n') if d.strip()]
        else:
            dirs = dir_paths
            
        all_gathered =[]
        all_supported = SUPPORTED_IMAGES + SUPPORTED_VIDEOS + SUPPORTED_TEXTS
        
        for d_path in dirs:
            if not os.path.exists(d_path): continue
            files_list = self.files_cache.list_files(d_path)
            if files_list is None:
                self.log(f"Индексация: {d_path}...")
                files_list =[]
                for root, _, files in os.walk(d_path):
                    if self.cancel_flag: break
                    for file in files:
                        if file.lower().endswith(all_supported):
                            files_list.append(os.path.join(root, file))
                if not self.cancel_flag:
                    self.files_cache._data[d_path] = files_list
                    self.files_cache.save_cache()
            if files_list:
                all_gathered.extend(files_list)
                
        # Убираем дубли путей (на случай, если папки пересекаются)
        all_gathered = list(set(all_gathered))
        return[f for f in all_gathered if f.lower().endswith(allowed_exts)]

    def _load_and_prep_file(self, file_path, phase='embedding'):
        ext = os.path.splitext(file_path)[1].lower()
        size_val = self.emb_size if phase == 'embedding' else self.rerank_size
        if ext in SUPPORTED_IMAGES:
            img = media_cache.get_image(file_path, size_val)
            if img:
                return img, f"{img.width}x{img.height}", 1
            return None, None, 0
        elif ext in SUPPORTED_VIDEOS:
            frames = media_cache.get_video_frames(file_path, size_val, self.video_frames)
            if frames:
                new_w, new_h = frames[0].width, frames[0].height
                stacked = np.stack([np.array(f) for f in frames])
                return {"video": stacked}, f"{new_w}x{new_h}", self.video_frames
            return None, None, 0
        elif ext in SUPPORTED_TEXTS:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()[:2000]
                    if text.strip(): return text, "text", 1
            except: pass
        return None, None, 0

    def prepare_query(self, raw_query):
        if os.path.isfile(raw_query):
            doc_emb, _, _ = self._load_and_prep_file(raw_query, phase='embedding')
            doc_rerank, _, _ = self._load_and_prep_file(raw_query, phase='rerank')
            return doc_emb, doc_rerank
        return raw_query, raw_query

    def build_cache(self, dir_path, emb_model_name, batch_size, allowed_exts, override_files=None):
        self.cancel_flag = False
        files_list = self._gather_files(dir_path, allowed_exts) if override_files is None else[f for f in override_files if f.lower().endswith(allowed_exts)]
        cache_key = emb_model_name if self.emb_size == 512 else f"{emb_model_name}_{self.emb_size}"
        
        path_to_hash = self.db_cache.get_or_create_hashes(files_list)
        paths_to_compute =[]
        for fp in files_list:
            if self.cancel_flag: break
            h = path_to_hash.get(fp)
            if h and self.db_cache.get_image_features(cache_key, h) is None:
                paths_to_compute.append(fp)
                
        if not paths_to_compute or self.cancel_flag:
            self.log("Кэш эмбеддингов полностью актуален.")
            return

        model = self._get_embedding_model(emb_model_name)
        self.log(f"Кэширование поиска: обработка {len(paths_to_compute)} новых файлов...")
        processed_count, total = 0, len(paths_to_compute)
        preload_chunk = max(64, batch_size * 4) 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 2)) as executor:
            for i in range(0, total, preload_chunk):
                if self.cancel_flag: break
                chunk_paths = paths_to_compute[i:i + preload_chunk]
                futures = {executor.submit(self._load_and_prep_file, p, 'embedding'): p for p in chunk_paths}
                buckets = defaultdict(list)
                for fut in concurrent.futures.as_completed(futures):
                    path = futures[fut]
                    doc, size_key, weight = fut.result()
                    if doc is not None: buckets[size_key].append((path, doc, weight))
                
                for size_key, items in buckets.items():
                    if self.cancel_flag: break
                    c_paths, c_docs, c_weight = [],[], 0
                    
                    for path, doc, weight in items:
                        if c_weight + weight > batch_size and len(c_docs) > 0:
                            try:
                                feats_batch = model.encode(c_docs, batch_size=len(c_docs), convert_to_tensor=True).cpu()
                                for p, feats in zip(c_paths, feats_batch):
                                    h = path_to_hash.get(p)
                                    if h: self.db_cache.save_image_features(cache_key, h, feats)
                            except Exception as e: self.log(f"Ошибка батча эмбеддингов: {e}")
                            
                            processed_count += len(c_paths)
                            self.progress(processed_count / total, f"Кэш эмбеддингов ({processed_count}/{total})...")
                            c_paths, c_docs, c_weight = [],[], 0
                            
                        c_paths.append(path)
                        c_docs.append(doc)
                        c_weight += weight
                        
                    if len(c_docs) > 0 and not self.cancel_flag:
                        try:
                            feats_batch = model.encode(c_docs, batch_size=len(c_docs), convert_to_tensor=True).cpu()
                            for p, feats in zip(c_paths, feats_batch):
                                h = path_to_hash.get(p)
                                if h: self.db_cache.save_image_features(cache_key, h, feats)
                        except Exception as e: self.log(f"Ошибка батча эмбеддингов: {e}")
                            
                        processed_count += len(c_paths)
                        self.progress(processed_count / total, f"Кэш эмбеддингов ({processed_count}/{total})...")

    def phase1_recall(self, dir_path, raw_query, query_input, top_k, emb_model_name, batch_size, allowed_exts):
        self.cancel_flag = False
        files_list = self._gather_files(dir_path, allowed_exts)
        
        results_phase1 =[]
        cache_key = emb_model_name if self.emb_size == 512 else f"{emb_model_name}_{self.emb_size}"
        cached_sims_hashes = self.db_cache.get_query_sims(cache_key, raw_query)
        path_to_hash = self.db_cache.get_or_create_hashes(files_list)
        
        self.log(f"Фильтрация {len(files_list)} файлов через кэш...")
        paths_needing_sims, paths_needing_features = [],[]
        
        for i, file_path in enumerate(files_list):
            if self.cancel_flag: break
            h = path_to_hash.get(file_path)
            if not h: continue
            
            if h in cached_sims_hashes:
                results_phase1.append((cached_sims_hashes[h], file_path))
            else:
                paths_needing_sims.append(file_path)
                
            if i % 500 == 0: 
                prog = 0.1 * (i / max(1, len(files_list)))
                self.progress(prog, f"Чтение кэша ({i}/{len(files_list)})...")
                
        if not paths_needing_sims or self.cancel_flag:
            self.log("⚡ Запрос полностью закэширован! Обход загрузки модели.")
            results_phase1.sort(key=lambda x: x[0], reverse=True)
            self.progress(0.8, "Поиск завершен.") 
            return results_phase1[:top_k]

        model = self._get_embedding_model(emb_model_name)
        self.log("Конвертация запроса в эмбеддинг...")
        query_emb = model.encode(query_input, convert_to_tensor=True).cpu()

        sims_to_save_hashes, sims_to_save_scores = [],[]
        
        for file_path in paths_needing_sims:
            if self.cancel_flag: break
            h = path_to_hash.get(file_path)
            features = self.db_cache.get_image_features(cache_key, h)
            if features is not None:
                sim = float(util.cos_sim(query_emb, features).item())
                results_phase1.append((sim, file_path))
                sims_to_save_hashes.append(h)
                sims_to_save_scores.append(sim)
            else: 
                paths_needing_features.append(file_path)

        if sims_to_save_hashes and not self.cancel_flag:
            self.db_cache.save_query_sims(cache_key, raw_query, sims_to_save_hashes, sims_to_save_scores)

        if not paths_needing_features or self.cancel_flag:
            results_phase1.sort(key=lambda x: x[0], reverse=True)
            self.progress(0.8, "Поиск завершен.") 
            return results_phase1[:top_k]

        self.log(f"ИИ обработка новых файлов: {len(paths_needing_features)} шт...")
        processed_count, total = 0, len(paths_needing_features)
        preload_chunk = max(64, batch_size * 4) 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 2)) as executor:
            for i in range(0, total, preload_chunk):
                if self.cancel_flag: break
                chunk_paths = paths_needing_features[i:i + preload_chunk]
                futures = {executor.submit(self._load_and_prep_file, p, 'embedding'): p for p in chunk_paths}
                buckets = defaultdict(list)
                
                for fut in concurrent.futures.as_completed(futures):
                    path = futures[fut]
                    doc, size_key, weight = fut.result()
                    if doc is not None: 
                        buckets[size_key].append((path, doc, weight))
                    else:
                        h = path_to_hash.get(path)
                        if h: self.db_cache.save_query_sims(cache_key, raw_query, [h], [0.0])
                
                for size_key, items in buckets.items():
                    if self.cancel_flag: break
                    c_paths, c_docs, c_weight = [],[], 0
                    
                    for path, doc, weight in items:
                        if c_weight + weight > batch_size and len(c_docs) > 0:
                            try:
                                feats_batch = model.encode(c_docs, batch_size=len(c_docs), convert_to_tensor=True).cpu()
                                sims_to_save, c_hashes = [],[]
                                for p, feats in zip(c_paths, feats_batch):
                                    h = path_to_hash.get(p)
                                    if h:
                                        self.db_cache.save_image_features(cache_key, h, feats)
                                        sim = float(util.cos_sim(query_emb, feats).item())
                                        sims_to_save.append(sim)
                                        c_hashes.append(h)
                                        results_phase1.append((sim, p))
                                self.db_cache.save_query_sims(cache_key, raw_query, c_hashes, sims_to_save)
                            except Exception as e: self.log(f"Ошибка батча: {e}")
                            
                            processed_count += len(c_paths)
                            self.progress(0.1 + 0.7 * (processed_count / total), f"Инференс ({processed_count}/{total})...")
                            c_paths, c_docs, c_weight = [],[], 0
                            
                        c_paths.append(path)
                        c_docs.append(doc)
                        c_weight += weight
                        
                    if len(c_docs) > 0 and not self.cancel_flag:
                        try:
                            feats_batch = model.encode(c_docs, batch_size=len(c_docs), convert_to_tensor=True).cpu()
                            sims_to_save, c_hashes = [],[]
                            for p, feats in zip(c_paths, feats_batch):
                                h = path_to_hash.get(p)
                                if h:
                                    self.db_cache.save_image_features(cache_key, h, feats)
                                    sim = float(util.cos_sim(query_emb, feats).item())
                                    sims_to_save.append(sim)
                                    c_hashes.append(h)
                                    results_phase1.append((sim, p))
                            self.db_cache.save_query_sims(cache_key, raw_query, c_hashes, sims_to_save)
                        except Exception as e: self.log(f"Ошибка батча (остаток): {e}")
                            
                        processed_count += len(c_paths)
                        self.progress(0.1 + 0.7 * (processed_count / total), f"Инференс ({processed_count}/{total})...")

        results_phase1.sort(key=lambda x: x[0], reverse=True)
        return results_phase1[:top_k]

    def phase2_rerank(self, raw_query, query_input, top_candidates, min_score, rerank_model_name):
        if not top_candidates or self.cancel_flag: return top_candidates
        cache_key = rerank_model_name if self.rerank_size == 800 else f"{rerank_model_name}_{self.rerank_size}"
        
        path_to_hash = self.db_cache.get_or_create_hashes([fp for _, fp in top_candidates])
        final_results =[]
        docs_to_compute, paths_to_compute = [],[]
        
        for i, (score, fp) in enumerate(top_candidates):
            if self.cancel_flag: break
            h = path_to_hash.get(fp)
            cached_score = self.db_cache.get_rerank_score(cache_key, raw_query, h) if h else None
            
            if cached_score is not None:
                if cached_score >= min_score: final_results.append((cached_score, fp))
            else:
                doc, _, _ = self._load_and_prep_file(fp, 'rerank') 
                if doc is not None:
                    docs_to_compute.append(doc)
                    paths_to_compute.append(fp)
                    
        if docs_to_compute and not self.cancel_flag:
            self._unload_embedding_model()
            self.log(f"Reranker: глубокая обработка {len(docs_to_compute)} кандидатов...")
            kwargs = dict(self.model_kwargs)
            kwargs = self._apply_quantization(kwargs)
            local_path = self._download_model(rerank_model_name)
            reranker = CrossEncoder(local_path, device=self.device, model_kwargs=kwargs, trust_remote_code=True)
            
            chunk_size = 4
            processed, len_total = 0, len(docs_to_compute)
            for i in range(0, len_total, chunk_size):
                if self.cancel_flag: break
                c_docs, c_paths = docs_to_compute[i:i+chunk_size], paths_to_compute[i:i+chunk_size]
                rankings = reranker.rank(query_input, c_docs, batch_size=len(c_docs))
                for rank in rankings:
                    s = float(rank['score'])
                    fp = c_paths[rank['corpus_id']]
                    h = path_to_hash.get(fp)
                    if h: self.db_cache.save_rerank_score(cache_key, raw_query, h, s)
                    if s >= min_score: final_results.append((s, fp))
                    
                processed += len(c_docs)
                self.progress(0.8 + 0.2 * (processed / len_total), f"Rerank ({processed}/{len_total})...")
                
            del reranker
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        final_results.sort(key=lambda x: x[0], reverse=True)
        return final_results

# ==========================================
# 3. ДВИЖКИ ЭСТЕТИКИ, NSFW, ЛИЦ И ТЕГОВ
# ==========================================
class AestheticEngine:
    def __init__(self, search_engine):
        self.se = search_engine
        self.db_cache = search_engine.db_cache
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = None
        self.preprocessor = None
        self.batch_size = 16
        self.max_dim = 512
        self.video_frames = 4
        self.quant_mode = "None"
        self.current_model_state = None

    def load_model(self):
        current_state = f"v2_5_{self.quant_mode}"
        if self.model is None or self.current_model_state != current_state:
            self.unload()
            state.add_log(f"Загрузка модели Aesthetic Predictor на {self.device}...")
            # Принудительная скачка модели в локальную папку моделей
            kwargs = {
                "low_cpu_mem_usage": True, 
                "trust_remote_code": True,
                "cache_dir": os.path.join(current_dir, "models"),
                "torch_dtype": self.dtype
            }
            if self.device == "cuda":
                kwargs["attn_implementation"] = "sdpa"
                
            if self.quant_mode != "None" and self.device == "cuda":
                kwargs["device_map"] = {"": self.device}
                try:
                    from transformers import BitsAndBytesConfig
                    # Игнорируем кастомную голову 'layers', так как она грузится отдельно через load_state_dict
                    bnb_kwargs = {"llm_int8_skip_modules": ["layers"]} 
                    
                    if self.quant_mode == "8-bit":
                        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, **bnb_kwargs)
                    elif self.quant_mode == "4-bit":
                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=self.dtype,
                            **bnb_kwargs
                        )
                except ImportError:
                    state.add_log("⚠️ ОШИБКА: Для квантования установите пакеты: pip install bitsandbytes accelerate")
                
            self.model, self.preprocessor = convert_v2_5_from_siglip(**kwargs)
            
            if self.quant_mode == "None":
                self.model = self.model.to(self.dtype).to(self.device)
            else:
                # Перекидываем пропущенную голову в нужный формат вручную
                if hasattr(self.model, "layers"):
                    self.model.layers.to(self.dtype).to(self.device)
                elif hasattr(self.model, "mlp"):
                    self.model.mlp.to(self.dtype).to(self.device)
                
            self.model.eval()
            self.current_model_state = current_state

    def unload(self):
        if self.model is not None:
            state.add_log(f"Выгрузка Aesthetic Predictor из VRAM...")
            del self.model
            del self.preprocessor
            self.model = None
            self.preprocessor = None
            self.current_model_state = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def evaluate_media(self, directory_path, allowed_exts, override_files=None):
        all_files = self.se._gather_files(directory_path, allowed_exts) if override_files is None else[f for f in override_files if f.lower().endswith(allowed_exts)]
        path_to_hash = self.db_cache.get_or_create_hashes(all_files)
        
        image_paths =[p for p in all_files if p.lower().endswith(SUPPORTED_IMAGES)]
        video_paths =[p for p in all_files if p.lower().endswith(SUPPORTED_VIDEOS)]
        
        state.add_log(f"Найдено для оценки: {len(image_paths)} изображений, {len(video_paths)} видео.")
        results =[]
        cache_key_img = "v2_5_siglip"
        cache_key_vid = "v2_5_siglip_vid_" + str(self.video_frames)

        images_to_process, videos_to_process = [],[]
        for p in image_paths:
            h = path_to_hash.get(p)
            if not h: continue
            cached = self.db_cache.get_aesthetic_score(cache_key_img, h)
            if cached is not None: results.append((cached[0], p, cached[1]))
            else: images_to_process.append(p)
                
        for p in video_paths:
            h = path_to_hash.get(p)
            if not h: continue
            cached = self.db_cache.get_aesthetic_score(cache_key_vid, h)
            if cached is not None: results.append((cached[0], p, cached[1]))
            else: videos_to_process.append(p)
                
        if images_to_process or videos_to_process: self.load_model()
        else:
            results.sort(key=lambda x: x[0], reverse=True)
            return results

        # --- ОБРАБОТКА ИЗОБРАЖЕНИЙ ---
        batch_images, batch_paths = [],[]
        for i, img_path in enumerate(images_to_process):
            if not state.is_processing: break
            state.status_text = f"Подготовка фото: {Path(img_path).name} ({i+1}/{len(images_to_process)})"
            h = path_to_hash.get(img_path)
            try:
                image = media_cache.get_image(img_path, self.max_dim)
                if image:
                    batch_images.append(image)
                    batch_paths.append(img_path)
                else:
                    if h: self.db_cache.save_aesthetic_score(cache_key_img, h, 0.0, 0.0)
                    results.append((0.0, img_path, 0.0))
            except Exception as e: 
                if h: self.db_cache.save_aesthetic_score(cache_key_img, h, 0.0, 0.0)
                results.append((0.0, img_path, 0.0))
                
            if len(batch_images) >= self.batch_size or (i == len(images_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(images_to_process))
                try:
                    pixel_values = self.preprocessor(images=batch_images, return_tensors="pt").pixel_values.to(self.dtype).to(self.device)
                    with torch.inference_mode():
                        logits = self.model(pixel_values).logits.flatten().float().cpu().tolist()
                    for score, p in zip(logits, batch_paths):
                        h_val = path_to_hash.get(p)
                        if h_val: self.db_cache.save_aesthetic_score(cache_key_img, h_val, score, score)
                        results.append((score, p, score))
                except Exception as e: state.add_log(f"Ошибка инференса: {e}")
                batch_images, batch_paths =[],[]

        # --- ОБРАБОТКА ВИДЕО ---
        batch_images, batch_frame_counts, batch_paths = [], [],[]
        for i, vid_path in enumerate(videos_to_process):
            if not state.is_processing: break
            state.status_text = f"Подготовка видео: {Path(vid_path).name} ({i+1}/{len(videos_to_process)})"
            h = path_to_hash.get(vid_path)
            try:
                frames = media_cache.get_video_frames(vid_path, self.max_dim, self.video_frames)
                if frames:
                    batch_images.extend(frames)
                    batch_paths.append(vid_path)
                    batch_frame_counts.append(len(frames))
                else:
                    if h: self.db_cache.save_aesthetic_score(cache_key_vid, h, 0.0, 0.0)
                    results.append((0.0, vid_path, 0.0))
            except Exception as e: 
                if h: self.db_cache.save_aesthetic_score(cache_key_vid, h, 0.0, 0.0)
                results.append((0.0, vid_path, 0.0))
                
            if len(batch_images) >= self.batch_size or (i == len(videos_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(videos_to_process))
                try:
                    all_scores =[]
                    for k in range(0, len(batch_images), self.batch_size):
                        chunk = batch_images[k:k+self.batch_size]
                        pixel_values = self.preprocessor(images=chunk, return_tensors="pt").pixel_values.to(self.dtype).to(self.device)
                        with torch.inference_mode():
                            logits = self.model(pixel_values).logits.flatten().float().cpu().tolist()
                        all_scores.extend(logits)
                        
                    idx = 0
                    for path, count in zip(batch_paths, batch_frame_counts):
                        vid_scores = all_scores[idx : idx + count]
                        idx += count
                        if vid_scores:
                            avg_s = sum(vid_scores) / len(vid_scores)
                            max_s = max(vid_scores)
                            h_val = path_to_hash.get(path)
                            if h_val: self.db_cache.save_aesthetic_score(cache_key_vid, h_val, avg_s, max_s)
                            results.append((avg_s, path, max_s))
                except Exception as e: state.add_log(f"Ошибка инференса: {e}")
                batch_images, batch_frame_counts, batch_paths = [], [],[]

        results.sort(key=lambda x: x[0], reverse=True)
        return results

class NsfwEngine:
    def __init__(self, search_engine):
        self.se = search_engine
        self.db_cache = search_engine.db_cache
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self.current_model_name = None
        self.batch_size = 16
        self.max_dim = 512
        self.video_frames = 4
        self.quant_mode = "None"
        self.current_model_state = None

    def load_model(self, model_name):
        current_state = f"{model_name}_{self.quant_mode}"
        if self.model is None or self.current_model_state != current_state:
            self.unload()
            state.add_log(f"Загрузка NSFW модели {model_name} на {self.device}...")
            
            local_dir = os.path.join(current_dir, "models", model_name.replace("/", "_"))
            if not os.path.exists(local_dir) or not os.listdir(local_dir):
                state.add_log(f"Скачивание модели {model_name}...")
                snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
                
                if["strangerguardhf", "prithivmlmods"] in model_name.lower():
                    for item in os.listdir(local_dir):
                        if item.startswith("checkpoint-"):
                            chk_path = os.path.join(local_dir, item)
                            if os.path.isdir(chk_path):
                                try:
                                    shutil.rmtree(chk_path)
                                    state.add_log(f"Удален лишний чекпоинт: {item}")
                                except Exception as e:
                                    state.add_log(f"Не удалось удалить {item}: {e}")
            
            kwargs = {
                "torch_dtype": self.dtype,
            }
            if self.quant_mode != "None" and self.device == "cuda":
                kwargs["device_map"] = {"": self.device}
                try:
                    from transformers import BitsAndBytesConfig
                    if self.quant_mode == "8-bit":
                        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    elif self.quant_mode == "4-bit":
                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=self.dtype
                        )
                except ImportError:
                    state.add_log("⚠️ ОШИБКА: Для квантования установите пакеты: pip install bitsandbytes accelerate")

            self.processor = AutoImageProcessor.from_pretrained(local_dir)
            if "siglip" in model_name.lower():
                self.model = SiglipForImageClassification.from_pretrained(local_dir, **kwargs)
            else:
                self.model = AutoModelForImageClassification.from_pretrained(local_dir, **kwargs)
                
            if self.quant_mode == "None":
                self.model.to(self.dtype).to(self.device)
            self.model.eval()
            self.current_model_name = model_name
            self.current_model_state = current_state

    def unload(self):
        if self.model is not None:
            state.add_log(f"Выгрузка NSFW модели {self.current_model_name} из VRAM...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.current_model_name = None
            self.current_model_state = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def compute_danger(self, details):
        """ Высчитывает 'вероятность опасности', игнорируя нейтральные классы для любой модели """
        safe_labels = {'safe', 'sfw', 'normal', 'general', 'neutral', 'drawing', 'safe_content', 'anime picture', 'anime'}
        return sum(prob for lbl, prob in details.items() if lbl.lower() not in safe_labels)

    def evaluate_media(self, directory_path, model_name, allowed_exts, override_files=None):
        all_files = self.se._gather_files(directory_path, allowed_exts) if override_files is None else[f for f in override_files if f.lower().endswith(allowed_exts)]
        path_to_hash = self.db_cache.get_or_create_hashes(all_files)
        
        image_paths = [p for p in all_files if p.lower().endswith(SUPPORTED_IMAGES)]
        video_paths =[p for p in all_files if p.lower().endswith(SUPPORTED_VIDEOS)]
        
        state.add_log(f"Найдено для NSFW детектора: {len(image_paths)} изображений, {len(video_paths)} видео.")
        results =[]
        cache_key = f"{model_name}_{self.video_frames}"

        images_to_process, videos_to_process = [],[]
        for p in image_paths:
            h = path_to_hash.get(p)
            if not h: continue
            cached = self.db_cache.get_nsfw_score(cache_key, h)
            if cached is not None:
                details_dict = json.loads(cached[2]) if cached[2] else {}
                results.append((cached[1], p, cached[0], details_dict))
            else: images_to_process.append(p)

        for p in video_paths:
            h = path_to_hash.get(p)
            if not h: continue
            cached = self.db_cache.get_nsfw_score(cache_key, h)
            if cached is not None:
                details_dict = json.loads(cached[2]) if cached[2] else {}
                results.append((cached[1], p, cached[0], details_dict))
            else: videos_to_process.append(p)
                
        if images_to_process or videos_to_process: self.load_model(model_name)
        else:
            results.sort(key=lambda x: x[0], reverse=True)
            return results

        # --- ИЗОБРАЖЕНИЯ ---
        batch_images, batch_paths = [],[]
        for i, img_path in enumerate(images_to_process):
            if not state.is_processing: break
            state.status_text = f"NSFW Фото: {Path(img_path).name} ({i+1}/{len(images_to_process)})"
            h = path_to_hash.get(img_path)
            try:
                image = media_cache.get_image(img_path, self.max_dim)
                if image:
                    batch_images.append(image)
                    batch_paths.append(img_path)
                else:
                    if h: self.db_cache.save_nsfw_score(cache_key, h, "error", 0.0, {"error": 1.0})
                    results.append((0.0, img_path, "error", {"error": 1.0}))
            except Exception as e: 
                if h: self.db_cache.save_nsfw_score(cache_key, h, "error", 0.0, {"error": 1.0})
                results.append((0.0, img_path, "error", {"error": 1.0}))
                
            if len(batch_images) >= self.batch_size or (i == len(images_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(images_to_process))
                try:
                    inputs = self.processor(images=batch_images, return_tensors="pt")
                    inputs = {k: v.to(self.dtype).to(self.device) if v.is_floating_point() else v.to(self.device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        logits = self.model(**inputs).logits
                    probs = torch.nn.functional.softmax(logits, dim=-1).cpu()
                    
                    for j, p in enumerate(batch_paths):
                        prob_dist = probs[j]
                        top_label = self.model.config.id2label[prob_dist.argmax(-1).item()]
                        details = {self.model.config.id2label[idx]: float(val) for idx, val in enumerate(prob_dist)}
                        danger = self.compute_danger(details)
                        h_val = path_to_hash.get(p)
                        if h_val: self.db_cache.save_nsfw_score(cache_key, h_val, top_label, danger, details)
                        results.append((danger, p, top_label, details))
                except Exception as e: state.add_log(f"Ошибка инференса: {e}")
                batch_images, batch_paths = [],[]

        # --- ВИДЕО ---
        batch_images, batch_frame_counts, batch_paths = [], [],[]
        for i, vid_path in enumerate(videos_to_process):
            if not state.is_processing: break
            state.status_text = f"NSFW Видео: {Path(vid_path).name} ({i+1}/{len(videos_to_process)})"
            h = path_to_hash.get(vid_path)
            try:
                frames = media_cache.get_video_frames(vid_path, self.max_dim, self.video_frames)
                if frames:
                    batch_images.extend(frames)
                    batch_paths.append(vid_path)
                    batch_frame_counts.append(len(frames))
                else:
                    if h: self.db_cache.save_nsfw_score(cache_key, h, "error", 0.0, {"error": 1.0})
                    results.append((0.0, vid_path, "error", {"error": 1.0}))
            except Exception as e: 
                if h: self.db_cache.save_nsfw_score(cache_key, h, "error", 0.0, {"error": 1.0})
                results.append((0.0, vid_path, "error", {"error": 1.0}))

            if len(batch_images) >= self.batch_size or (i == len(videos_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(videos_to_process))
                try:
                    all_probs =[]
                    for k in range(0, len(batch_images), self.batch_size):
                        chunk = batch_images[k:k+self.batch_size]
                        inputs = self.processor(images=chunk, return_tensors="pt")
                        inputs = {k: v.to(self.dtype).to(self.device) if v.is_floating_point() else v.to(self.device) for k, v in inputs.items()}
                        with torch.inference_mode():
                            logits = self.model(**inputs).logits
                        all_probs.extend(torch.nn.functional.softmax(logits, dim=-1).cpu())
                    
                    idx = 0
                    for p, count in zip(batch_paths, batch_frame_counts):
                        vid_probs = torch.stack(all_probs[idx : idx + count])
                        idx += count
                        avg_probs = vid_probs.mean(dim=0)
                        top_label = self.model.config.id2label[avg_probs.argmax(-1).item()]
                        details = {self.model.config.id2label[k]: float(val) for k, val in enumerate(avg_probs)}
                        danger = self.compute_danger(details)
                        
                        h_val = path_to_hash.get(p)
                        if h_val: self.db_cache.save_nsfw_score(cache_key, h_val, top_label, danger, details)
                        results.append((danger, p, top_label, details))
                except Exception as e: state.add_log(f"Ошибка инференса: {e}")
                batch_images, batch_frame_counts, batch_paths = [], [], []

        results.sort(key=lambda x: x[0], reverse=True)
        return results

class FaceEngine:
    def __init__(self, search_engine):
        self.se = search_engine
        self.db_cache = search_engine.db_cache
        self.app = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.providers =['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else['CPUExecutionProvider']
        self.batch_size = 16

    def load_model(self):
        global INSIGHTFACE_AVAILABLE
        if not INSIGHTFACE_AVAILABLE:
            raise Exception("InsightFace не установлен! Установите: pip install insightface onnxruntime-gpu")
            
        if self.app is None:
            state.add_log(f"Загрузка InsightFace (buffalo_l) на {self.device}...")
            model_dir = os.path.join(current_dir, "models", "insightface")
            os.makedirs(model_dir, exist_ok=True)
            # Внимание: insightface сам создаст внутри подпапку models/buffalo_l
            self.app = FaceAnalysis(name='buffalo_l', root=model_dir, providers=self.providers)
            self.app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

    def unload(self):
        if self.app is not None:
            state.add_log(f"Выгрузка InsightFace из VRAM...")
            del self.app
            self.app = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def extract_faces(self, img_path):
        try:
            # Читаем через PIL для обхода проблем с Unicode-путями (в отличие от cv2.imread)
            img = Image.open(img_path).convert('RGB')
            # Слегка уменьшаем гигантские фото, чтобы InsightFace не выпадал с ООМ
            img.thumbnail((1920, 1920))
            img_arr = np.array(img)
            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            faces = self.app.get(img_bgr)
            return [f.embedding for f in faces]
        except Exception as e:
            state.add_log(f"Ошибка извлечения лиц {Path(img_path).name}: {e}")
            return[]

    def search_faces(self, ref_img_path, directory_path, allowed_exts, threshold, override_files=None):
        self.load_model()
        ref_embs = self.extract_faces(ref_img_path)
        if not ref_embs: raise Exception("На референсном фото (шаблоне) не найдено лиц!")
        
        ref_n = ref_embs[0] / np.linalg.norm(ref_embs[0])
        all_files = self.se._gather_files(directory_path, allowed_exts) if override_files is None else[f for f in override_files if f.lower().endswith(allowed_exts)]
        path_to_hash = self.db_cache.get_or_create_hashes(all_files)
        
        results, images_to_process =[],[]
        
        for p in all_files:
            h = path_to_hash.get(p)
            if not h: continue
            cached = self.db_cache.get_face_embeddings(h)
            if cached is not None:
                if len(cached) > 0:
                    max_sim = max([np.dot(emb / np.linalg.norm(emb), ref_n) for emb in cached], default=-1.0)
                    if max_sim >= threshold: results.append((float(max_sim), p))
            else: images_to_process.append(p)
                
        if images_to_process:
            state.add_log(f"Извлечение лиц для {len(images_to_process)} новых файлов...")
            batch_paths =[]
            for i, p in enumerate(images_to_process):
                if not state.is_processing: break
                batch_paths.append(p)
                
                if len(batch_paths) >= self.batch_size or i == len(images_to_process) - 1:
                    state.progress = (i + 1) / max(1, len(images_to_process))
                    batch_db_data =[]
                    for path in batch_paths:
                        ext = os.path.splitext(path)[1].lower()
                        if ext in SUPPORTED_IMAGES: embs = self.extract_faces(path)
                        elif ext in SUPPORTED_VIDEOS:
                            frames = media_cache.get_video_frames(path, 640, 1)
                            if frames and len(frames) > 0:
                                faces = self.app.get(cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2BGR))
                                embs = [f.embedding for f in faces]
                            else: embs =[]
                        else: embs =[]
                            
                        h_val = path_to_hash.get(path)
                        if h_val: batch_db_data.append((h_val, embs))
                        
                        if embs:
                            max_sim = max([np.dot(emb / np.linalg.norm(emb), ref_n) for emb in embs], default=-1.0)
                            if max_sim >= threshold: results.append((float(max_sim), path))
                                
                    self.db_cache.save_face_embeddings_batch(batch_db_data)
                    batch_paths =[]
                        
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def build_cache(self, directory_path, allowed_exts, override_files=None):
        self.load_model()
        all_files = self.se._gather_files(directory_path, allowed_exts) if override_files is None else[f for f in override_files if f.lower().endswith(allowed_exts)]
        path_to_hash = self.db_cache.get_or_create_hashes(all_files)
        
        images_to_process =[p for p in all_files if path_to_hash.get(p) and self.db_cache.get_face_embeddings(path_to_hash[p]) is None]
        if not images_to_process: return
            
        state.add_log(f"Кэширование лиц для {len(images_to_process)} файлов...")
        batch_paths =[]
        for i, p in enumerate(images_to_process):
            if not state.is_processing: break
            batch_paths.append(p)
            
            if len(batch_paths) >= self.batch_size or i == len(images_to_process) - 1:
                state.progress = (i + 1) / max(1, len(images_to_process))
                batch_db_data =[]
                for path in batch_paths:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in SUPPORTED_IMAGES: embs = self.extract_faces(path)
                    elif ext in SUPPORTED_VIDEOS:
                        frames = media_cache.get_video_frames(path, 640, 1)
                        if frames and len(frames) > 0:
                            faces = self.app.get(cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2BGR))
                            embs =[f.embedding for f in faces]
                        else: embs =[]
                    else: embs =[]
                        
                    h_val = path_to_hash.get(path)
                    if h_val: batch_db_data.append((h_val, embs))
                    
                self.db_cache.save_face_embeddings_batch(batch_db_data)
                batch_paths =[]

# --- ДВИЖОК ТЕГИРОВАНИЯ DANBOORU ---
class TagEngine:
    def __init__(self, search_engine):
        self.se = search_engine
        self.db_cache = search_engine.db_cache
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.session = None
        self.tag_names =[]
        self.model_name = None
        self.target_size = 448
        self.batch_size = 16
        self.video_frames = 4
        self.min_save_threshold = 0.1

    def load_model(self, model_repo):
        if self.model_name == model_repo and self.session is not None:
            return

        self.unload()
        state.add_log(f"Загрузка Tag-модели {model_repo} на {self.device}...")
        
        try:
            import onnxruntime as rt
            import pandas as pd
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise Exception("Установите зависимости: pip install onnxruntime-gpu pandas huggingface_hub")

        local_dir = os.path.join(current_dir, "models", model_repo.replace("/", "_"))
        os.makedirs(local_dir, exist_ok=True)

        # 1. Загрузка CSV тегов
        from huggingface_hub import list_repo_files, hf_hub_download
        
        # 1. Умный поиск файла тегов и ONNX модели
        csv_path = os.path.join(local_dir, "tags.csv")
        json_path = os.path.join(local_dir, "tags.json")
        txt_path = os.path.join(local_dir, "tags.txt")
        onnx_path = os.path.join(local_dir, "model.onnx")
        
        # Миграция со старых версий файлов
        old_csv = os.path.join(local_dir, "selected_tags.csv")
        if os.path.exists(old_csv) and not os.path.exists(csv_path): os.rename(old_csv, csv_path)
        old_json = os.path.join(local_dir, "tag_mapping.json")
        if os.path.exists(old_json) and not os.path.exists(json_path): os.rename(old_json, json_path)
        old_txt = os.path.join(local_dir, "top_tags.txt")
        if os.path.exists(old_txt) and not os.path.exists(txt_path): os.rename(old_txt, txt_path)

        repo_files =[]
        if not os.path.exists(onnx_path) or (not os.path.exists(csv_path) and not os.path.exists(json_path) and not os.path.exists(txt_path)):
            try:
                repo_files = list_repo_files(repo_id=model_repo)
            except Exception as e:
                raise Exception(f"Не удалось получить список файлов репозитория {model_repo}: {e}")

        # --- ЗАГРУЗКА ТЕГОВ ---
        if not os.path.exists(csv_path) and not os.path.exists(json_path) and not os.path.exists(txt_path):
            tag_filename = None
            # Приоритет 1: CSV файлы с 'tag' или 'class'
            for f in repo_files:
                if f.endswith('.csv') and ('tag' in f.lower() or 'class' in f.lower()):
                    tag_filename = f; break
            # Приоритет 2: JSON файлы с 'tag' или 'metadata'
            if not tag_filename:
                for f in repo_files:
                    if f.endswith('.json') and ('tag' in f.lower() or 'metadata' in f.lower()) and 'config' not in f.lower():
                        tag_filename = f; break
            # Приоритет 3: TXT файлы с 'tag' (для моделей вроде joytag)
            if not tag_filename:
                for f in repo_files:
                    if f.endswith('.txt') and 'tag' in f.lower():
                        tag_filename = f; break
                        
            if not tag_filename:
                raise Exception(f"Не удалось найти файл с тегами (.csv, .json или .txt) в репозитории {model_repo}")
                
            downloaded_path = hf_hub_download(repo_id=model_repo, filename=tag_filename, local_dir=local_dir)
            if downloaded_path.endswith('.csv'):
                os.rename(downloaded_path, csv_path)
            elif downloaded_path.endswith('.json'):
                os.rename(downloaded_path, json_path)
            elif downloaded_path.endswith('.txt'):
                os.rename(downloaded_path, txt_path)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'name' in df.columns:
                self.tag_names = df['name'].fillna('unknown_tag').astype(str).tolist()
            else:
                self.tag_names = df.iloc[:, 0].fillna('unknown_tag').astype(str).tolist()
        elif os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            extracted_names =[]
            
            def extract_from_list(lst):
                names =[]
                if len(lst) > 0 and isinstance(lst[0], dict):
                    if any(key in lst[0] for key in['id', 'tag_id', 'tag_index']):
                        max_id = max((int(item.get('id', item.get('tag_id', item.get('tag_index', -1)))) for item in lst if isinstance(item, dict) and str(item.get('id', item.get('tag_id', item.get('tag_index', '')))).lstrip('-').isdigit()), default=-1)
                        if max_id >= 0:
                            names = ['unknown_tag'] * (max_id + 1)
                            for item in lst:
                                if isinstance(item, dict):
                                    val = str(item.get('id', item.get('tag_id', item.get('tag_index', '-1'))))
                                    if val.lstrip('-').isdigit():
                                        idx = int(val)
                                        if idx >= 0: names[idx] = str(item.get('name', item.get('tag', str(item))))
                            return names
                    for item in lst:
                        if isinstance(item, dict):
                            names.append(str(item.get('name', item.get('tag', str(item)))))
                        else:
                            names.append(str(item))
                else:
                    names =[str(x) for x in lst]
                return names

            if isinstance(data, dict):
                # Уникальная структура для Camie-Tagger v2
                if "dataset_info" in data and isinstance(data["dataset_info"], dict):
                    mapping = data["dataset_info"].get("tag_mapping", {})
                    if isinstance(mapping, dict):
                        if "idx_to_tag" in mapping: data = mapping["idx_to_tag"]
                        elif "tag_to_idx" in mapping: data = mapping["tag_to_idx"]

                if isinstance(data, dict):
                    for key in["tags", "tag_names", "classes", "labels"]:
                        if key in data and isinstance(data[key], (list, dict)):
                            data = data[key]
                            break

            if isinstance(data, list):
                extracted_names = extract_from_list(data)
            elif isinstance(data, dict):
                keys_are_ints = all(str(k).isdigit() for k in data.keys() if k != "meta")
                if keys_are_ints:
                    int_keys =[int(k) for k in data.keys() if str(k).isdigit()]
                    if int_keys:
                        max_idx = max(int_keys)
                        extracted_names = ['unknown_tag'] * (max_idx + 1)
                        for k, v in data.items():
                            if not str(k).isdigit(): continue
                            idx = int(k)
                            if isinstance(v, dict):
                                extracted_names[idx] = str(v.get('name', v.get('tag', str(v))))
                            else:
                                extracted_names[idx] = str(v)
                else:
                    is_name_to_id = any(isinstance(v, int) or str(v).lstrip('-').isdigit() for v in data.values())
                    if is_name_to_id:
                        valid_pairs =[(int(v), str(k)) for k, v in data.items() if isinstance(v, (int, str)) and str(v).lstrip('-').isdigit()]
                        if valid_pairs:
                            max_idx = max(idx for idx, _ in valid_pairs)
                            extracted_names = ['unknown_tag'] * (max_idx + 1)
                            for idx, name in valid_pairs:
                                if idx >= 0: extracted_names[idx] = name
                    
                    if not extracted_names:
                        for k, v in data.items():
                            if isinstance(v, list) and len(v) > 50:
                                extracted_names = extract_from_list(v)
                                break
                            elif isinstance(v, dict) and len(v) > 50:
                                valid_vals =[int(val) for val in v.values() if isinstance(val, int) or str(val).lstrip('-').isdigit()]
                                if valid_vals:
                                    max_val = max(valid_vals)
                                    if max_val >= 0:
                                        extracted_names = ['unknown_tag'] * (max_val + 1)
                                        for tk, tv in v.items():
                                            if isinstance(tv, int) or str(tv).lstrip('-').isdigit():
                                                idx = int(tv)
                                                if idx >= 0: extracted_names[idx] = str(tk)
                                        break

            if extracted_names:
                self.tag_names = extracted_names
            else:
                self.tag_names =['unknown_tag'] * 100000
                state.add_log(f"⚠️ ОШИБКА ЧТЕНИЯ ТЕГОВ! Структура неизвестна. Первые 200 символов: {str(data)[:200]}")
        elif os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                self.tag_names =[line.strip() for line in f if line.strip()]

        # --- ЗАГРУЗКА ONNX ---
        if not os.path.exists(onnx_path):
            onnx_filename = "model.onnx"
            if "model.onnx" not in repo_files:
                for f in repo_files:
                    if f.endswith(".onnx"):
                        onnx_filename = f; break
            try:
                downloaded_onnx = hf_hub_download(repo_id=model_repo, filename=onnx_filename, local_dir=local_dir)
                if downloaded_onnx != onnx_path:
                    os.rename(downloaded_onnx, onnx_path)
            except Exception:
                raise Exception(f"Не удалось скачать ONNX модель из репозитория {model_repo}")

        providers =['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else['CPUExecutionProvider']
        self.session = rt.InferenceSession(onnx_path, providers=providers)
        self.model_name = model_repo

        # Динамическое определение размера входа (NCHW или NHWC)
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            self.target_size = input_shape[2] if input_shape[1] == 3 else input_shape[1]
            if not isinstance(self.target_size, int): self.target_size = 448
        else:
            self.target_size = 448

    def unload(self):
        if self.session is not None:
            state.add_log(f"Выгрузка Tag-модели {self.model_name} из VRAM...")
            del self.session
            self.session = None
            self.model_name = None
            self.tag_names =[]
            gc.collect()

    def evaluate_media(self, directory_path, model_name, allowed_exts, override_files=None):
        all_files = self.se._gather_files(directory_path, allowed_exts) if override_files is None else[f for f in override_files if f.lower().endswith(allowed_exts)]
        image_paths =[p for p in all_files if p.lower().endswith(SUPPORTED_IMAGES)]
        video_paths =[p for p in all_files if p.lower().endswith(SUPPORTED_VIDEOS)]
        
        state.add_log(f"Найдено для тегирования: {len(image_paths)} фото, {len(video_paths)} видео.")
        cache_key = f"{model_name}_{self.video_frames}"
        
        path_to_hash = self.db_cache.get_or_create_hashes(all_files)
        
        images_to_process, videos_to_process = [],[]
        for p in image_paths:
            if path_to_hash.get(p) and self.db_cache.get_tags(cache_key, path_to_hash[p]) is None: images_to_process.append(p)
        for p in video_paths:
            if path_to_hash.get(p) and self.db_cache.get_tags(cache_key, path_to_hash[p]) is None: videos_to_process.append(p)

        if images_to_process or videos_to_process: self.load_model(model_name)
        else: return

        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        is_nchw = (input_shape[1] == 3)

        def process_batch(imgs, paths):
            try:
                img_arrs =[]
                for img in imgs:
                    max_dim = max(img.width, img.height)
                    padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
                    padded.paste(img, ((max_dim - img.width) // 2, (max_dim - img.height) // 2))
                    padded = padded.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)
                    arr = np.array(padded, dtype=np.float32)[:, :, ::-1]
                    if is_nchw: arr = arr.transpose(2, 0, 1)
                    img_arrs.append(arr)
                    
                probs_batch = self.session.run(None, {input_name: np.stack(img_arrs)})[0]
                probs_batch = np.array(probs_batch, dtype=np.float32)
                if probs_batch.max() > 1.0 or probs_batch.min() < 0.0:
                    probs_batch = 1 / (1 + np.exp(-np.clip(probs_batch, -100, 100)))
                
                db_data =[]
                for j, p in enumerate(paths):
                    probs = probs_batch[j]
                    tags_dict = {str(tag): float(prob) for tag, prob in zip(self.tag_names, probs) if float(prob) >= self.min_save_threshold}
                    h_val = path_to_hash.get(p)
                    if h_val: db_data.append((cache_key, h_val, tags_dict))
                    
                self.db_cache.save_tags_batch(db_data)
            except Exception as e:
                state.add_log(f"⚠️ Ошибка инференса тегов: {e}")
                self.db_cache.save_tags_batch([(cache_key, path_to_hash.get(p), {}) for p in paths if path_to_hash.get(p)])

        # Обработка фото (Только меняем save_tags на использование хеша)
        batch_images, batch_paths = [],[]
        for i, img_path in enumerate(images_to_process):
            if not state.is_processing: break
            state.status_text = f"Теги фото: {Path(img_path).name} ({i+1}/{len(images_to_process)})"
            try:
                image = media_cache.get_image(img_path, self.target_size)
                if image:
                    batch_images.append(image)
                    batch_paths.append(img_path)
                else:
                    if path_to_hash.get(img_path): self.db_cache.save_tags(cache_key, path_to_hash[img_path], {})
            except Exception as e:
                if path_to_hash.get(img_path): self.db_cache.save_tags(cache_key, path_to_hash[img_path], {})

            if len(batch_images) >= self.batch_size or (i == len(images_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(images_to_process))
                process_batch(batch_images, batch_paths)
                batch_images, batch_paths =[],[]

        # --- ОБРАБОТКА ВИДЕО ---
        batch_images, batch_frame_counts, batch_paths = [], [],[]
        for i, vid_path in enumerate(videos_to_process):
            time.sleep(0.002)
            if not state.is_processing: break
            state.status_text = f"Теги видео: {Path(vid_path).name} ({i+1}/{len(videos_to_process)})"
            
            try:
                frames = media_cache.get_video_frames(vid_path, self.target_size, self.video_frames)
                if frames:
                    batch_images.extend(frames)
                    batch_paths.append(vid_path)
                    batch_frame_counts.append(len(frames))
                else:
                    state.add_log(f"⚠️ Ошибка: не удалось извлечь кадры из {Path(vid_path).name}")
                    self.db_cache.save_tags(cache_key, vid_path, {})
            except Exception as e:
                state.add_log(f"⚠️ Ошибка загрузки видео {Path(vid_path).name}: {e}")
                self.db_cache.save_tags(cache_key, vid_path, {})

            if len(batch_images) >= self.batch_size or (i == len(videos_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(videos_to_process))
                
                try:
                    all_probs =[]
                    for k in range(0, len(batch_images), self.batch_size):
                        chunk = batch_images[k:k+self.batch_size]
                        img_arrs =[]
                        for img in chunk:
                            max_dim = max(img.width, img.height)
                            padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
                            padded.paste(img, ((max_dim - img.width) // 2, (max_dim - img.height) // 2))
                            padded = padded.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)
                            arr = np.array(padded, dtype=np.float32)[:, :, ::-1]
                            if is_nchw: arr = arr.transpose(2, 0, 1)
                            img_arrs.append(arr)
                        batch_inputs = np.stack(img_arrs)
                            
                        probs_chunk = self.session.run(None, {input_name: batch_inputs})[0]
                        probs_chunk = np.array(probs_chunk, dtype=np.float32)
                        if probs_chunk.max() > 1.0 or probs_chunk.min() < 0.0:
                            probs_chunk = 1 / (1 + np.exp(-np.clip(probs_chunk, -100, 100)))
                        all_probs.extend(probs_chunk)
                        
                    idx = 0
                    db_data =[]
                    for p, count in zip(batch_paths, batch_frame_counts):
                        vid_probs = np.stack(all_probs[idx : idx + count])
                        idx += count
                        max_probs = vid_probs.max(axis=0)
                        tags_dict = {str(tag): float(prob) for tag, prob in zip(self.tag_names, max_probs) if float(prob) >= self.min_save_threshold}
                        h_val = path_to_hash.get(p)
                        if h_val: db_data.append((cache_key, h_val, tags_dict))
                        
                    self.db_cache.save_tags_batch(db_data)
                except Exception as e: state.add_log(f"⚠️ Ошибка инференса тегов (видео): {e}")
                
                batch_images, batch_frame_counts, batch_paths = [], [],[]

class DuplicatesEngine:
    def __init__(self, search_engine):
        self.se = search_engine
        self.db_cache = search_engine.db_cache

    def _extract_video_frames_for_phash(self, path, mode):
        try:
            import av
            frames =[]
            with av.open(path) as container:
                stream = container.streams.video[0]
                total_frames = stream.frames
                if not total_frames or total_frames <= 0:
                    total_frames = 100 # Фолбэк, если метаданные битые
                    
                if mode == '1 кадр (Самое начало 0%)': indices = [0]
                elif mode == '1 кадр (Середина 50%)': indices =[total_frames // 2]
                elif mode == '3 кадра (0%, 50%, 100%)': indices = [0, total_frames // 2, total_frames - 1]
                elif mode == '5 кадров (Равномерно)': indices =[int(i * (total_frames - 1) / 4) for i in range(5)]
                elif mode == '10 кадров (Равномерно)': indices =[int(i * (total_frames - 1) / 9) for i in range(10)]
                else: indices =[total_frames // 2]
                
                target_indices = sorted(list(set(indices)))
                target_idx = 0
                
                for i, frame in enumerate(container.decode(video=0)):
                    if target_idx < len(target_indices) and i == target_indices[target_idx]:
                        frames.append(frame.to_image().convert("RGB"))
                        target_idx += 1
                    if target_idx >= len(target_indices):
                        break
                        
            # Фолбэк, если видео оказалось короче и не выдало кадров
            if not frames and target_indices:
                with av.open(path) as container:
                    for frame in container.decode(video=0):
                        frames.append(frame.to_image().convert("RGB"))
                        break
            return frames
        except Exception as e:
            state.add_log(f"⚠️ Ошибка извлечения кадров (pHash) из {path}: {e}")
            return[]
        
    def find_exact(self, dir_paths, allowed_exts):
        files = self.se._gather_files(dir_paths, allowed_exts)
        path_to_hash = self.db_cache.get_or_create_hashes(files)
        
        hash_groups = defaultdict(list)
        for p, h in path_to_hash.items():
            hash_groups[h].append(p)
            
        results =[paths for h, paths in hash_groups.items() if len(paths) > 1]
        results.sort(key=lambda g: len(g), reverse=True)
        for g in results: g.sort()
        return results

    def find_similar(self, dir_paths, allowed_exts, threshold, video_frames_mode):
        files = self.se._gather_files(dir_paths, allowed_exts)
        path_to_hash = self.db_cache.get_or_create_hashes(files)
        
        c = self.db_cache.conn.cursor()
        hashes = list(set(path_to_hash.values()))
        
        phash_dict = {}
        chunk_size = 900
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i:i+chunk_size]
            ph = ','.join(['?']*len(chunk))
            c.execute(f"SELECT hash, phash FROM phash_cache WHERE hash IN ({ph})", chunk)
            for row in c.fetchall(): phash_dict[row[0]] = row[1]
                
        missing_paths =[p for p in files if path_to_hash.get(p) not in phash_dict]
        if missing_paths:
            state.add_log(f"Вычисление pHash для {len(missing_paths)} новых файлов (включая видео)...")
            to_insert =[]
            last_update = time.time()
            for i, p in enumerate(missing_paths):
                if state.is_processing == False or self.se.cancel_flag: break
                try:
                    ext = os.path.splitext(p)[1].lower()
                    hashes_str = ""
                    if ext in SUPPORTED_IMAGES:
                        with Image.open(p) as img:
                            hashes_str = str(imagehash.phash(img))
                    elif ext in SUPPORTED_VIDEOS:
                        frames = self._extract_video_frames_for_phash(p, video_frames_mode)
                        h_list =[str(imagehash.phash(f)) for f in frames]
                        if h_list:
                            hashes_str = ",".join(h_list)
                            
                    if hashes_str:
                        h = path_to_hash[p]
                        phash_dict[h] = hashes_str
                        to_insert.append((h, hashes_str))
                except Exception: pass
                
                # Защита от обрыва: обновляем UI
                if time.time() - last_update > 0.5:
                    state.progress = i / max(1, len(missing_paths))
                    last_update = time.time()
                    time.sleep(0.001)
            
            if to_insert:
                c.executemany("INSERT OR REPLACE INTO phash_cache (hash, phash) VALUES (?, ?)", to_insert)
                self.db_cache.conn.commit()
        
        state.add_log("Кластеризация дубликатов (Кросс-медиа пулинг)...")
        
        # Конвертируем HEX-строки в числа uint64 для молниеносных битовых операций
        hash_objs = {}
        for h, ph_str in phash_dict.items():
            if not ph_str: continue
            try:
                hash_objs[h] = [int(x, 16) for x in ph_str.split(',')]
            except Exception: pass
            
        parent = {h: h for h in hash_objs.keys()}
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j: parent[root_i] = root_j
        
        hash_list = list(hash_objs.keys())
        total = len(hash_list)
        last_update = time.time()
        
        for i in range(total):
            if self.se.cancel_flag: break
            h1 = hash_list[i]
            obj1_list = hash_objs[h1]
            
            for j in range(i+1, total):
                h2 = hash_list[j]
                obj2_list = hash_objs[h2]
                
                # Матричное сравнение Картинка<->Видео (если дистанция <= порогу хоть для 1 пары -> дубликат)
                match = False
                for val1 in obj1_list:
                    for val2 in obj2_list:
                        # Используем bit_count (Python 3.10+) или фолбэк для старых версий
                        try:
                            dist = (val1 ^ val2).bit_count()
                        except AttributeError:
                            dist = bin(val1 ^ val2).count('1')
                            
                        if dist <= threshold:
                            match = True
                            break
                    if match: break
                        
                if match:
                    union(h1, h2)
                    
            if time.time() - last_update > 0.5:
                state.progress = i / max(1, total)
                last_update = time.time()
                time.sleep(0.002)
                    
        groups = defaultdict(list)
        for p in files:
            h = path_to_hash.get(p)
            if h in hash_objs:
                root = find(h)
                groups[root].append(p)
                
        results = [paths for paths in groups.values() if len(paths) > 1]
        results.sort(key=lambda g: len(g), reverse=True)
        for g in results: g.sort()
        return results

class ClusteringEngine:
    def __init__(self, search_engine):
        self.se = search_engine
        self.db_cache = search_engine.db_cache

    def build_clusters(self, dir_paths, allowed_exts, algo, n_clusters, eps, emb_model_name, emb_size, batch_size, video_frames, quant_mode, align_domains=True):
        if not SKLEARN_AVAILABLE:
            raise Exception("Установите библиотеку scikit-learn: pip install scikit-learn")

        files = self.se._gather_files(dir_paths, allowed_exts)
        path_to_hash = self.db_cache.get_or_create_hashes(files)
        
        cache_key = emb_model_name if emb_size == 512 else f"{emb_model_name}_{emb_size}"
        
        c = self.db_cache.conn.cursor()
        hashes = list(set(path_to_hash.values()))
        
        state.add_log(f"Сбор нейросетевых признаков (эмбеддингов) из БД для {len(hashes)} файлов...")
        
        emb_dict = {}
        chunk_size = 900
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i:i+chunk_size]
            ph = ','.join(['?']*len(chunk))
            c.execute(f"SELECT hash, features FROM emb_cache WHERE model=? AND hash IN ({ph})",[cache_key] + chunk)
            for row in c.fetchall():
                try:
                    feat_tensor = torch.load(io.BytesIO(row[1]), weights_only=False)
                    emb_dict[row[0]] = feat_tensor.float().cpu().numpy().flatten()
                except Exception as e: pass

        valid_paths =[]
        valid_embs = []
        missing_paths =[]
        
        for p in files:
            h = path_to_hash.get(p)
            if h in emb_dict:
                valid_paths.append(p)
                valid_embs.append(emb_dict[h])
            else:
                missing_paths.append(p)

        # Вычисление "на лету"
        if missing_paths:
            state.add_log(f"Вычисление ИИ-векторов для {len(missing_paths)} новых файлов (On-the-fly)...")
            old_vf, old_es, old_qm = self.se.video_frames, self.se.emb_size, self.se.quant_mode
            self.se.video_frames, self.se.emb_size, self.se.quant_mode = video_frames, emb_size, quant_mode
            
            self.se.build_cache(dir_paths, emb_model_name, batch_size, allowed_exts, override_files=missing_paths)
            self.se.video_frames, self.se.emb_size, self.se.quant_mode = old_vf, old_es, old_qm
            
            if self.se.cancel_flag: raise Exception("Инференс отменен пользователем.")
                
            for i in range(0, len(missing_paths), chunk_size):
                chunk_paths = missing_paths[i:i+chunk_size]
                chunk_hashes =[path_to_hash[p] for p in chunk_paths]
                ph = ','.join(['?']*len(chunk_hashes))
                c.execute(f"SELECT hash, features FROM emb_cache WHERE model=? AND hash IN ({ph})",[cache_key] + chunk_hashes)
                for row in c.fetchall():
                    try:
                        feat_tensor = torch.load(io.BytesIO(row[1]), weights_only=False)
                        emb_dict[row[0]] = feat_tensor.float().cpu().numpy().flatten()
                    except Exception: pass
                        
            for p in missing_paths:
                h = path_to_hash.get(p)
                if h in emb_dict:
                    valid_paths.append(p)
                    valid_embs.append(emb_dict[h])

        if not valid_paths:
            raise Exception("Не удалось получить эмбеддинги для кластеризации. Возможно, файлы повреждены.")

        self.se._unload_embedding_model()

        state.add_log(f"Найдено {len(valid_paths)} файлов. Обработка математического пространства...")
        
        X = np.array(valid_embs)
        X = normalize(X) # L2 Нормализация

        # МАГИЯ ЗДЕСЬ: Устранение "разрыва доменов" (Mean Centering)
        if align_domains:
            img_indices =[i for i, p in enumerate(valid_paths) if p.lower().endswith(SUPPORTED_IMAGES)]
            vid_indices =[i for i, p in enumerate(valid_paths) if p.lower().endswith(SUPPORTED_VIDEOS)]
            
            if len(img_indices) > 0 and len(vid_indices) > 0:
                state.add_log("Применение алгоритма слияния доменов (Коррекция разрыва Картинка ↔ Видео)...")
                mean_img = np.mean(X[img_indices], axis=0)
                mean_vid = np.mean(X[vid_indices], axis=0)
                # Сдвигаем векторы видео в центр векторов картинок
                X[vid_indices] += (mean_img - mean_vid)
                # Повторная нормализация после сдвига
                X = normalize(X)

        state.add_log(f"Запуск алгоритма {algo}...")

        if algo == 'K-Means':
            n_c = min(n_clusters, len(X))
            model = KMeans(n_clusters=n_c, random_state=42, n_init='auto')
            labels = model.fit_predict(X)
        else:
            model = DBSCAN(eps=eps, min_samples=2, metric='euclidean')
            labels = model.fit_predict(X)

        state.add_log("Кластеризация завершена. Генерация названий папок на основе тегов...")
        
        clusters = defaultdict(list)
        for p, lbl in zip(valid_paths, labels):
            clusters[lbl].append(p)

        tags_dict_global = {}
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i:i+chunk_size]
            ph = ','.join(['?']*len(chunk))
            c.execute(f"SELECT hash, tags FROM tags_cache WHERE hash IN ({ph})", chunk)
            for row in c.fetchall():
                if row[1]: tags_dict_global[row[0]] = json.loads(row[1])

        results =[]
        for lbl, paths in clusters.items():
            if lbl == -1:
                name = "Outliers_Noise"
            else:
                tag_counts = defaultdict(float)
                for p in paths:
                    h = path_to_hash.get(p)
                    if h in tags_dict_global:
                        for t, prob in tags_dict_global[h].items():
                            tag_counts[t] += prob
                
                sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
                top_tags = [t[0].replace(' ', '_').replace(':', '') for t in sorted_tags[:3]]
                name = f"Cluster_{lbl:03d}" + ("_" + "_".join(top_tags) if top_tags else "")
                
            results.append({"name": name, "paths": paths})

        results.sort(key=lambda x: len(x["paths"]), reverse=True)
        return results
        
# ==========================================
# 4. СОСТОЯНИЕ И UI УТИЛИТЫ
# ==========================================
class AppState:
    def __init__(self):
        self.search_results =[]
        self.aesthetic_results =[]
        self.nsfw_results =[]
        self.face_results =[]
        self.tags_results =[]
        self.dupes_results =[]
        self.cluster_results =[]
        self.sel_cluster = {}
        self.cluster_page = 1
        self.cluster_base_dir = ""
        self.cluster_res_filter = 'Все'
        
        self.sel_search = {}
        self.sel_aes = {}
        self.sel_nsfw = {}
        self.sel_face = {}
        self.sel_tags = {}
        self.sel_dupes = {}
        
        self.search_page = 1
        self.aes_page = 1
        self.nsfw_page = 1
        self.face_page = 1
        self.tags_page = 1
        
        self.search_base_dir = ""
        self.aes_base_dir = ""
        self.nsfw_base_dir = ""
        self.face_base_dir = ""
        self.tags_base_dir = ""
        
        self.search_res_filter = 'Все'
        self.aes_res_filter = 'Все'
        self.nsfw_res_filter = 'Все'
        self.face_res_filter = 'Все'
        self.tags_res_filter = 'Все'
        
        self.viewer_open = False
        self.viewer_items =[]
        self.viewer_index = 0
        
        self.is_processing = False
        self.progress = 0.0
        self.status_text = "Готов к работе"
        
        self.logs =[]
        self.full_log_history =[]
        self.current_tab = 'Search'

        # Глобальные настройки
        self.nsfw_threshold = 0.45
        self.flatten_structure = False
        self.grid_columns = 4
        self.groups_per_page = 5

        self.filter_min_res = 0
        self.filter_max_res = 10000
        self.filter_max_size = 10000.0
        self.filter_orientation = 'Любая'

    def add_log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self.logs.append(line)
        self.full_log_history.append(line)

state = AppState()
search_engine = SearchEngine(
    log_callback=lambda m: state.add_log(m),
    progress_callback=lambda p, m: setattr(state, 'status_text', m) or setattr(state, 'progress', p)
)
aesthetic_engine = AestheticEngine(search_engine)
nsfw_engine = NsfwEngine(search_engine)
face_engine = FaceEngine(search_engine)
tag_engine = TagEngine(search_engine)
dupes_engine = DuplicatesEngine(search_engine)
cluster_engine = ClusteringEngine(search_engine)

def open_file_native(filepath):
    try: os.startfile(filepath) if os.name == 'nt' else subprocess.call(('xdg-open', filepath))
    except Exception as e: ui.notify(f"Ошибка открытия: {e}", type='negative')

def reveal_file_native(filepath):
    try:
        if os.name == 'nt': # Windows
            subprocess.run(['explorer', '/select,', os.path.normpath(filepath)])
        elif sys.platform == 'darwin': # macOS
            subprocess.run(['open', '-R', filepath])
        else: # Linux
            desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
            if 'gnome' in desktop or 'unity' in desktop:
                subprocess.Popen(['nautilus', '--select', filepath])
            elif 'kde' in desktop:
                subprocess.Popen(['dolphin', '--select', filepath])
            else: # Fallback для остальных Linux
                subprocess.Popen(['xdg-open', os.path.dirname(filepath)])
    except Exception as e: 
        ui.notify(f"Ошибка открытия папки: {e}", type='negative')

def pick_folder_native():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder = filedialog.askdirectory()
    root.destroy()
    return folder

async def select_folder(input_element):
    folder = await run.io_bound(pick_folder_native)
    if folder: input_element.value = folder

async def select_folder_multi(textarea_element):
    folder = await run.io_bound(pick_folder_native)
    if folder:
        current = textarea_element.value.strip()
        textarea_element.value = current + "\n" + folder if current else folder

def clear_folder_cache_multi(paths_str):
    if not paths_str: return
    dirs =[d.strip() for d in paths_str.replace('\r', '\n').split('\n') if d.strip()]
    cleared = 0
    for d in dirs:
        if d in search_engine.files_cache._data:
            del search_engine.files_cache._data[d]
            cleared += 1
    if cleared:
        search_engine.files_cache.save_cache()
        ui.notify(f'Кэш очищен для {cleared} папок!', type='positive')

def pick_file_native():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    file = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff")])
    root.destroy()
    return file

async def select_file(input_element):
    file = await run.io_bound(pick_file_native)
    if file: input_element.value = file

def clear_folder_cache(folder_path):
    if not folder_path: return
    if folder_path in search_engine.files_cache._data:
        del search_engine.files_cache._data[folder_path]
        search_engine.files_cache.save_cache()
        ui.notify(f'Кэш индекса файлов для папки очищен!', type='positive')
    else:
        ui.notify(f'Папка не найдена в индексе (кэш пуст)', type='info')

def update_ui_logs():
    if 'ui_log_element' in globals() and state.logs:
        for msg in state.logs: ui_log_element.push(msg)
        state.logs.clear()

def clear_logs():
    state.full_log_history.clear()
    ui_log_element.clear()
    state.add_log("Логи очищены.")

def copy_logs():
    ui.clipboard.write('\n'.join(state.full_log_history))
    ui.notify('Логи скопированы!', type='positive', color='green')

# --- КРОССПЛАТФОРМЕННОЕ КОПИРОВАНИЕ В БУФЕР ОБМЕНА ---
def copy_image_to_clipboard(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_VIDEOS:
        ui.notify('Видео нельзя скопировать в буфер обмена', type='warning')
        return
    if ext in SUPPORTED_TEXTS:
        ui.notify('Текст нельзя скопировать как картинку', type='warning')
        return
        
    try:
        if os.name == 'nt':
            import ctypes
            from PIL import Image
            import io
            
            # Читаем любую картинку через PIL (отлично понимает webp)
            img = Image.open(path).convert('RGB')
            output = io.BytesIO()
            img.save(output, 'BMP')
            data = output.getvalue()[14:] # Пропускаем 14 байт заголовка BMP-файла
            output.close()
            
            # Используем встроенный ctypes для прямого доступа к API Windows
            CF_DIB = 8
            GMEM_MOVEABLE = 0x0002
            
            # Явно указываем типы, чтобы 64-битные указатели не обрезались до 32-битных
            ctypes.windll.kernel32.GlobalAlloc.restype = ctypes.c_void_p
            ctypes.windll.kernel32.GlobalAlloc.argtypes =[ctypes.c_uint, ctypes.c_size_t]
            ctypes.windll.kernel32.GlobalLock.restype = ctypes.c_void_p
            ctypes.windll.kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            ctypes.windll.kernel32.GlobalUnlock.restype = ctypes.c_int
            ctypes.windll.kernel32.GlobalUnlock.argtypes =[ctypes.c_void_p]
            ctypes.windll.user32.SetClipboardData.restype = ctypes.c_void_p
            ctypes.windll.user32.SetClipboardData.argtypes =[ctypes.c_uint, ctypes.c_void_p]
            
            hGlobalMem = ctypes.windll.kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not hGlobalMem:
                raise Exception("Не удалось выделить память")
                
            lpGlobalMem = ctypes.windll.kernel32.GlobalLock(hGlobalMem)
            if not lpGlobalMem:
                raise Exception("Не удалось заблокировать память")
                
            ctypes.memmove(lpGlobalMem, data, len(data))
            ctypes.windll.kernel32.GlobalUnlock(hGlobalMem)
            
            if not ctypes.windll.user32.OpenClipboard(0):
                raise Exception("Буфер обмена занят другим процессом")
                
            try:
                ctypes.windll.user32.EmptyClipboard()
                ctypes.windll.user32.SetClipboardData(CF_DIB, hGlobalMem)
            finally:
                ctypes.windll.user32.CloseClipboard()
                
        elif sys.platform == 'darwin':
            abs_path = os.path.abspath(path)
            subprocess.run(['osascript', '-e', f'set the clipboard to (read (POSIX file "{abs_path}") as JPEG picture)'])
        else:
            import mimetypes
            mimetype, _ = mimetypes.guess_type(path)
            if not mimetype: mimetype = 'image/png'
            subprocess.run(['xclip', '-selection', 'clipboard', '-t', mimetype, '-i', path])
            
        ui.notify('Картинка скопирована в буфер обмена!', type='positive')
    except Exception as e:
        ui.notify(f'Ошибка копирования в буфер: {e}', type='negative')

# ==========================================
# 5. ВЕРСТКА И ИНТЕРФЕЙС NICEGUI
# ==========================================
@ui.page('/')
async def index_page():
    cfg = load_config()
    state.nsfw_threshold = float(cfg.get('nsfw_threshold', 0.45))
    state.flatten_structure = bool(cfg.get('flatten_structure', False))
    state.grid_columns = int(cfg.get('grid_columns', 4))
    state.groups_per_page = int(cfg.get('groups_per_page', 5))

    def cancel_all_tasks():
        if state.is_processing:
            search_engine.cancel()         # Флаг для Поиска и Индексатора
            state.is_processing = False    # Флаг для Эстетики и NSFW
            state.add_log("🛑 Отправлен сигнал прерывания...")
            state.status_text = "Останавливаем процессы (завершение текущего батча)..."
            ui.notify('Останавливаем выполнение...', type='warning', position='top')

    ui.colors(primary='#2563eb', secondary='#10b981', accent='#f59e0b', dark='#1e1e2f')
    ui.query('body').classes('bg-[#121212] text-white overflow-hidden m-0 p-0')

    with ui.header().classes('bg-gray-900 border-b border-gray-800 flex justify-between items-center px-4 py-0 shrink-0 h-[60px]'):
        ui.label('🤖 AI Media Organizer Pro').classes('text-xl font-bold tracking-wider text-blue-400 shrink-0')
        
        # Добавляем серый цвет по умолчанию и пропсы для активного состояния
        with ui.tabs().bind_value(state, 'current_tab').classes('h-full text-gray-400 font-semibold').props('active-color=white indicator-color=primary') as tabs:
            tab_search = ui.tab('Search', label='Умный Поиск', icon='search')
            tab_aesthetic = ui.tab('Aesthetic', label='Оценка Эстетики', icon='star')
            tab_nsfw = ui.tab('NSFW', label='NSFW Детектор', icon='visibility_off')
            tab_face = ui.tab('Face', label='Поиск по лицу', icon='face')
            tab_tags = ui.tab('Tags', label='Danbooru Теги', icon='label')
            tab_dupes = ui.tab('Dupes', label='Дубликаты', icon='content_copy')
            tab_cluster = ui.tab('Cluster', label='AI Сортировка', icon='auto_awesome_mosaic')
            tab_cache = ui.tab('Cache', label='Индексатор', icon='storage')
            
        ui.button(icon='settings', on_click=lambda: global_settings_dialog.open()).props('flat round dense text-color=white').classes('shrink-0').tooltip('Глобальные настройки')
        
    with ui.dialog() as global_settings_dialog:
        with ui.card().classes('w-[500px] max-w-full bg-gray-900 text-white border border-gray-700'):
            ui.label('Глобальные настройки').classes('text-xl font-bold mb-2 text-blue-400')
            
            ui.number('Порог опасности NSFW (0.0 - 1.0)', value=state.nsfw_threshold, min=0.0, max=1.0, step=0.01, format='%.2f').bind_value(state, 'nsfw_threshold').classes('w-full')
            ui.number('Колонок в сетке (чем больше - тем меньше плитки)', value=state.grid_columns, min=1, max=12, format='%d').bind_value(state, 'grid_columns').classes('w-full mt-2')
            ui.number('Групп на странице (Дубликаты/Сортировка)', value=state.groups_per_page, min=1, max=20, format='%d').bind_value(state, 'groups_per_page').classes('w-full mt-2')
            ui.checkbox('Копировать/Перемещать без структуры папок (в одну директорию)', value=state.flatten_structure).bind_value(state, 'flatten_structure').classes('w-full mt-2')
            
            # --- УПРАВЛЕНИЕ КЭШЕМ ---
            ui.label('Управление базой данных и кэшем').classes('text-lg font-bold mt-6 mb-2 text-red-400')
            
            with ui.row().classes('w-full gap-2 items-center'):
                model_to_clear = ui.select(['Все модели'], value='Все модели', label='Выберите модель для очистки').classes('flex-grow')
                
                def clear_selected_model():
                    if model_to_clear.value == 'Все модели':
                        search_engine.db_cache.clear_model_cache(None)
                        ui.notify("База данных ПОЛНОСТЬЮ очищена и сжата!", type="positive")
                    else:
                        search_engine.db_cache.clear_model_cache(model_to_clear.value)
                        ui.notify(f"Кэш модели {model_to_clear.value} очищен!", type="positive")
                    refresh_models_list()

                ui.button(icon='delete_forever', on_click=clear_selected_model).props('color=red').tooltip('Очистить БД для выбранной модели')

            def refresh_models_list():
                models = search_engine.db_cache.get_all_models()
                model_to_clear.options = ['Все модели'] + models
                model_to_clear.value = 'Все модели'
                model_to_clear.update()

            global_settings_dialog.on('show', refresh_models_list)

            # --- БЛОК ТОЧЕЧНОЙ ОЧИСТКИ ---
            ui.label('Точечная очистка (по категориям)').classes('text-md font-bold mt-6 mb-2 text-orange-400')
            with ui.grid(columns=2).classes('w-full gap-2'):
                def _clear_cache(ctype, name):
                    search_engine.db_cache.clear_specific_cache(ctype)
                    ui.notify(f"Кэш '{name}' успешно очищен!", type='positive')
                
                ui.button('Дубликаты (pHash)', on_click=lambda: _clear_cache('phash', 'Дубликатов')).props('outline color=orange').classes('w-full')
                ui.button('Теги (Danbooru)', on_click=lambda: _clear_cache('tags', 'Тегов')).props('outline color=pink').classes('w-full')
                ui.button('NSFW и Эстетика', on_click=lambda: _clear_cache('nsfw_aes', 'NSFW/Эстетики')).props('outline color=red').classes('w-full')
                ui.button('История поиска', on_click=lambda: _clear_cache('search_history', 'Истории поиска')).props('outline color=blue').classes('w-full').tooltip('Удаляет текстовые запросы (сами ИИ-векторы файлов останутся)')
                ui.button('Лица (InsightFace)', on_click=lambda: _clear_cache('faces', 'Лиц')).props('outline color=teal').classes('w-full')
                ui.label('Освобождает место в БД. После очистки нажмите "Сжать базу", чтобы файл .db уменьшился на диске.').classes('text-[10px] text-gray-500 col-span-2 leading-tight')

            async def cleanup_dead_links():
                ui.notify("Ищем удаленные файлы... Это может занять время", type="info")
                def task():
                    paths = search_engine.db_cache.get_all_paths()
                    dead =[p for p in paths if not os.path.exists(p)]
                    if dead:
                        search_engine.db_cache.remove_paths(dead)
                        search_engine.db_cache.conn.execute("VACUUM")
                    return len(dead)
                try:
                    dead_count = await run.io_bound(task)
                    if dead_count > 0:
                        ui.notify(f"Удалено {dead_count} мертвых записей из базы!", type="positive")
                    else:
                        ui.notify("Мертвых записей не найдено, база в порядке.", type="positive")
                except Exception as e:
                    ui.notify(f"Ошибка: {e}", type="negative")

            def cleanup_thumbnails():
                count = 0
                for f in os.listdir(THUMB_CACHE_DIR):
                    try:
                        os.remove(os.path.join(THUMB_CACHE_DIR, f))
                        count += 1
                    except: pass
                ui.notify(f"Удалено {count} миниатюр", type="positive")

            def cleanup_file_index():
                search_engine.files_cache._data.clear()
                search_engine.files_cache.save_cache()
                ui.notify("Индекс файлов (кэш путей) сброшен", type="positive")

            async def vacuum_database():
                ui.notify("Начато сжатие базы данных и очистка WAL... Это может занять время.", type="warning")
                def task():
                    db_path = 'image_cache.db'
                    wal_path = db_path + '-wal'
                    
                    # 1. Замеряем исходный размер (DB + WAL)
                    start_db = os.path.getsize(db_path) / (1024*1024) if os.path.exists(db_path) else 0
                    start_wal = os.path.getsize(wal_path) / (1024*1024) if os.path.exists(wal_path) else 0
                    start_total = start_db + start_wal

                    # 2. Выполняем сжатие
                    search_engine.db_cache.conn.execute("VACUUM")
                    
                    # 3. ПРИНУДИТЕЛЬНО очищаем и обрезаем файл WAL до 0 байт!
                    search_engine.db_cache.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    
                    # 4. Замеряем итоговый размер
                    end_db = os.path.getsize(db_path) / (1024*1024) if os.path.exists(db_path) else 0
                    end_wal = os.path.getsize(wal_path) / (1024*1024) if os.path.exists(wal_path) else 0
                    end_total = end_db + end_wal
                    
                    return start_total, end_total
                    
                try:
                    start_sz, end_sz = await run.io_bound(task)
                    ui.notify(f"База успешно сжата! Было (с WAL): {start_sz:.1f} МБ -> Стало: {end_sz:.1f} МБ", type="positive", timeout=8000)
                except Exception as e:
                    ui.notify(f"Ошибка: {e}", type="negative")
                
            with ui.column().classes('w-full gap-2 mt-4'):
                ui.button('Удалить "Мертвые души" (Файлы, которых больше нет на диске)', on_click=cleanup_dead_links).props('outline color=orange').classes('w-full')
                ui.button('Сжать базу данных (VACUUM)', on_click=vacuum_database).props('outline color=blue').classes('w-full').tooltip('Уменьшает размер файла .db на диске после удаления данных')
                with ui.row().classes('w-full gap-2'):
                    ui.button('Очистить миниатюры', on_click=cleanup_thumbnails).props('outline color=gray').classes('flex-grow')
                    ui.button('Сбросить индекс папок', on_click=cleanup_file_index).props('outline color=gray').classes('flex-grow')

            def save_global_settings():
                state.grid_columns = int(state.grid_columns)
                state.groups_per_page = int(state.groups_per_page)
                save_config({
                    'nsfw_threshold': state.nsfw_threshold,
                    'flatten_structure': state.flatten_structure,
                    'grid_columns': state.grid_columns,
                    'groups_per_page': state.groups_per_page,
                })
                ui.notify('Глобальные настройки сохранены', type='positive')
                search_gallery_ui.refresh()
                aesthetic_gallery_ui.refresh()
                nsfw_gallery_ui.refresh()
                face_gallery_ui.refresh()
                tags_gallery_ui.refresh()
                dupes_gallery_ui.refresh()
                cluster_gallery_ui.refresh()
                global_settings_dialog.close()
                
            ui.button('Сохранить и закрыть', on_click=save_global_settings).classes('w-full mt-6 bg-blue-600 hover:bg-blue-500 font-bold')

    with ui.right_drawer(value=False).props('width=550').classes('bg-gray-900 border-l border-gray-800 p-4 z-50 flex flex-col') as log_drawer:
        with ui.row().classes('w-full flex justify-between items-center mb-2 shrink-0'):
            ui.label('Системные Логи').classes('text-lg font-bold text-white')
            with ui.row().classes('gap-2'):
                ui.button(icon='content_copy', on_click=copy_logs).props('flat round dense text-color=gray').tooltip('Скопировать все логи')
                ui.button(icon='delete_sweep', on_click=clear_logs).props('flat round dense text-color=red').tooltip('Очистить окно логов')
        
        global ui_log_element
        ui_log_element = ui.log().classes('w-full flex-grow bg-black text-green-400 font-mono text-xs p-2 rounded overflow-y-auto whitespace-pre-wrap break-words')

    # --- ДИАЛОГ ДЕБАГА NSFW ---
    with ui.dialog() as nsfw_debug_dialog:
        with ui.card().classes('w-[500px] max-w-full bg-gray-900 text-white border border-gray-700'):
            debug_title = ui.label('Детали NSFW').classes('text-lg font-bold mb-2 break-all')
            debug_container = ui.column().classes('w-full gap-1 max-h-[60vh] overflow-y-auto')
            ui.button('Закрыть', on_click=nsfw_debug_dialog.close).classes('w-full mt-4 bg-gray-800 hover:bg-gray-700')

    def show_nsfw_debug(path, details):
        debug_title.set_text(os.path.basename(path))
        debug_container.clear()
        safe_set = {'safe', 'sfw', 'normal', 'general', 'neutral', 'drawing', 'safe_content', 'anime picture', 'anime'}
        with debug_container:
            sorted_details = sorted(details.items(), key=lambda x: x[1], reverse=True)
            for lbl, prob in sorted_details:
                color = "text-red-400 font-bold" if prob > 0.1 and lbl.lower() not in safe_set else "text-green-400" if lbl.lower() in safe_set else "text-gray-400"
                with ui.row().classes('w-full justify-between border-b border-gray-800 py-1 px-2'):
                    ui.label(lbl).classes(f'font-mono text-sm {color}')
                    ui.label(f"{prob*100:.2f}%").classes(f'font-mono text-sm {color}')
        nsfw_debug_dialog.open()

    with ui.dialog() as tags_debug_dialog:
        with ui.card().classes('w-[500px] max-w-full bg-gray-900 text-white border border-gray-700'):
            tags_debug_title = ui.label('Теги (Danbooru)').classes('text-lg font-bold mb-2 break-all')
            tags_debug_container = ui.column().classes('w-full gap-1 max-h-[60vh] overflow-y-auto')
            ui.button('Закрыть', on_click=tags_debug_dialog.close).classes('w-full mt-4 bg-gray-800 hover:bg-gray-700')

    def show_tags_debug(path, tags_dict):
        tags_debug_title.set_text(os.path.basename(path))
        tags_debug_container.clear()
        with tags_debug_container:
            sorted_tags = sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)
            for lbl, prob in sorted_tags:
                with ui.row().classes('w-full justify-between border-b border-gray-800 py-1 px-2'):
                    ui.label(lbl).classes('font-mono text-sm text-pink-300')
                    ui.label(f"{prob*100:.2f}%").classes('font-mono text-sm text-gray-400')
        tags_debug_dialog.open()

    # --- ПОЛНОЭКРАННЫЙ ПЛЕЕР ---
    def sync_gallery_page():
        if not state.viewer_items: return
        
        # Для кластеров и дубликатов логика плеера работает иначе (внутри группы).
        # Поэтому мы отключаем синхронизацию главной страницы для этих вкладок.
        if state.current_tab in ['Cluster', 'Dupes']:
            return

        target_page = (state.viewer_index // ITEMS_PER_PAGE) + 1
        changed = False
        scroll_id = ""

        # Проверяем, изменилась ли страница, и обновляем интерфейс
        if state.current_tab == 'Search' and state.search_page != target_page:
            state.search_page = target_page
            search_gallery_ui.refresh()
            changed = True
            scroll_id = "search_scroll_area"
        elif state.current_tab == 'Aesthetic' and state.aes_page != target_page:
            state.aes_page = target_page
            aesthetic_gallery_ui.refresh()
            changed = True
            scroll_id = "aes_scroll_area"
        elif state.current_tab == 'NSFW' and state.nsfw_page != target_page:
            state.nsfw_page = target_page
            nsfw_gallery_ui.refresh()
            changed = True
            scroll_id = "nsfw_scroll_area"
        elif state.current_tab == 'Face' and state.face_page != target_page:
            state.face_page = target_page
            face_gallery_ui.refresh()
            changed = True
            scroll_id = "face_scroll_area"
        elif state.current_tab == 'Tags' and state.tags_page != target_page:
            state.tags_page = target_page
            tags_gallery_ui.refresh()
            changed = True
            scroll_id = "tags_scroll_area"

        # Если страница изменилась, прокручиваем список в самое начало
        if changed:
            ui.run_javascript(f'setTimeout(() => {{ let el = document.getElementById("{scroll_id}"); if(el) el.scrollTo({{top: 0, behavior: "instant"}}); }}, 100);')

    # Привязываем функцию sync_gallery_page к событию hide (закрытие диалога)
    with ui.dialog().on('value', lambda e: setattr(state, 'viewer_open', e.value)).on('hide', sync_gallery_page).props('maximized transition-show=fade transition-hide=fade') as media_dialog:
        with ui.element('div') \
            .classes('w-full h-full bg-black/95 p-0 flex flex-col relative items-center justify-center overflow-hidden') \
            .on('wheel.prevent', lambda e: change_media(1 if e.args['deltaY'] > 0 else -1),['deltaY']) \
            .on('click.self', media_dialog.close):
            
            ui.button(icon='close', on_click=media_dialog.close).classes('absolute top-4 right-4 z-50 bg-white/10 hover:bg-white/20 text-white').props('flat round')
            
            ui.button(icon='chevron_left', on_click=lambda: change_media(-1)).classes('absolute left-4 top-1/2 -translate-y-1/2 z-50 bg-white/10 hover:bg-white/20 text-white text-4xl p-2').props('flat round').tooltip('Предыдущий (←)')
            ui.button(icon='chevron_right', on_click=lambda: change_media(1)).classes('absolute right-4 top-1/2 -translate-y-1/2 z-50 bg-white/10 hover:bg-white/20 text-white text-4xl p-2').props('flat round').tooltip('Следующий (→)')
            
            # Отступ p-8 не даст картинке залезть под боковые кнопки
            media_container = ui.element('div') \
                .classes('absolute inset-0 w-full h-full flex items-center justify-center z-0 p-8') \
                .on('click.self', media_dialog.close)
            
            with ui.row().classes('absolute bottom-6 left-1/2 -translate-x-1/2 bg-black/80 border border-gray-700 px-4 py-2 rounded-full text-white flex-nowrap items-center gap-3 z-50 shadow-lg'):
                btn_viewer_select = ui.button(on_click=lambda: toggle_selection()).props('flat round dense size=sm').tooltip('Выделить (Space)')
                lbl_media_name = ui.label().classes('font-mono text-xs text-center whitespace-nowrap overflow-hidden text-ellipsis min-w-[150px] max-w-[400px] px-2')
                ui.button(icon='content_copy', on_click=lambda: copy_image_to_clipboard(state.viewer_items[state.viewer_index])).props('flat round dense size=sm color=white').tooltip('Копировать картинку в буфер (C)')
                ui.button(icon='download', on_click=lambda: download_current_item()).props('flat round dense size=sm color=white').tooltip('Сохранить в Downloads (D)')

    def update_viewer_selection_ui():
        if not state.viewer_items: return
        path = state.viewer_items[state.viewer_index]
        is_selected = False
        if state.current_tab == 'Search' and path in state.sel_search: is_selected = state.sel_search[path]
        elif state.current_tab == 'Aesthetic' and path in state.sel_aes: is_selected = state.sel_aes[path]
        elif state.current_tab == 'NSFW' and path in state.sel_nsfw: is_selected = state.sel_nsfw[path]
        elif state.current_tab == 'Face' and path in state.sel_face: is_selected = state.sel_face[path]
        elif state.current_tab == 'Tags' and path in state.sel_tags: is_selected = state.sel_tags[path]
        elif state.current_tab == 'Cluster' and path in state.sel_cluster: is_selected = state.sel_cluster[path]
            
        btn_viewer_select._props['icon'] = 'check_box' if is_selected else 'check_box_outline_blank'
        btn_viewer_select._props['color'] = 'green' if is_selected else 'white'
        btn_viewer_select.update()

    def toggle_selection():
        if not state.viewer_items: return
        path = state.viewer_items[state.viewer_index]
        if state.current_tab == 'Search' and path in state.sel_search:
            state.sel_search[path] = not state.sel_search[path]
        elif state.current_tab == 'Aesthetic' and path in state.sel_aes:
            state.sel_aes[path] = not state.sel_aes[path]
        elif state.current_tab == 'NSFW' and path in state.sel_nsfw:
            state.sel_nsfw[path] = not state.sel_nsfw[path]
        elif state.current_tab == 'Face' and path in state.sel_face:
            state.sel_face[path] = not state.sel_face[path]
        elif state.current_tab == 'Tags' and path in state.sel_tags:
            state.sel_tags[path] = not state.sel_tags[path]
        elif state.current_tab == 'Cluster' and path in state.sel_cluster:
            state.sel_cluster[path] = not state.sel_cluster[path]
        update_viewer_selection_ui()

    def download_current_item():
        if not state.viewer_items: return
        path = state.viewer_items[state.viewer_index]
        tab = state.current_tab.lower()
        base_dir = getattr(state, f"{tab}_base_dir", state.search_base_dir)
        try:
            dl_dir = os.path.join(str(Path.home()), 'Downloads')
            try:
                rel_path = os.path.relpath(path, base_dir)
                if rel_path.startswith('..') or os.path.isabs(rel_path): rel_path = os.path.basename(path)
            except: rel_path = os.path.basename(path)
                
            dest = os.path.join(dl_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            
            if os.path.exists(dest):
                base, ext = os.path.splitext(os.path.basename(dest))
                dest = os.path.join(os.path.dirname(dest), f"{base}_{int(time.time())}{ext}")
                
            shutil.copy2(path, dest)
            ui.notify(f'Сохранено: {rel_path}', type='positive', position='bottom-right', timeout=1500, group='downloads')
        except Exception as e:
            ui.notify(f'Ошибка: {e}', type='negative', position='bottom-right', timeout=2000, group='err_downloads')
        
    def render_viewer():
        media_container.clear()
        if not state.viewer_items: return
        path = state.viewer_items[state.viewer_index]
        safe_path = urllib.parse.quote(path)
        ext = os.path.splitext(path)[1].lower()
        lbl_media_name.set_text(f"{state.viewer_index + 1} / {len(state.viewer_items)} — {os.path.basename(path)}")
        update_viewer_selection_ui()
        
        with media_container:
            if ext in SUPPORTED_VIDEOS:
                ui.video(f'/media/{safe_path}') \
                    .classes('outline-none') \
                    .style('max-width: 100%; max-height: 90vh; width: auto; height: auto; object-fit: contain;') \
                    .props('autoplay controls loop')
            elif ext in SUPPORTED_IMAGES:
                ui.element('img').props(f'src="/media/{safe_path}"') \
                    .classes('outline-none cursor-pointer') \
                    .style('max-width: 100%; max-height: 90vh; width: auto; height: auto; object-fit: contain;') \
                    .on('click', media_dialog.close)
            else:
                ui.icon('article', size='15rem').classes('text-gray-500 cursor-pointer').on('click', media_dialog.close)

    def open_media(index, items):
        state.viewer_index = index
        state.viewer_items = items
        state.viewer_open = True
        render_viewer()
        media_dialog.open()

    def change_media(delta):
        new_idx = state.viewer_index + delta
        if 0 <= new_idx < len(state.viewer_items):
            state.viewer_index = new_idx
            render_viewer()

    def handle_keyboard(e):
        if not e.action.keydown or not state.viewer_open: return
        if e.key.arrow_right: change_media(1)
        elif e.key.arrow_left: change_media(-1)
        elif e.key.space: toggle_selection()
        elif e.key.name and e.key.name.lower() == 'd': download_current_item()
        elif e.key.name and e.key.name.lower() == 'c': copy_image_to_clipboard(state.viewer_items[state.viewer_index])
        elif e.key.name == 'Delete': 
            path = state.viewer_items[state.viewer_index]
            tab = state.current_tab.lower()
            if tab == 'search': tab_name = 'search'
            elif tab == 'aesthetic': tab_name = 'aes'
            elif tab == 'nsfw': tab_name = 'nsfw'
            elif tab == 'face': tab_name = 'face'
            elif tab == 'tags': tab_name = 'tags'
            elif tab == 'dupes': tab_name = 'dupes'
            elif tab == 'cluster': tab_name = 'cluster'
            
            # Закрываем плеер, если удалили последний файл
            if len(state.viewer_items) <= 1:
                media_dialog.close()
            else:
                change_media(1 if state.viewer_index < len(state.viewer_items)-1 else -1)
            
            delete_items([path], tab_name)

    ui.keyboard(on_key=handle_keyboard, ignore=['input', 'textarea', 'select'])

    # --- ЭКСПОРТ HTML ---
    async def export_html_action(tab='search'):
        folder = await run.io_bound(pick_folder_native)
        if not folder: return
        
        html_path = os.path.join(folder, f"gallery_{tab}.html")
        ui.notify("Создание HTML галереи...", type='info')
        
        items = getattr(state, f"{tab}_results",[])
        filter_val = getattr(state, f"{tab}_res_filter", 'Все')
        
        def _export():
            html_content =[
                "<html><body style='background-color:#1e1e1e; color:white; font-family:sans-serif;'>",
                f"<h2>Экспорт результатов</h2>",
                "<div style='display:flex; flex-wrap:wrap; gap:15px;'>"
            ]
            
            for item in items:
                path = item[1]
                
                if filter_val == 'Картинки' and not path.lower().endswith(SUPPORTED_IMAGES): continue
                if filter_val == 'Видео' and not path.lower().endswith(SUPPORTED_VIDEOS): continue
                
                if tab == 'search': label_text = f"Score: {item[0]:.3f}"
                elif tab == 'aes': label_text = f"★ {item[0]:.2f} (Пик: {item[2]:.2f})"
                elif tab == 'nsfw': label_text = f"🚨 Danger: {item[0]*100:.1f}% | {item[2].upper()}"
                elif tab == 'face': label_text = f"Match: {item[0]*100:.1f}%"
                elif tab == 'tags': label_text = f"Tags Score: {item[0]:.2f}"
                    
                uri = Path(path).absolute().as_uri()
                ext = os.path.splitext(path)[1].lower()
                
                if ext in SUPPORTED_VIDEOS:
                    path_hash = hashlib.md5(path.encode('utf-8')).hexdigest()
                    thumb_path = os.path.join(THUMB_CACHE_DIR, f"{path_hash}.jpg")
                    if not os.path.exists(thumb_path):
                        try:
                            with av.open(path) as container:
                                for frame in container.decode(video=0):
                                    img = frame.to_image()
                                    img.thumbnail((300, 300))
                                    img.convert('RGB').save(thumb_path, format="JPEG", quality=80)
                                    break
                        except: pass
                    
                    thumb_uri = Path(thumb_path).absolute().as_uri() if os.path.exists(thumb_path) else uri
                    
                    html_content.append(
                        f"<div style='background:#2d2d2d; padding:10px; border-radius:8px; text-align:center; max-width:320px;'>"
                        f"<a href='{uri}' target='_blank' title='Кликните, чтобы открыть видео'>"
                        f"<div style='position:relative; width:300px; height:200px; background:#111; border-radius:4px; display:flex; align-items:center; justify-content:center; overflow:hidden;'>"
                        f"<img src='{thumb_uri}' style='max-width:100%; max-height:100%; object-fit:contain;'>"
                        f"<div style='position:absolute; top:5px; right:5px; background:rgba(0,0,0,0.7); padding:3px 6px; border-radius:4px; font-size:12px;'>▶ Video</div>"
                        f"</div></a>"
                        f"<h4 style='margin:10px 0 5px 0; color:#4caf50;'>{label_text}</h4>"
                        f"<div style='font-size:11px; color:#aaa; word-wrap:break-word;'>{os.path.basename(path)}</div></div>"
                    )
                else:
                    html_content.append(
                        f"<div style='background:#2d2d2d; padding:10px; border-radius:8px; text-align:center; max-width:320px;'>"
                        f"<a href='{uri}' target='_blank'>"
                        f"<div style='width:300px; height:200px; background:#111; border-radius:4px; display:flex; align-items:center; justify-content:center; overflow:hidden;'>"
                        f"<img src='{uri}' style='max-width:100%; max-height:100%; object-fit:contain;'></div></a>"
                        f"<h4 style='margin:10px 0 5px 0; color:#4caf50;'>{label_text}</h4>"
                        f"<div style='font-size:11px; color:#aaa; word-wrap:break-word;'>{os.path.basename(path)}</div></div>"
                    )
            html_content.append("</div></body></html>")
            
            try:
                with open(html_path, "w", encoding="utf-8") as f: f.write("\n".join(html_content))
                return True
            except Exception as e:
                return e

        res = await run.io_bound(_export)
        if res is True:
            ui.notify(f"Галерея сохранена: {html_path}", type='positive')
        else:
            ui.notify(f"Ошибка экспорта: {res}", type='negative')

    # --- БЕЗОПАСНОЕ ОБНОВЛЕНИЕ UI ---
    def refresh_tab_ui(tab_name):
        if tab_name == 'search': search_gallery_ui.refresh()
        elif tab_name == 'aes': aesthetic_gallery_ui.refresh()
        elif tab_name == 'nsfw': nsfw_gallery_ui.refresh()
        elif tab_name == 'face': face_gallery_ui.refresh()
        elif tab_name == 'tags': tags_gallery_ui.refresh()
        elif tab_name == 'dupes': dupes_gallery_ui.refresh()
        elif tab_name == 'cluster': cluster_gallery_ui.refresh()

    def get_physical_info(p):
        try:
            size_mb = os.path.getsize(p) / (1024 * 1024)
            w, h_dim = 0, 0
            ext = os.path.splitext(p)[1].lower()
            if ext in SUPPORTED_IMAGES:
                with Image.open(p) as img:
                    w, h_dim = img.size
            elif ext in SUPPORTED_VIDEOS:
                with av.open(p) as container:
                    stream = container.streams.video[0]
                    w, h_dim = stream.width, stream.height
            return size_mb, w, h_dim
        except Exception:
            return 0.0, 0, 0

    async def apply_physical_filters_async(results_list):
        if not results_list: return[]
        
        min_r = state.filter_min_res
        max_r = state.filter_max_res
        max_s = state.filter_max_size
        orient = state.filter_orientation

        # Если фильтры по умолчанию, мы можем сразу вернуть исходный список без обращений к БД или диску!
        if min_r <= 0 and max_r >= 10000 and max_s >= 10000 and orient == 'Любая':
            return results_list

        def _filter_task():
            c = search_engine.db_cache.conn.cursor()
            paths = [item[1] for item in results_list]
            file_infos = {}
            
            # Запрашиваем из базы только те пути, которые реально попали в поиск
            chunk_size = 900
            for i in range(0, len(paths), chunk_size):
                chunk = paths[i:i+chunk_size]
                ph = ','.join(['?']*len(chunk))
                c.execute(f"SELECT path, size_mb, width, height FROM files WHERE path IN ({ph})", chunk)
                for row in c.fetchall():
                    file_infos[row[0]] = (row[1], row[2], row[3])
            
            filtered =[]
            db_needs_commit = False
            
            for item in results_list:
                p = item[1]
                info = file_infos.get(p)
                
                # Если в базе нет размеров (старый кэш или файл не индексировался), то открываем файл
                if not info or info[1] is None or info[2] is None:
                    size_mb, w, h = get_physical_info(p)
                    hash_val = search_engine.db_cache.get_hash_by_path(p) or get_fast_hash(p)
                    c.execute("INSERT OR REPLACE INTO files (hash, path, size_mb, width, height) VALUES (?, ?, ?, ?, ?)", (hash_val, p, size_mb, w, h))
                    db_needs_commit = True
                else:
                    size_mb, w, h = info
                    
                if size_mb is not None and size_mb > max_s: continue
                
                max_dim = max(w, h) if w and h else 0
                if max_dim > 0:
                    if max_dim < min_r or max_dim > max_r: continue
                    if orient != 'Любая':
                        if orient == 'Горизонтальная' and w <= h * 1.05: continue
                        if orient == 'Вертикальная' and h <= w * 1.05: continue
                        if orient == 'Квадрат' and (w > h * 1.05 or h > w * 1.05): continue
                
                filtered.append(item)
                
            if db_needs_commit:
                search_engine.db_cache.conn.commit()
            return filtered

        return await run.io_bound(_filter_task)

    async def delete_items(paths, tab_name):
        if not paths: return
        attr_name = "aesthetic_results" if tab_name == "aes" else f"{tab_name}_results"
        
        # Выносим физическое удаление файлов и работу с БД в отдельный поток
        def _trash_files():
            deleted_count = 0
            for p in paths:
                try:
                    send2trash(os.path.normpath(p))
                    deleted_count += 1
                    search_engine.db_cache.remove_paths([p])
                except Exception as e:
                    state.add_log(f"Ошибка удаления {p}: {e}")
            return deleted_count

        ui.notify("Удаление файлов...", type='info')
        deleted = await run.io_bound(_trash_files)
        
        # Обновление списков в состоянии делаем в основном потоке
        res_list = getattr(state, attr_name)
        if tab_name == 'dupes':
            new_dupes = []
            for g in res_list:
                new_g =[item for item in g if item not in paths]
                if len(new_g) > 1: new_dupes.append(new_g)
            setattr(state, attr_name, new_dupes)
        elif tab_name == 'cluster':
            new_clusters =[]
            for c in res_list:
                new_paths = [item for item in c["paths"] if item not in paths]
                if len(new_paths) > 0: new_clusters.append({"name": c["name"], "paths": new_paths})
            setattr(state, attr_name, new_clusters)
        else:
            setattr(state, attr_name, [item for item in res_list if item[1] not in paths])
            
        ui.notify(f"🗑️ Отправлено в корзину: {deleted} шт.", type='positive', color='red')
        sel_dict = getattr(state, f"sel_{tab_name}")
        for p in paths:
            if p in sel_dict: del sel_dict[p]
            
        refresh_tab_ui(tab_name)

    # --- ПАКЕТНЫЕ ДЕЙСТВИЯ ---
    async def execute_batch(action='copy', tab='search', prepend_score=False, export_txt=False, txt_threshold=0.1):
        sel_dict = getattr(state, f"sel_{tab}")
        selected_paths =[p for p, checked in sel_dict.items() if checked]
        if not selected_paths:
            return ui.notify('Ничего не выбрано!', type='warning')
            
        folder = await run.io_bound(pick_folder_native)
        if not folder: return
        
        base_dir = getattr(state, f"{tab}_base_dir", state.search_base_dir)
        ui.notify(f"Начато {action} {len(selected_paths)} файлов...", type='info')

        # Фоновая задача для файловых операций
        def _process_files():
            success = 0
            moved_paths = set()
            for path in selected_paths:
                try:
                    rel_path = os.path.relpath(path, base_dir)
                    if rel_path.startswith('..') or os.path.isabs(rel_path): rel_path = os.path.basename(path)
                except Exception: rel_path = os.path.basename(path)
                    
                rel_dir, fname = os.path.split(rel_path)
                if state.flatten_structure:
                    rel_dir = ""

                prefix = ""
                if prepend_score:
                    if tab == 'search': prefix = f"{next((s for s, p in state.search_results if p == path), 0):.3f}_"
                    elif tab == 'aes': prefix = f"{next((a for a, p, m in state.aesthetic_results if p == path), 0):05.2f}_"
                    elif tab == 'nsfw': prefix = f"{next((d for d, p, l, dt in state.nsfw_results if p == path), 0)*100:05.1f}_"
                    elif tab == 'face': prefix = f"{next((s for s, p in state.face_results if p == path), 0)*100:05.1f}_"
                        
                dest = os.path.join(folder, rel_dir, prefix + fname)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                
                try:
                    if action == 'copy': shutil.copy2(path, dest)
                    else: 
                        shutil.move(path, dest)
                        moved_paths.add(path)
                    
                    if export_txt and tab == 'tags':
                        txt_dest = os.path.splitext(dest)[0] + '.txt'
                        item_data = next((i for i in state.tags_results if i[1] == path), None)
                        if item_data and len(item_data) > 2:
                            tags_dict = item_data[2]
                            valid_tags =[t for t, s in tags_dict.items() if s >= txt_threshold]
                            if valid_tags:
                                with open(txt_dest, 'w', encoding='utf-8') as f:
                                    f.write(", ".join(valid_tags))
                    success += 1
                except Exception as e: state.add_log(f"Ошибка {path}: {e}")
            return success, moved_paths

        # Ждем выполнения без блокировки UI
        success, moved_paths = await run.io_bound(_process_files)
                
        ui.notify(f'Успешно {action}: {success} файлов', type='positive')
        
        if action == 'move' and moved_paths:
            setattr(state, f"{tab}_results",[i for i in getattr(state, f"{tab}_results") if i[1] not in moved_paths])
            refresh_tab_ui(tab)

    async def handle_shift_click(e, idx, path, tab, custom_paths=None):
        is_shift = isinstance(e.args, dict) and e.args.get('shiftKey', False)
        await asyncio.sleep(0.05) 
        
        sel_dict = getattr(state, f"sel_{tab}")
        
        # Если передали кастомный список (для вкладок со сложной структурой вроде dupes)
        all_p = custom_paths if custom_paths is not None else[p for i in getattr(state, f"{tab}_results") for p in[i[1]]]

        last_idx = getattr(state, f'last_clicked_{tab}', None)

        if not is_shift:
            setattr(state, f'last_clicked_{tab}', idx)
        else:
            if last_idx is not None and last_idx < len(all_p) and idx < len(all_p):
                start = min(idx, last_idx)
                end = max(idx, last_idx)
                target_val = sel_dict.get(path, True)
                for i in range(start, end + 1):
                    sel_dict[all_p[i]] = target_val

    async def set_all(tab, value):
        # Даем UI перерисоваться перед лагом
        await asyncio.sleep(0.01) 
        
        if tab == 'dupes':
            for g in state.dupes_results:
                for p in g: state.sel_dupes[p] = value
            dupes_gallery_ui.refresh()
            return 
        if tab == 'cluster':
            for c in state.cluster_results:
                for p in c["paths"]: state.sel_cluster[p] = value
            cluster_gallery_ui.refresh()
            return
            
        filter_val = getattr(state, f"{tab}_res_filter")
        sel_dict = getattr(state, f"sel_{tab}")
        
        # Если элементов слишком много, можно использовать io_bound, но обычно dict comprehension работает мгновенно. 
        # Главное - освободить поток перед рефрешем
        for item in getattr(state, f"{tab}_results"):
            p = item[1]
            if filter_val == 'Картинки' and not p.lower().endswith(SUPPORTED_IMAGES): continue
            if filter_val == 'Видео' and not p.lower().endswith(SUPPORTED_VIDEOS): continue
            if p in sel_dict: sel_dict[p] = value
            
        # Обновляем UI
        if tab == 'search': search_gallery_ui.refresh()
        elif tab == 'aes': aesthetic_gallery_ui.refresh()
        elif tab == 'nsfw': nsfw_gallery_ui.refresh()
        elif tab == 'face': face_gallery_ui.refresh()
        elif tab == 'tags': tags_gallery_ui.refresh()

    # --- КОМПОНЕНТЫ ГАЛЕРЕИ ---
    @ui.refreshable
    async def search_gallery_ui():
        if not state.search_results:
            ui.label("Здесь появятся результаты...").classes("text-gray-400 m-4")
            return

        await asyncio.sleep(0.001)

        filtered_results =[]
        for item in state.search_results:
            p = item[1].lower()
            if state.search_res_filter == 'Картинки' and not p.endswith(SUPPORTED_IMAGES): continue
            if state.search_res_filter == 'Видео' and not p.endswith(SUPPORTED_VIDEOS): continue
            filtered_results.append(item)
            
        filtered_results = await apply_physical_filters_async(filtered_results)

        total_pages = max(1, (len(filtered_results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        if state.search_page > total_pages: state.search_page = 1

        def change_page(d):
            state.search_page = max(1, min(total_pages, state.search_page + d))
            search_gallery_ui.refresh()

        def apply_filter(e):
            state.search_res_filter = e.value
            state.search_page = 1
            search_gallery_ui.refresh()

        with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
            with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('Выбрать всё', on_click=lambda: ui.timer(0, lambda: set_all('search', True), once=True)).props('outline color=white dense')
                        ui.button('Снять всё', on_click=lambda: ui.timer(0, lambda: set_all('search', False), once=True)).props('outline color=white dense')
                        ui.toggle(['Все', 'Картинки', 'Видео'], value=state.search_res_filter, on_change=apply_filter).classes('text-xs ml-2')
                        ui.button(icon='filter_alt', on_click=lambda: (setattr(state, 'show_phys_filters', not getattr(state, 'show_phys_filters', False)), search_gallery_ui.refresh())).props('flat color=gray dense').tooltip('Доп. фильтры')
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('HTML Экспорт', icon='html', on_click=lambda: export_html_action('search')).props('color=purple dense outline')
                        ui.button('Копировать ✔', icon='content_copy', on_click=lambda: execute_batch('copy', 'search', chk_prefix_search.value)).props('color=blue dense')
                        ui.button('Переместить ✔', icon='drive_file_move', on_click=lambda: execute_batch('move', 'search', chk_prefix_search.value)).props('color=red dense')
                        ui.button('УДАЛИТЬ ✔', icon='delete_forever', on_click=lambda: delete_items([p for p, c in state.sel_search.items() if c], 'search')).props('color=red-10 text-white dense')
                
                if getattr(state, 'show_phys_filters', False):
                    with ui.row().classes('w-full bg-gray-800/50 p-2 rounded-lg mb-2 items-end gap-4 border border-gray-700'):
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Разрешение (Max сторона, px):').classes('text-xs text-gray-400')
                            with ui.row().classes('w-full gap-2 flex-nowrap'):
                                ui.number('Мин', value=0, format='%.0f').bind_value(state, 'filter_min_res').classes('flex-1')
                                ui.number('Макс', value=10000, format='%.0f').bind_value(state, 'filter_max_res').classes('flex-1')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Макс. вес:').classes('text-xs text-gray-400')
                            ui.number('МБ', value=10000, format='%.0f').bind_value(state, 'filter_max_size').classes('w-full')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Ориентация:').classes('text-xs text-gray-400')
                            ui.select(['Любая', 'Горизонтальная', 'Вертикальная', 'Квадрат'], value='Любая').bind_value(state, 'filter_orientation').classes('w-full')
                        ui.button('Применить', on_click=search_gallery_ui.refresh).props('outline color=blue').classes('h-[40px]')

                with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                    ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                    ui.label(f'Страница {state.search_page} из {total_pages}').classes('text-gray-300 font-bold')
                    ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')

            scroll_id = 'search_scroll_area'
            with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                start_idx = (state.search_page - 1) * ITEMS_PER_PAGE
                page_items = filtered_results[start_idx : start_idx + ITEMS_PER_PAGE]
                all_paths =[p for s, p in filtered_results]

                if not page_items:
                    ui.label("Нет файлов, подходящих под выбранный фильтр.").classes("text-gray-400 m-4")

                with ui.grid(columns=int(state.grid_columns)).classes('w-full gap-6 pb-10'):
                    for score, path in page_items:
                        safe_path = urllib.parse.quote(path)
                        global_index = all_paths.index(path)
                        
                        with ui.card().classes('bg-gray-800 border border-gray-700 hover:border-blue-500 transition-colors p-0 overflow-hidden relative'):
                            with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                ui.checkbox().bind_value(state.sel_search, path).on('click', lambda e, i=global_index, p=path: handle_shift_click(e, i, p, 'search'), ['shiftKey'])
                            
                            if path.lower().endswith(SUPPORTED_VIDEOS):
                                ui.label('▶ ВИДЕО').classes('absolute top-2 right-2 bg-blue-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded z-10 pointer-events-none shadow')

                            with ui.context_menu():
                                ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                ui.menu_item('Копировать картинку', on_click=lambda p=path: copy_image_to_clipboard(p))
                                ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))
                                ui.separator()
                                ui.menu_item('Удалить файл (В корзину)', on_click=lambda p=path: delete_items([p], 'search')).classes('text-red-400')

                            if os.path.splitext(path)[1].lower() in SUPPORTED_TEXTS:
                                ui.icon('article', size='4rem').classes('w-full aspect-square flex items-center justify-center bg-gray-900 cursor-pointer text-gray-500').on('click', lambda p=path: open_file_native(p))
                            else:
                                ui.image(f"/thumb/{safe_path}").classes('w-full aspect-square object-contain cursor-pointer bg-black').props('fit=contain loading="lazy"').on('click', lambda e, idx=global_index: open_media(idx, all_paths))
                            
                            with ui.row().classes('w-full justify-between items-center p-2 bg-gray-800'):
                                ui.label(f"Score: {score:.3f}").classes('text-green-400 font-bold text-sm')
                                ui.button(icon='folder', on_click=lambda p=path: reveal_file_native(p)).props('flat round dense color=white')
                            ui.label(os.path.basename(path)).classes('text-xs text-gray-400 px-2 pb-2 truncate w-full').tooltip(path)

            ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=blue').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')

    @ui.refreshable
    async def aesthetic_gallery_ui():
        if not state.aesthetic_results:
            ui.label("Здесь появятся топовые фото/видео...").classes("text-gray-400 m-4")
            return

        await asyncio.sleep(0.001)

        filtered_results =[]
        for item in state.aesthetic_results:
            p = item[1].lower()
            if state.aes_res_filter == 'Картинки' and not p.endswith(SUPPORTED_IMAGES): continue
            if state.aes_res_filter == 'Видео' and not p.endswith(SUPPORTED_VIDEOS): continue
            filtered_results.append(item)
            
        filtered_results = await apply_physical_filters_async(filtered_results)

        total_pages = max(1, (len(filtered_results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        if state.aes_page > total_pages: state.aes_page = 1

        def change_page(d):
            state.aes_page = max(1, min(total_pages, state.aes_page + d))
            aesthetic_gallery_ui.refresh()

        def apply_filter(e):
            state.aes_res_filter = e.value
            state.aes_page = 1
            aesthetic_gallery_ui.refresh()

        with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
            with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('Выбрать всё', on_click=lambda: ui.timer(0, lambda: set_all('aes', True), once=True)).props('outline color=white dense')
                        ui.button('Снять всё', on_click=lambda: ui.timer(0, lambda: set_all('aes', False), once=True)).props('outline color=white dense')
                        ui.toggle(['Все', 'Картинки', 'Видео'], value=state.aes_res_filter, on_change=apply_filter).classes('text-xs ml-2')
                        ui.button(icon='filter_alt', on_click=lambda: (setattr(state, 'show_phys_filters', not getattr(state, 'show_phys_filters', False)), aesthetic_gallery_ui.refresh())).props('flat color=gray dense').tooltip('Доп. фильтры')
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('HTML Экспорт', icon='html', on_click=lambda: export_html_action('aes')).props('color=purple dense outline')
                        ui.button('Копировать ✔', icon='content_copy', on_click=lambda: execute_batch('copy', 'aes', chk_prefix_aes.value)).props('color=yellow-800 dense')
                        ui.button('Переместить ✔', icon='drive_file_move', on_click=lambda: execute_batch('move', 'aes', chk_prefix_aes.value)).props('color=red dense')
                        ui.button('УДАЛИТЬ ✔', icon='delete_forever', on_click=lambda: delete_items([p for p, c in state.sel_aes.items() if c], 'aes')).props('color=red-10 text-white dense')

                if getattr(state, 'show_phys_filters', False):
                    with ui.row().classes('w-full bg-gray-800/50 p-2 rounded-lg mb-2 items-end gap-4 border border-gray-700'):
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Разрешение (Max сторона, px):').classes('text-xs text-gray-400')
                            with ui.row().classes('w-full gap-2 flex-nowrap'):
                                ui.number('Мин', value=0, format='%.0f').bind_value(state, 'filter_min_res').classes('flex-1')
                                ui.number('Макс', value=10000, format='%.0f').bind_value(state, 'filter_max_res').classes('flex-1')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Макс. вес:').classes('text-xs text-gray-400')
                            ui.number('МБ', value=10000, format='%.0f').bind_value(state, 'filter_max_size').classes('w-full')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Ориентация:').classes('text-xs text-gray-400')
                            ui.select(['Любая', 'Горизонтальная', 'Вертикальная', 'Квадрат'], value='Любая').bind_value(state, 'filter_orientation').classes('w-full')
                        ui.button('Применить', on_click=aesthetic_gallery_ui.refresh).props('outline color=yellow-800').classes('h-[40px]')

                with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                    ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                    ui.label(f'Страница {state.aes_page} из {total_pages}').classes('text-gray-300 font-bold')
                    ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')

            scroll_id = 'aes_scroll_area'
            with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                start_idx = (state.aes_page - 1) * ITEMS_PER_PAGE
                page_items = filtered_results[start_idx : start_idx + ITEMS_PER_PAGE]
                all_paths =[p for a, p, m in filtered_results]

                if not page_items:
                    ui.label("Нет файлов, подходящих под выбранный фильтр.").classes("text-gray-400 m-4")

                with ui.grid(columns=int(state.grid_columns)).classes('w-full gap-6 pb-10'):
                    for avg_score, path, max_score in page_items:
                        safe_path = urllib.parse.quote(path)
                        global_index = all_paths.index(path)
                        
                        with ui.card().classes('bg-gray-800 border border-gray-700 hover:border-yellow-500 transition-colors p-0 overflow-hidden relative'):
                            with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                ui.checkbox().bind_value(state.sel_aes, path).on('click', lambda e, i=global_index, p=path: handle_shift_click(e, i, p, 'aes'),['shiftKey'])

                            if path.lower().endswith(SUPPORTED_VIDEOS):
                                ui.label('▶ ВИДЕО').classes('absolute top-2 right-2 bg-blue-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded z-10 pointer-events-none shadow')

                            with ui.context_menu():
                                ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                ui.menu_item('Копировать картинку', on_click=lambda p=path: copy_image_to_clipboard(p))
                                ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))
                                ui.separator()
                                ui.menu_item('Удалить файл (В корзину)', on_click=lambda p=path: delete_items([p], 'aes')).classes('text-red-400')

                            ui.image(f"/thumb/{safe_path}").classes('w-full aspect-square object-contain cursor-pointer bg-black').props('fit=contain loading="lazy"').on('click', lambda e, idx=global_index: open_media(idx, all_paths))
                            
                            with ui.row().classes('w-full justify-between items-center p-2'):
                                ui.label(f"★ {avg_score:.2f}").classes('text-yellow-400 font-bold text-lg')
                                if avg_score != max_score: ui.label(f"Пик: {max_score:.2f}").classes('text-xs text-gray-500')
                            ui.label(os.path.basename(path)).classes('text-xs text-gray-400 px-2 pb-2 truncate w-full').tooltip(path)

            ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=yellow-800').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')

    @ui.refreshable
    async def nsfw_gallery_ui():
        if not state.nsfw_results:
            ui.label("Здесь появятся результаты NSFW сканирования...").classes("text-gray-400 m-4")
            return
        
        await asyncio.sleep(0.001)

        filtered_results =[]
        for item in state.nsfw_results:
            p = item[1].lower()
            if state.nsfw_res_filter == 'Картинки' and not p.endswith(SUPPORTED_IMAGES): continue
            if state.nsfw_res_filter == 'Видео' and not p.endswith(SUPPORTED_VIDEOS): continue
            filtered_results.append(item)
            
        filtered_results = await apply_physical_filters_async(filtered_results)

        total_pages = max(1, (len(filtered_results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        if state.nsfw_page > total_pages: state.nsfw_page = 1

        def change_page(d):
            state.nsfw_page = max(1, min(total_pages, state.nsfw_page + d))
            nsfw_gallery_ui.refresh()

        def apply_filter(e):
            state.nsfw_res_filter = e.value
            state.nsfw_page = 1
            nsfw_gallery_ui.refresh()

        with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
            with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('Выбрать всё', on_click=lambda: ui.timer(0, lambda: set_all('nsfw', True), once=True)).props('outline color=white dense')
                        ui.button('Снять всё', on_click=lambda: ui.timer(0, lambda: set_all('nsfw', False), once=True)).props('outline color=white dense')
                        ui.toggle(['Все', 'Картинки', 'Видео'], value=state.nsfw_res_filter, on_change=apply_filter).classes('text-xs ml-2')
                        ui.button(icon='filter_alt', on_click=lambda: (setattr(state, 'show_phys_filters', not getattr(state, 'show_phys_filters', False)), nsfw_gallery_ui.refresh())).props('flat color=gray dense').tooltip('Доп. фильтры')
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('HTML Экспорт', icon='html', on_click=lambda: export_html_action('nsfw')).props('color=purple dense outline')
                        ui.button('Копировать ✔', icon='content_copy', on_click=lambda: execute_batch('copy', 'nsfw', chk_prefix_nsfw.value)).props('color=red-800 dense')
                        ui.button('Переместить ✔', icon='drive_file_move', on_click=lambda: execute_batch('move', 'nsfw', chk_prefix_nsfw.value)).props('color=red dense')
                        ui.button('УДАЛИТЬ ✔', icon='delete_forever', on_click=lambda: delete_items([p for p, c in state.sel_nsfw.items() if c], 'nsfw')).props('color=red-10 text-white dense')

                if getattr(state, 'show_phys_filters', False):
                    with ui.row().classes('w-full bg-gray-800/50 p-2 rounded-lg mb-2 items-end gap-4 border border-gray-700'):
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Разрешение (Max сторона, px):').classes('text-xs text-gray-400')
                            with ui.row().classes('w-full gap-2 flex-nowrap'):
                                ui.number('Мин', value=0, format='%.0f').bind_value(state, 'filter_min_res').classes('flex-1')
                                ui.number('Макс', value=10000, format='%.0f').bind_value(state, 'filter_max_res').classes('flex-1')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Макс. вес:').classes('text-xs text-gray-400')
                            ui.number('МБ', value=10000, format='%.0f').bind_value(state, 'filter_max_size').classes('w-full')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Ориентация:').classes('text-xs text-gray-400')
                            ui.select(['Любая', 'Горизонтальная', 'Вертикальная', 'Квадрат'], value='Любая').bind_value(state, 'filter_orientation').classes('w-full')
                        ui.button('Применить', on_click=nsfw_gallery_ui.refresh).props('outline color=red-800').classes('h-[40px]')

                with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                    ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                    ui.label(f'Страница {state.nsfw_page} из {total_pages}').classes('text-gray-300 font-bold')
                    ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')

            scroll_id = 'nsfw_scroll_area'
            with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                start_idx = (state.nsfw_page - 1) * ITEMS_PER_PAGE
                page_items = filtered_results[start_idx : start_idx + ITEMS_PER_PAGE]
                all_paths =[p for d, p, l, dt in filtered_results]

                if not page_items:
                    ui.label("Нет файлов, подходящих под выбранный фильтр.").classes("text-gray-400 m-4")

                with ui.grid(columns=int(state.grid_columns)).classes('w-full gap-6 pb-10'):
                    for danger_score, path, top_label, details in page_items:
                        safe_path = urllib.parse.quote(path)
                        global_index = all_paths.index(path)
                        
                        with ui.card().classes('bg-gray-800 border border-gray-700 hover:border-red-500 transition-colors p-0 overflow-hidden relative'):
                            with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                ui.checkbox().bind_value(state.sel_nsfw, path).on('click', lambda e, i=global_index, p=path: handle_shift_click(e, i, p, 'nsfw'), ['shiftKey'])

                            if path.lower().endswith(SUPPORTED_VIDEOS):
                                ui.label('▶ ВИДЕО').classes('absolute top-2 right-2 bg-blue-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded z-10 pointer-events-none shadow')

                            with ui.context_menu():
                                ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                ui.menu_item('Копировать картинку', on_click=lambda p=path: copy_image_to_clipboard(p))
                                ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))
                                ui.separator()
                                ui.menu_item('Удалить файл (В корзину)', on_click=lambda p=path: delete_items([p], 'nsfw')).classes('text-red-400')

                            ui.image(f"/thumb/{safe_path}").classes('w-full aspect-square object-contain cursor-pointer bg-black').props('fit=contain loading="lazy"').on('click', lambda e, idx=global_index: open_media(idx, all_paths))
                            
                            with ui.row().classes('w-full justify-between items-center p-2'):
                                ui.label(f"🚨 {danger_score*100:.1f}%").classes('text-red-500 font-bold text-lg')
                                ui.button(icon='troubleshoot', on_click=lambda p=path, d=details: show_nsfw_debug(p, d)).props('flat round dense color=white').tooltip('Детальный разбор категорий')
                                
                            ui.label(top_label.upper()).classes('text-xs text-gray-400 font-bold px-2')
                            ui.label(os.path.basename(path)).classes('text-xs text-gray-400 px-2 pb-2 truncate w-full').tooltip(path)

            ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=red-800').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')

    @ui.refreshable
    async def face_gallery_ui():
        if not state.face_results:
            ui.label("Здесь появятся найденные фотографии с искомым лицом...").classes("text-gray-400 m-4")
            return
        
        await asyncio.sleep(0.001)

        filtered_results =[]
        for item in state.face_results:
            p = item[1].lower()
            if state.face_res_filter == 'Картинки' and not p.endswith(SUPPORTED_IMAGES): continue
            if state.face_res_filter == 'Видео' and not p.endswith(SUPPORTED_VIDEOS): continue
            filtered_results.append(item)
            
        filtered_results = await apply_physical_filters_async(filtered_results)

        total_pages = max(1, (len(filtered_results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        if state.face_page > total_pages: state.face_page = 1

        def change_page(d):
            state.face_page = max(1, min(total_pages, state.face_page + d))
            face_gallery_ui.refresh()

        def apply_filter(e):
            state.face_res_filter = e.value
            state.face_page = 1
            face_gallery_ui.refresh()

        with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
            with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('Выбрать всё', on_click=lambda: ui.timer(0, lambda: set_all('face', True), once=True)).props('outline color=white dense')
                        ui.button('Снять всё', on_click=lambda: ui.timer(0, lambda: set_all('face', False), once=True)).props('outline color=white dense')
                        ui.toggle(['Все', 'Картинки', 'Видео'], value=state.face_res_filter, on_change=apply_filter).classes('text-xs ml-2')
                        ui.button(icon='filter_alt', on_click=lambda: (setattr(state, 'show_phys_filters', not getattr(state, 'show_phys_filters', False)), face_gallery_ui.refresh())).props('flat color=gray dense').tooltip('Доп. фильтры')
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('HTML Экспорт', icon='html', on_click=lambda: export_html_action('face')).props('color=purple dense outline')
                        ui.button('Копировать ✔', icon='content_copy', on_click=lambda: execute_batch('copy', 'face', chk_prefix_face.value)).props('color=teal-800 dense')
                        ui.button('Переместить ✔', icon='drive_file_move', on_click=lambda: execute_batch('move', 'face', chk_prefix_face.value)).props('color=red dense')
                        ui.button('УДАЛИТЬ ✔', icon='delete_forever', on_click=lambda: delete_items([p for p, c in state.sel_face.items() if c], 'face')).props('color=red-10 text-white dense')

                if getattr(state, 'show_phys_filters', False):
                    with ui.row().classes('w-full bg-gray-800/50 p-2 rounded-lg mb-2 items-end gap-4 border border-gray-700'):
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Разрешение (Max сторона, px):').classes('text-xs text-gray-400')
                            with ui.row().classes('w-full gap-2 flex-nowrap'):
                                ui.number('Мин', value=0, format='%.0f').bind_value(state, 'filter_min_res').classes('flex-1')
                                ui.number('Макс', value=10000, format='%.0f').bind_value(state, 'filter_max_res').classes('flex-1')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Макс. вес:').classes('text-xs text-gray-400')
                            ui.number('МБ', value=10000, format='%.0f').bind_value(state, 'filter_max_size').classes('w-full')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Ориентация:').classes('text-xs text-gray-400')
                            ui.select(['Любая', 'Горизонтальная', 'Вертикальная', 'Квадрат'], value='Любая').bind_value(state, 'filter_orientation').classes('w-full')
                        ui.button('Применить', on_click=face_gallery_ui.refresh).props('outline color=teal-800').classes('h-[40px]')

                with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                    ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                    ui.label(f'Страница {state.face_page} из {total_pages}').classes('text-gray-300 font-bold')
                    ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')

            scroll_id = 'face_scroll_area'
            with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                start_idx = (state.face_page - 1) * ITEMS_PER_PAGE
                page_items = filtered_results[start_idx : start_idx + ITEMS_PER_PAGE]
                all_paths =[p for s, p in filtered_results]

                if not page_items:
                    ui.label("Нет файлов, подходящих под выбранный фильтр.").classes("text-gray-400 m-4")

                with ui.grid(columns=int(state.grid_columns)).classes('w-full gap-6 pb-10'):
                    for sim_score, path in page_items:
                        safe_path = urllib.parse.quote(path)
                        global_index = all_paths.index(path)
                        
                        with ui.card().classes('bg-gray-800 border border-gray-700 hover:border-teal-500 transition-colors p-0 overflow-hidden relative'):
                            with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                ui.checkbox().bind_value(state.sel_face, path).on('click', lambda e, i=global_index, p=path: handle_shift_click(e, i, p, 'face'),['shiftKey'])

                            if path.lower().endswith(SUPPORTED_VIDEOS):
                                ui.label('▶ ВИДЕО').classes('absolute top-2 right-2 bg-blue-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded z-10 pointer-events-none shadow')

                            with ui.context_menu():
                                ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                ui.menu_item('Копировать картинку', on_click=lambda p=path: copy_image_to_clipboard(p))
                                ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))
                                ui.separator()
                                ui.menu_item('Удалить файл (В корзину)', on_click=lambda p=path: delete_items([p], 'face')).classes('text-red-400')

                            ui.image(f"/thumb/{safe_path}").classes('w-full aspect-square object-contain cursor-pointer bg-black').props('fit=contain loading="lazy"').on('click', lambda e, idx=global_index: open_media(idx, all_paths))
                            
                            with ui.row().classes('w-full justify-between items-center p-2 bg-gray-800'):
                                ui.label(f"Сходство: {sim_score*100:.1f}%").classes('text-teal-400 font-bold text-sm')
                                ui.button(icon='folder', on_click=lambda p=path: reveal_file_native(p)).props('flat round dense color=white')
                            ui.label(os.path.basename(path)).classes('text-xs text-gray-400 px-2 pb-2 truncate w-full').tooltip(path)

            ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=teal-800').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')

    # --- КОМПОНЕНТ ГАЛЕРЕИ ТЕГОВ ---
    @ui.refreshable
    async def tags_gallery_ui():
        if not state.tags_results:
            ui.label("Здесь появятся картинки, подходящие под выбранные теги...").classes("text-gray-400 m-4")
            return
        
        await asyncio.sleep(0.001)

        filtered_results =[]
        for item in state.tags_results:
            p = item[1].lower()
            if state.tags_res_filter == 'Картинки' and not p.endswith(SUPPORTED_IMAGES): continue
            if state.tags_res_filter == 'Видео' and not p.endswith(SUPPORTED_VIDEOS): continue
            filtered_results.append(item)
            
        filtered_results = await apply_physical_filters_async(filtered_results)

        total_pages = max(1, (len(filtered_results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        if state.tags_page > total_pages: state.tags_page = 1

        def change_page(d):
            state.tags_page = max(1, min(total_pages, state.tags_page + d))
            tags_gallery_ui.refresh()

        def apply_filter(e):
            state.tags_res_filter = e.value
            state.tags_page = 1
            tags_gallery_ui.refresh()

        with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
            with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('Выбрать всё', on_click=lambda: ui.timer(0, lambda: set_all('tags', True), once=True)).props('outline color=white dense')
                        ui.button('Снять всё', on_click=lambda: ui.timer(0, lambda: set_all('tags', False), once=True)).props('outline color=white dense')
                        ui.toggle(['Все', 'Картинки', 'Видео'], value=state.tags_res_filter, on_change=apply_filter).classes('text-xs ml-2')
                        ui.button(icon='filter_alt', on_click=lambda: (setattr(state, 'show_phys_filters', not getattr(state, 'show_phys_filters', False)), tags_gallery_ui.refresh())).props('flat color=gray dense').tooltip('Доп. фильтры')
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('HTML Экспорт', icon='html', on_click=lambda: export_html_action('tags')).props('color=purple dense outline')
                        ui.button('Копировать ✔', icon='content_copy', on_click=lambda: execute_batch('copy', 'tags', False, chk_txt_tags.value, tags_threshold.value)).props('color=pink-800 dense')
                        ui.button('Переместить ✔', icon='drive_file_move', on_click=lambda: execute_batch('move', 'tags', False, chk_txt_tags.value, tags_threshold.value)).props('color=red dense')
                        ui.button('УДАЛИТЬ ✔', icon='delete_forever', on_click=lambda: delete_items([p for p, c in state.sel_tags.items() if c], 'tags')).props('color=red-10 text-white dense')

                if getattr(state, 'show_phys_filters', False):
                    with ui.row().classes('w-full bg-gray-800/50 p-2 rounded-lg mb-2 items-end gap-4 border border-gray-700'):
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Разрешение (Max сторона, px):').classes('text-xs text-gray-400')
                            with ui.row().classes('w-full gap-2 flex-nowrap'):
                                ui.number('Мин', value=0, format='%.0f').bind_value(state, 'filter_min_res').classes('flex-1')
                                ui.number('Макс', value=10000, format='%.0f').bind_value(state, 'filter_max_res').classes('flex-1')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Макс. вес:').classes('text-xs text-gray-400')
                            ui.number('МБ', value=10000, format='%.0f').bind_value(state, 'filter_max_size').classes('w-full')
                        with ui.column().classes('gap-1 flex-1'):
                            ui.label('Ориентация:').classes('text-xs text-gray-400')
                            ui.select(['Любая', 'Горизонтальная', 'Вертикальная', 'Квадрат'], value='Любая').bind_value(state, 'filter_orientation').classes('w-full')
                        ui.button('Применить', on_click=tags_gallery_ui.refresh).props('outline color=pink-800').classes('h-[40px]')

                with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                    ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                    ui.label(f'Страница {state.tags_page} из {total_pages}').classes('text-gray-300 font-bold')
                    ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')

            scroll_id = 'tags_scroll_area'
            with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                start_idx = (state.tags_page - 1) * ITEMS_PER_PAGE
                page_items = filtered_results[start_idx : start_idx + ITEMS_PER_PAGE]
                all_paths = [p for s, p, t in filtered_results]

                if not page_items:
                    ui.label("Нет файлов, подходящих под выбранный фильтр.").classes("text-gray-400 m-4")

                with ui.grid(columns=int(state.grid_columns)).classes('w-full gap-6 pb-10'):
                    for score, path, tags_dict in page_items:
                        safe_path = urllib.parse.quote(path)
                        global_index = all_paths.index(path)
                        
                        # Берем топ-3 тегов для предпросмотра
                        top_tags = sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                        top_tags_str = ", ".join([f"{k}" for k, v in top_tags])
                        
                        with ui.card().classes('bg-gray-800 border border-gray-700 hover:border-pink-500 transition-colors p-0 overflow-hidden relative'):
                            with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                ui.checkbox().bind_value(state.sel_tags, path).on('click', lambda e, i=global_index, p=path: handle_shift_click(e, i, p, 'tags'), ['shiftKey'])

                            if path.lower().endswith(SUPPORTED_VIDEOS):
                                ui.label('▶ ВИДЕО').classes('absolute top-2 right-2 bg-blue-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded z-10 pointer-events-none shadow')

                            with ui.context_menu():
                                ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                ui.menu_item('Копировать картинку', on_click=lambda p=path: copy_image_to_clipboard(p))
                                ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))
                                ui.separator()
                                ui.menu_item('Удалить файл (В корзину)', on_click=lambda p=path: delete_items([p], 'tags')).classes('text-red-400')

                            ui.image(f"/thumb/{safe_path}").classes('w-full aspect-square object-contain cursor-pointer bg-black').props('fit=contain loading="lazy"').on('click', lambda e, idx=global_index: open_media(idx, all_paths))
                            
                            with ui.row().classes('w-full justify-between items-center p-2 bg-gray-800'):
                                ui.label(top_tags_str).classes('text-pink-400 font-bold text-xs truncate max-w-[80%]').tooltip(", ".join([f"{k} ({v:.2f})" for k, v in top_tags]))
                                ui.button(icon='sell', on_click=lambda p=path, d=tags_dict: show_tags_debug(p, d)).props('flat round dense color=white').tooltip('Все теги')
                            ui.label(os.path.basename(path)).classes('text-xs text-gray-400 px-2 pb-2 truncate w-full').tooltip(path)
                        
            ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=pink-800').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')

    async def auto_select_worst_dupes():
        ui.notify("Анализ всех файлов (в т.ч. скрытых)...", type="info")
        
        def task():
            c = search_engine.db_cache.conn.cursor()
            c.execute("SELECT path, size_mb, width, height FROM files")
            info = {r[0]: (r[1], r[2], r[3]) for r in c.fetchall()}
            
            updates = {}
            for group in state.dupes_results:
                scored =[]
                for p in group:
                    size, w, h = info.get(p, (0, 0, 0))
                    res = (w or 0) * (h or 0)
                    scored.append((res, size or 0, p))
                
                # Сортируем: сначала самое большое разрешение, потом самый большой вес
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                updates[scored[0][2]] = False # Лучший файл - оставляем (снимаем галочку)
                for item in scored[1:]:
                    updates[item[2]] = True   # Все остальные (в т.ч. скрытые) - помечаем на удаление
            return updates
            
        updates = await run.io_bound(task)
        state.sel_dupes.update(updates)
        dupes_gallery_ui.refresh()
        ui.notify("Худшие дубликаты (включая скрытые) помечены на удаление!", type="positive", color="green")

    @ui.refreshable
    async def dupes_gallery_ui():
        if not state.dupes_results:
            ui.label("Найденные дубликаты появятся здесь...").classes("text-gray-400 m-4")
            return
        
        await asyncio.sleep(0.001)
            
        # ЖЕСТКИЕ ЛИМИТЫ ДЛЯ АБСОЛЮТНОЙ СТАБИЛЬНОСТИ
        GROUPS_PER_PAGE = int(state.groups_per_page)
        MAX_ITEMS_PER_GROUP = 40  
        
        total_pages = max(1, (len(state.dupes_results) + GROUPS_PER_PAGE - 1) // GROUPS_PER_PAGE)
        if getattr(state, 'dupes_page', 1) > total_pages: state.dupes_page = 1
        
        def change_page(d):
            state.dupes_page = max(1, min(total_pages, getattr(state, 'dupes_page', 1) + d))
            dupes_gallery_ui.refresh()

        with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
            with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('АВТО-ВЫБОР ХУДШИХ', icon='auto_awesome', on_click=auto_select_worst_dupes).props('color=orange text-black font-bold dense')
                        ui.button('Снять всё', on_click=lambda: ui.timer(0, lambda: set_all('dupes', False), once=True)).props('outline color=white dense')
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('УДАЛИТЬ ВЫДЕЛЕННЫЕ ✔', icon='delete_forever', on_click=lambda: delete_items([p for p, c in state.sel_dupes.items() if c], 'dupes')).props('color=red-10 text-white dense')
                
                with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                    ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                    ui.label(f'Страница {getattr(state, "dupes_page", 1)} из {total_pages}').classes('text-gray-300 font-bold')
                    ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')
            
            scroll_id = 'dupes_scroll_area'
            with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                start_idx = (getattr(state, 'dupes_page', 1) - 1) * GROUPS_PER_PAGE
                page_groups = state.dupes_results[start_idx : start_idx + GROUPS_PER_PAGE]
                
                ITEMS_PER_INNER_PAGE = 60 # Лимит при развороте
                
                def render_dupe_group(group_idx, group):
                    is_expanded = {'val': False}
                    inner_page = {'val': 1}
                    
                    with ui.card().classes('w-full bg-gray-800 border border-gray-700 p-2 mb-4'):
                        with ui.row().classes('w-full justify-between items-center px-2 mb-2'):
                            ui.label(f"Группа {start_idx + group_idx + 1} (Всего файлов: {len(group)})").classes('font-bold text-orange-400')
                            
                            with ui.row().classes('gap-2 items-center'):
                                def select_group(val):
                                    for p in group: state.sel_dupes[p] = val
                                    update_view()
                                    
                                ui.button('Выделить группу', on_click=lambda: select_group(True)).props('outline size=sm color=green')
                                ui.button('Снять выделение', on_click=lambda: select_group(False)).props('outline size=sm color=red')
                                
                                btn_toggle = ui.button('Развернуть', on_click=lambda: toggle_expand()).props('size=sm color=gray')
                                if len(group) <= MAX_ITEMS_PER_GROUP:
                                    btn_toggle.set_visibility(False)

                        content_container = ui.column().classes('w-full p-0 m-0')

                        def toggle_expand():
                            is_expanded['val'] = not is_expanded['val']
                            inner_page['val'] = 1
                            btn_toggle.text = 'Свернуть' if is_expanded['val'] else 'Развернуть'
                            btn_toggle._props['color'] = 'orange' if is_expanded['val'] else 'gray'
                            btn_toggle.update()
                            update_view()

                        def update_view():
                            content_container.clear()
                            with content_container:
                                if is_expanded['val']:
                                    start_i = (inner_page['val'] - 1) * ITEMS_PER_INNER_PAGE
                                    end_i = start_i + ITEMS_PER_INNER_PAGE
                                    visible_group = group[start_i:end_i]
                                    row_cls = 'w-full gap-4 pb-2 items-start flex-wrap'
                                else:
                                    visible_group = group[:MAX_ITEMS_PER_GROUP]
                                    row_cls = 'w-full gap-4 pb-2 items-start overflow-x-auto flex-nowrap'

                                hidden_count = 0 if is_expanded['val'] else len(group) - MAX_ITEMS_PER_GROUP

                                with ui.row().classes(row_cls):
                                    for path in visible_group:
                                        safe_path = urllib.parse.quote(path)
                                        local_index = group.index(path)
                                        
                                        with ui.column().classes('w-[200px] shrink-0 relative bg-gray-900 rounded overflow-hidden border border-gray-700 hover:border-orange-500 transition-colors'):
                                            with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                                ui.checkbox().bind_value(state.sel_dupes, path).on('click', lambda e, i=local_index, p=path, paths=group: handle_shift_click(e, i, p, 'dupes', paths),['shiftKey'])
                                            
                                            if path.lower().endswith(SUPPORTED_VIDEOS):
                                                ui.label('▶ ВИДЕО').classes('absolute top-2 right-2 bg-blue-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded z-10 pointer-events-none shadow')

                                            with ui.context_menu():
                                                ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                                ui.menu_item('Копировать картинку', on_click=lambda p=path: copy_image_to_clipboard(p))
                                                ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))
                                                ui.separator()
                                                ui.menu_item('Удалить файл (В корзину)', on_click=lambda p=path: delete_items([p], 'dupes')).classes('text-red-400')

                                            # Плеер ограничен текущей группой
                                            ui.image(f"/thumb/{safe_path}").classes('w-full h-[150px] object-contain cursor-pointer bg-black').props('fit=contain loading="lazy"').on('click', lambda e, idx=local_index, paths=group: open_media(idx, paths))
                                            
                                            c = search_engine.db_cache.conn.cursor()
                                            c.execute("SELECT size_mb, width, height FROM files WHERE path=?", (path,))
                                            info = c.fetchone()
                                            size_str = f"{info[0]:.2f} MB" if info and info[0] else "N/A"
                                            res_str = f"{info[1]}x{info[2]}" if info and info[1] else "N/A"
                                            
                                            with ui.column().classes('p-2 gap-0 w-full'):
                                                with ui.row().classes('w-full justify-between items-center'):
                                                    ui.label(res_str).classes('text-green-400 font-bold text-xs')
                                                    ui.button(icon='folder', on_click=lambda p=path: reveal_file_native(p)).props('flat round dense color=white size=xs').tooltip('Открыть папку')
                                                ui.label(size_str).classes('text-yellow-400 font-bold text-xs')
                                                ui.label(os.path.basename(path)).classes('text-gray-400 text-[10px] truncate w-full').tooltip(path)

                                    if hidden_count > 0 and not is_expanded['val']:
                                        with ui.card().classes('w-[200px] h-[210px] shrink-0 flex flex-col items-center justify-center bg-gray-900 border border-dashed border-gray-600 gap-2 p-4 cursor-pointer hover:border-orange-500 transition-colors').on('click', toggle_expand):
                                            ui.icon('more_horiz', size='3rem').classes('text-gray-500')
                                            ui.label(f"+ еще {hidden_count} шт.").classes('text-center font-bold text-gray-300 text-lg')
                                            ui.label("Нажмите, чтобы развернуть").classes('text-[10px] text-center text-orange-400')
                                            
                                # Внутренняя пагинация для больших групп
                                if is_expanded['val'] and len(group) > ITEMS_PER_INNER_PAGE:
                                    tot_inner_pages = max(1, (len(group) + ITEMS_PER_INNER_PAGE - 1) // ITEMS_PER_INNER_PAGE)
                                    
                                    def change_inner_page(d):
                                        inner_page['val'] = max(1, min(tot_inner_pages, inner_page['val'] + d))
                                        update_view()

                                    with ui.row().classes('w-full justify-center items-center gap-4 py-2 border-t border-gray-700 mt-2'):
                                        ui.button(icon='chevron_left', on_click=lambda: change_inner_page(-1)).props('flat outline color=orange size=sm')
                                        ui.label(f'Под-страница {inner_page["val"]} из {tot_inner_pages}').classes('text-gray-400 text-xs font-bold')
                                        ui.button(icon='chevron_right', on_click=lambda: change_inner_page(1)).props('flat outline color=orange size=sm')

                        update_view() # Первичная отрисовка при создании

                for group_idx, group in enumerate(page_groups):
                    render_dupe_group(group_idx, group)

            ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=orange-800').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')

    # --- ОСНОВНАЯ РАБОЧАЯ ОБЛАСТЬ ---
    with ui.tab_panels(tabs).bind_value(state, 'current_tab').classes('w-full bg-[#121212] p-0'):
        
        # ВКЛАДКА 1: ПОИСК
        with ui.tab_panel(tab_search).classes('w-full h-[calc(100vh-115px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('Параметры поиска').classes('text-lg font-bold')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto p-4 gap-2 min-h-0'):
                    with ui.row().classes('w-full items-start gap-1 flex-nowrap'):
                        inp_dir = ui.textarea('Папки (каждая с новой строки)', value=cfg.get('inp_dir', '')).classes('flex-grow').props('rows=2')
                        with ui.column().classes('gap-0 pt-2'):
                            ui.button(icon='create_new_folder', on_click=lambda: select_folder_multi(inp_dir)).props('flat round dense').tooltip('Добавить папку')
                            ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache_multi(inp_dir.value)).props('flat round dense text-color=red').tooltip('Очистить кэш этих папок')
                    
                    inp_query = ui.input('Запрос или путь', value=cfg.get('inp_query', '')).classes('w-full')
                    
                    with ui.row().classes('w-full gap-2'):
                        chk_img = ui.checkbox('Картинки', value=cfg.get('chk_img', True))
                        chk_vid = ui.checkbox('Видео', value=cfg.get('chk_vid', True))
                        chk_txt = ui.checkbox('Текст', value=cfg.get('chk_txt', False))
                    
                    emb_model = ui.select(['Qwen/Qwen3-VL-Embedding-2B', 'Qwen/Qwen3-VL-Embedding-8B'], value=cfg.get('emb_model', 'Qwen/Qwen3-VL-Embedding-2B'), label='Модель').classes('w-full')
                    top_k = ui.number('Топ K', value=cfg.get('top_k', 50), format='%.0f').classes('w-full')
                    use_reranker = ui.switch('Глубокий анализ (Reranker)', value=cfg.get('use_reranker', False))
                    rerank_model = ui.select(['Qwen/Qwen3-VL-Reranker-2B', 'Qwen/Qwen3-VL-Reranker-8B'], value=cfg.get('rerank_model', 'Qwen/Qwen3-VL-Reranker-2B')).classes('w-full').bind_visibility_from(use_reranker, 'value')
                    
                    chk_prefix_search = ui.checkbox('Писать Score в имя при копировании', value=cfg.get('chk_prefix_search', False)).classes('text-sm text-gray-300 w-full mt-2')

                    with ui.expansion('Тонкие настройки поиска', icon='tune').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-2'):
                        with ui.row().classes('w-full gap-2 px-2 pt-2'):
                            batch_size = ui.number('Батч', value=cfg.get('batch_size', 16), format='%.0f').classes('w-[45%]')
                            video_frames = ui.number('Кадры', value=cfg.get('video_frames', 4), format='%.0f').classes('w-[45%]')
                        with ui.row().classes('w-full gap-2 px-2'):
                            emb_size = ui.number('Рез. (Ф1)', value=cfg.get('emb_size', 512), format='%.0f').classes('w-[45%]')
                            rerank_size = ui.number('Рез. (Ф2)', value=cfg.get('rerank_size', 800), format='%.0f').classes('w-[45%]')
                        with ui.row().classes('w-full gap-2 px-2 pb-2'):
                            search_quant_mode = ui.select(['None', '8-bit', '4-bit'], value=cfg.get('search_quant_mode', 'None'), label='Квант').classes('w-[45%]')
                            min_score = ui.number('Мин Score', value=cfg.get('min_score', 0.25), format='%.2f', step=0.05).classes('w-[45%]')
                        
                        with ui.column().classes('w-full gap-0 px-2 pb-2 pt-2 border-t border-gray-700'):
                            search_nsfw_filter = ui.toggle(['Все', 'Только SFW', 'Только NSFW'], value=cfg.get('search_nsfw_filter', 'Все')).classes('w-full text-xs mb-1')
                            search_strict_nsfw = ui.checkbox('Строгий режим (скрывать файлы, которых нет в базе NSFW)', value=cfg.get('search_strict_nsfw', False)) \
                                .classes('text-xs text-red-400') \
                                .bind_visibility_from(search_nsfw_filter, 'value', value=lambda v: v != 'Все')

                async def run_search_action():
                    save_config({
                        'inp_dir': inp_dir.value, 'inp_query': inp_query.value,
                        'chk_img': chk_img.value, 'chk_vid': chk_vid.value, 'chk_txt': chk_txt.value,
                        'emb_model': emb_model.value, 'top_k': top_k.value,
                        'use_reranker': use_reranker.value, 'rerank_model': rerank_model.value,
                        'batch_size': batch_size.value, 'video_frames': video_frames.value,
                        'emb_size': emb_size.value, 'rerank_size': rerank_size.value,
                        'search_quant_mode': search_quant_mode.value, 'min_score': min_score.value,
                        'chk_prefix_search': chk_prefix_search.value,
                        'search_nsfw_filter': search_nsfw_filter.value, 'search_strict_nsfw': search_strict_nsfw.value
                    })
                    if not inp_dir.value or not inp_query.value: return ui.notify("Укажите папку и запрос!", type='warning')
                    
                    state.is_processing = True
                    search_engine.cancel_flag = False
                    state.search_results.clear()
                    state.sel_search.clear()
                    state.search_page = 1
                    search_gallery_ui.refresh()
                    btn_search.disable()
                    aesthetic_engine.unload()
                    nsfw_engine.unload()
                    face_engine.unload()
                    tag_engine.unload()
                    
                    exts =[]
                    if chk_img.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid.value: exts.extend(SUPPORTED_VIDEOS)
                    if chk_txt.value: exts.extend(SUPPORTED_TEXTS)
                    
                    def task():
                        try:
                            state.add_log(f"Запуск умного поиска по запросу: '{inp_query.value}'")
                            search_engine.video_frames = int(video_frames.value)
                            search_engine.emb_size = int(emb_size.value)
                            search_engine.rerank_size = int(rerank_size.value)
                            search_engine.quant_mode = search_quant_mode.value
                            
                            state.search_base_dir = inp_dir.value
                            q_emb, q_rank = search_engine.prepare_query(inp_query.value)
                            cands = search_engine.phase1_recall(inp_dir.value, inp_query.value, q_emb, int(top_k.value), emb_model.value, int(batch_size.value), tuple(exts))
                            
                            if use_reranker.value: 
                                cands = search_engine.phase2_rerank(inp_query.value, q_rank, cands, float(min_score.value), rerank_model.value)

                            if search_nsfw_filter.value != 'Все':
                                filtered_cands =[]
                                for score, path in cands:
                                    danger = search_engine.db_cache.get_max_danger_score(path)
                                    if danger == -1.0:
                                        if search_strict_nsfw.value: continue
                                        else:
                                            if search_nsfw_filter.value == 'Только NSFW': continue 
                                    else:
                                        is_nsfw = danger >= state.nsfw_threshold
                                        if search_nsfw_filter.value == 'Только SFW' and is_nsfw: continue
                                        if search_nsfw_filter.value == 'Только NSFW' and not is_nsfw: continue
                                    filtered_cands.append((score, path))
                                cands = filtered_cands
                                
                            state.search_results = cands
                            state.sel_search = {path: False for _, path in cands}
                            state.add_log("✅ Поиск успешно завершен!")
                        except Exception as e: state.add_log(f"❌ Ошибка поиска: {e}")
                        finally:
                            state.status_text = "Применение фильтров и рендеринг..."
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(task)
                    search_gallery_ui.refresh()
                    btn_search.enable()
                    state.status_text = "Готово!"

                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10'):
                    btn_search = ui.button('🚀 Искать', on_click=run_search_action).classes('w-full bg-blue-600 hover:bg-blue-500 font-bold')

            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                await search_gallery_ui()

        # ВКЛАДКА 2: ЭСТЕТИКА
        with ui.tab_panel(tab_aesthetic).classes('w-full h-[calc(100vh-115px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('Оценка Эстетики').classes('text-lg font-bold')
                
                    with ui.row().classes('w-full items-start gap-1 flex-nowrap'):
                        rate_dir = ui.textarea('Папки (каждая с новой строки)', value=cfg.get('rate_dir', '')).classes('flex-grow').props('rows=2')
                        with ui.column().classes('gap-0 pt-2'):
                            ui.button(icon='create_new_folder', on_click=lambda: select_folder_multi(rate_dir)).props('flat round dense').tooltip('Добавить папку')
                            ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache_multi(rate_dir.value)).props('flat round dense text-color=red').tooltip('Очистить кэш этих папок')

                    with ui.row().classes('w-full gap-2'):
                        chk_img_aes = ui.checkbox('Картинки', value=cfg.get('chk_img_aes', True))
                        chk_vid_aes = ui.checkbox('Видео', value=cfg.get('chk_vid_aes', False))

                    top_n_rate = ui.number('Оставить ТОП (шт)', value=cfg.get('top_n_rate', 100), format='%.0f').classes('w-full')
                    chk_prefix_aes = ui.checkbox('Писать Оценку в имя при копировании', value=cfg.get('chk_prefix_aes', False)).classes('text-sm text-gray-300 w-full mt-2')

                    with ui.expansion('Тонкие настройки', icon='tune').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-2'):
                        with ui.row().classes('w-full gap-2 px-2 pt-2'):
                            aes_batch_size = ui.number('Батч', value=cfg.get('aes_batch_size', 16), format='%.0f').classes('w-[45%]')
                            aes_video_frames = ui.number('Кадры', value=cfg.get('aes_video_frames', 4), format='%.0f').classes('w-[45%]')
                        with ui.row().classes('w-full gap-2 px-2 pb-2'):
                            aes_max_dim = ui.number('Лимит разр.', value=cfg.get('aes_max_dim', 512), format='%.0f').classes('w-[45%]')
                            aes_quant_mode = ui.select(['None', '8-bit', '4-bit'], value=cfg.get('aes_quant_mode', 'None'), label='Квант').classes('w-[45%]')
                            
                        with ui.column().classes('w-full gap-0 px-2 pb-2 pt-2 border-t border-gray-700'):
                            aes_nsfw_filter = ui.toggle(['Все', 'Только SFW', 'Только NSFW'], value=cfg.get('aes_nsfw_filter', 'Все')).classes('w-full text-xs mb-1')
                            aes_strict_nsfw = ui.checkbox('Строгий режим (скрывать файлы, которых нет в базе NSFW)', value=cfg.get('aes_strict_nsfw', False)) \
                                .classes('text-xs text-red-400') \
                                .bind_visibility_from(aes_nsfw_filter, 'value', value=lambda v: v != 'Все')
                
                async def run_aesthetic_action():
                    save_config({
                        'rate_dir': rate_dir.value, 'chk_img_aes': chk_img_aes.value, 'chk_vid_aes': chk_vid_aes.value,
                        'top_n_rate': top_n_rate.value, 'aes_batch_size': aes_batch_size.value, 
                        'aes_video_frames': aes_video_frames.value, 'aes_max_dim': aes_max_dim.value, 
                        'aes_quant_mode': aes_quant_mode.value, 'chk_prefix_aes': chk_prefix_aes.value,
                        'aes_nsfw_filter': aes_nsfw_filter.value, 'aes_strict_nsfw': aes_strict_nsfw.value
                    })
                    if not rate_dir.value: return ui.notify("Укажите папку!", type='warning')
                    
                    state.is_processing = True
                    search_engine.cancel_flag = False
                    state.aesthetic_results.clear()
                    state.sel_aes.clear()
                    state.aes_page = 1
                    aesthetic_gallery_ui.refresh()
                    btn_rate.disable()
                    search_engine._unload_embedding_model()
                    nsfw_engine.unload()
                    face_engine.unload()
                    tag_engine.unload()
                    
                    exts =[]
                    if chk_img_aes.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid_aes.value: exts.extend(SUPPORTED_VIDEOS)

                    def bg_task():
                        try:
                            state.add_log(f"Запуск оценки эстетики для папки: '{rate_dir.value}'")
                            aesthetic_engine.batch_size = int(aes_batch_size.value)
                            aesthetic_engine.video_frames = int(aes_video_frames.value)
                            aesthetic_engine.max_dim = int(aes_max_dim.value)
                            aesthetic_engine.quant_mode = aes_quant_mode.value
                            
                            state.aes_base_dir = rate_dir.value
                            n = int(top_n_rate.value)
                            
                            res = aesthetic_engine.evaluate_media(rate_dir.value, tuple(exts))

                            if aes_nsfw_filter.value != 'Все':
                                filtered_res =[]
                                for item in res:
                                    path = item[1] 
                                    danger = aesthetic_engine.db_cache.get_max_danger_score(path)
                                    if danger == -1.0: 
                                        if aes_strict_nsfw.value: continue
                                        else:
                                            if aes_nsfw_filter.value == 'Только NSFW': continue
                                    else:
                                        is_nsfw = danger >= state.nsfw_threshold
                                        if aes_nsfw_filter.value == 'Только SFW' and is_nsfw: continue
                                        if aes_nsfw_filter.value == 'Только NSFW' and not is_nsfw: continue
                                    filtered_res.append(item)
                                res = filtered_res
                                
                            state.aesthetic_results = res[:n]
                            state.sel_aes = {path: False for _, path, _ in state.aesthetic_results}
                            state.add_log("✅ Оценка эстетики завершена!")
                        except Exception as e: state.add_log(f"❌ Ошибка: {e}")
                        finally:
                            state.status_text = "Применение фильтров и рендеринг..."
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(bg_task)
                    aesthetic_gallery_ui.refresh()
                    btn_rate.enable()
                    state.status_text = "Готово!"
                    
                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10'):
                    btn_rate = ui.button('✨ Оценить', on_click=run_aesthetic_action).classes('w-full bg-yellow-600 hover:bg-yellow-500 font-bold text-lg')

            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                await aesthetic_gallery_ui()

        # ВКЛАДКА 3: NSFW
        with ui.tab_panel(tab_nsfw).classes('w-full h-[calc(100vh-115px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('NSFW Детектор').classes('text-lg font-bold')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto p-4 gap-2 min-h-0'):
                    with ui.row().classes('w-full items-start gap-1 flex-nowrap'):
                        nsfw_dir = ui.textarea('Папки (каждая с новой строки)', value=cfg.get('nsfw_dir', '')).classes('flex-grow').props('rows=2')
                        with ui.column().classes('gap-0 pt-2'):
                            ui.button(icon='create_new_folder', on_click=lambda: select_folder_multi(nsfw_dir)).props('flat round dense').tooltip('Добавить папку')
                            ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache_multi(nsfw_dir.value)).props('flat round dense text-color=red').tooltip('Очистить кэш этих папок')

                    with ui.row().classes('w-full gap-2'):
                        chk_img_nsfw = ui.checkbox('Картинки', value=cfg.get('chk_img_nsfw', True))
                        chk_vid_nsfw = ui.checkbox('Видео', value=cfg.get('chk_vid_nsfw', False))
                    
                    nsfw_model_sel = ui.select(['prithivMLmods/siglip2-x256-explicit-content', 'strangerguardhf/nsfw-image-detection'], value=cfg.get('nsfw_model', 'prithivMLmods/siglip2-x256-explicit-content'), label='Модель').classes('w-full text-xs')
                    
                    top_n_nsfw = ui.number('Оставить ТОП (шт)', value=cfg.get('top_n_nsfw', 100), format='%.0f').classes('w-full')
                    chk_prefix_nsfw = ui.checkbox('Писать Датчик Опасности в имя', value=cfg.get('chk_prefix_nsfw', False)).classes('text-sm text-gray-300 w-full mt-2')

                    with ui.expansion('Тонкие настройки', icon='tune').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-2'):
                        with ui.row().classes('w-full gap-2 px-2 pt-2'):
                            nsfw_batch_size = ui.number('Батч', value=cfg.get('nsfw_batch_size', 16), format='%.0f').classes('w-[45%]')
                            nsfw_video_frames = ui.number('Кадры', value=cfg.get('nsfw_video_frames', 4), format='%.0f').classes('w-[45%]')
                        with ui.row().classes('w-full gap-2 px-2 pb-2'):
                            nsfw_max_dim = ui.number('Лимит разр.', value=cfg.get('nsfw_max_dim', 512), format='%.0f').classes('w-[45%]')
                            nsfw_quant_mode = ui.select(['None', '8-bit', '4-bit'], value=cfg.get('nsfw_quant_mode', 'None'), label='Квант').classes('w-[45%]')
                
                async def run_nsfw_action():
                    save_config({
                        'nsfw_dir': nsfw_dir.value, 'chk_img_nsfw': chk_img_nsfw.value, 'chk_vid_nsfw': chk_vid_nsfw.value,
                        'nsfw_model': nsfw_model_sel.value, 'top_n_nsfw': top_n_nsfw.value, 
                        'nsfw_batch_size': nsfw_batch_size.value, 'nsfw_video_frames': nsfw_video_frames.value, 
                        'nsfw_max_dim': nsfw_max_dim.value, 'nsfw_quant_mode': nsfw_quant_mode.value,
                        'chk_prefix_nsfw': chk_prefix_nsfw.value
                    })
                    if not nsfw_dir.value: return ui.notify("Укажите папку!", type='warning')
                    
                    state.is_processing = True
                    search_engine.cancel_flag = False
                    state.nsfw_results.clear()
                    state.sel_nsfw.clear()
                    state.nsfw_page = 1
                    nsfw_gallery_ui.refresh()
                    btn_nsfw.disable()
                    search_engine._unload_embedding_model()
                    aesthetic_engine.unload()
                    face_engine.unload()
                    tag_engine.unload()
                    
                    exts =[]
                    if chk_img_nsfw.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid_nsfw.value: exts.extend(SUPPORTED_VIDEOS)

                    def bg_task():
                        try:
                            state.add_log(f"Запуск NSFW сканирования для папки: '{nsfw_dir.value}'")
                            nsfw_engine.batch_size = int(nsfw_batch_size.value)
                            nsfw_engine.video_frames = int(nsfw_video_frames.value)
                            nsfw_engine.max_dim = int(nsfw_max_dim.value)
                            nsfw_engine.quant_mode = nsfw_quant_mode.value
                            
                            state.nsfw_base_dir = nsfw_dir.value
                            n = int(top_n_nsfw.value)
                            
                            res = nsfw_engine.evaluate_media(nsfw_dir.value, nsfw_model_sel.value, tuple(exts))
                                
                            state.nsfw_results = res[:n]
                            state.sel_nsfw = {path: False for _, path, _, _ in state.nsfw_results}
                            state.add_log("✅ NSFW сканирование завершено!")
                        except Exception as e: state.add_log(f"❌ Ошибка: {e}")
                        finally:
                            state.status_text = "Применение фильтров и рендеринг..."
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(bg_task)
                    nsfw_gallery_ui.refresh()
                    btn_nsfw.enable()
                    state.status_text = "Готово!"
                    
                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10'):
                    btn_nsfw = ui.button('🚨 Анализ', on_click=run_nsfw_action).classes('w-full bg-red-800 hover:bg-red-700 font-bold text-lg')

            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                await nsfw_gallery_ui()

        # ВКЛАДКА 4: ПОИСК ПО ЛИЦУ (FACE SEARCH)
        with ui.tab_panel(tab_face).classes('w-full h-[calc(100vh-115px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('Поиск по лицу').classes('text-lg font-bold')

                with ui.column().classes('w-full flex-1 overflow-y-auto p-4 gap-2 min-h-0'):
                    with ui.row().classes('w-full items-start gap-1 flex-nowrap'):
                        face_dir = ui.textarea('Папки (каждая с новой строки)', value=cfg.get('face_dir', '')).classes('flex-grow').props('rows=2')
                        with ui.column().classes('gap-0 pt-2'):
                            ui.button(icon='create_new_folder', on_click=lambda: select_folder_multi(face_dir)).props('flat round dense').tooltip('Добавить папку')
                            ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache_multi(face_dir.value)).props('flat round dense text-color=red').tooltip('Очистить кэш этих папок')

                    with ui.row().classes('w-full items-center gap-1 flex-nowrap mt-2'):
                        ref_img = ui.input('Фото с лицом (Референс)', value=cfg.get('ref_img', '')).classes('flex-grow')
                        ui.button(icon='image', on_click=lambda: select_file(ref_img)).props('flat round dense')
                        
                    with ui.row().classes('w-full gap-2 mt-2'):
                        chk_img_face = ui.checkbox('Картинки', value=cfg.get('chk_img_face', True))
                        chk_vid_face = ui.checkbox('Видео (1-й кадр)', value=cfg.get('chk_vid_face', False))
                    
                    face_threshold = ui.number('Мин. Сходство (0.0 - 1.0)', value=cfg.get('face_threshold', 0.40), format='%.2f', step=0.05).classes('w-full mt-2')
                    chk_prefix_face = ui.checkbox('Писать Сходство в имя при копировании', value=cfg.get('chk_prefix_face', False)).classes('text-sm text-gray-300 w-full mt-2')

                    with ui.expansion('Тонкие настройки', icon='tune').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-2'):
                        with ui.row().classes('w-full gap-2 px-2 pt-2 pb-2'):
                            face_batch_size = ui.number('Батч', value=cfg.get('face_batch_size', 16), format='%.0f').classes('w-[45%]')

                async def run_face_action():
                    save_config({
                        'face_dir': face_dir.value, 'ref_img': ref_img.value,
                        'chk_img_face': chk_img_face.value, 'chk_vid_face': chk_vid_face.value,
                        'face_threshold': face_threshold.value, 'chk_prefix_face': chk_prefix_face.value,
                        'face_batch_size': face_batch_size.value
                    })
                    if not face_dir.value or not ref_img.value: return ui.notify("Укажите папку и референсное фото!", type='warning')
                    
                    state.is_processing = True
                    search_engine.cancel_flag = False
                    state.face_results.clear()
                    state.sel_face.clear()
                    state.face_page = 1
                    face_gallery_ui.refresh()
                    btn_face.disable()
                    
                    search_engine._unload_embedding_model()
                    aesthetic_engine.unload()
                    nsfw_engine.unload()
                    tag_engine.unload()
                    
                    exts =[]
                    if chk_img_face.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid_face.value: exts.extend(SUPPORTED_VIDEOS)

                    def bg_task():
                        try:
                            state.add_log(f"Запуск поиска лиц для папки: '{face_dir.value}'")
                            face_engine.batch_size = int(face_batch_size.value)
                            state.face_base_dir = face_dir.value
                            
                            res = face_engine.search_faces(ref_img.value, face_dir.value, tuple(exts), float(face_threshold.value))
                                
                            state.face_results = res
                            state.sel_face = {path: False for _, path in state.face_results}
                            state.add_log("✅ Поиск лиц завершен!")
                        except Exception as e: state.add_log(f"❌ Ошибка поиска лиц: {e}")
                        finally:
                            state.status_text = "Применение фильтров и рендеринг..."
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(bg_task)
                    face_gallery_ui.refresh()
                    btn_face.enable()
                    state.status_text = "Готово!"
                    
                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10'):
                    btn_face = ui.button('🕵️ Искать Лицо', on_click=run_face_action).classes('w-full bg-teal-600 hover:bg-teal-500 font-bold text-lg')

            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                await face_gallery_ui()

        # ВКЛАДКА 5: ТЕГИ (DANBOORU)
        with ui.tab_panel(tab_tags).classes('w-full h-[calc(100vh-115px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('Поиск по тегам').classes('text-lg font-bold')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto p-4 gap-2 min-h-0'):
                    with ui.row().classes('w-full items-start gap-1 flex-nowrap'):
                        tags_dir = ui.textarea('Папки (каждая с новой строки)', value=cfg.get('tags_dir', '')).classes('flex-grow').props('rows=2')
                        with ui.column().classes('gap-0 pt-2'):
                            ui.button(icon='create_new_folder', on_click=lambda: select_folder_multi(tags_dir)).props('flat round dense').tooltip('Добавить папку')
                            ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache_multi(tags_dir.value)).props('flat round dense text-color=red').tooltip('Очистить кэш этих папок')

                    with ui.row().classes('w-full gap-2'):
                        chk_img_tags = ui.checkbox('Картинки', value=cfg.get('chk_img_tags', True))
                        chk_vid_tags = ui.checkbox('Видео', value=cfg.get('chk_vid_tags', False))
                    
                    tags_model_sel = ui.select([
                        'SmilingWolf/wd-swinv2-tagger-v3',
                        'SmilingWolf/wd-convnext-tagger-v3',
                        'SmilingWolf/wd-eva02-large-tagger-v3',
                        'SmilingWolf/wd-vit-tagger-v3',
                        'Camais03/camie-tagger-v2',
                        'fancyfeast/joytag'
                    ], value=cfg.get('tags_model', 'SmilingWolf/wd-swinv2-tagger-v3'), label='Модель').classes('w-full text-xs')
                    
                    # Кнопка подгрузки тегов из БД
                    async def load_available_tags():
                        if not tags_dir.value: return ui.notify("Сначала выберите папку", type='warning')
                        
                        dir_val = tags_dir.value
                        exts =[]
                        if chk_img_tags.value: exts.extend(SUPPORTED_IMAGES)
                        if chk_vid_tags.value: exts.extend(SUPPORTED_VIDEOS)
                        
                        # Берем значение кадров из конфига, так как UI элемент tags_video_frames создается ниже
                        frames_val = int(cfg.get('tags_video_frames', 4))
                        cache_key = f"{tags_model_sel.value}_{frames_val}"
                        
                        ui.notify("🔄 Идет сбор тегов из БД, подождите...", type='info', timeout=2000)
                        
                        def process_tags(directory, extensions, key):
                            all_files = search_engine._gather_files(directory, tuple(extensions))
                            path_to_hash = search_engine.db_cache.get_or_create_hashes(all_files)
                            valid_hashes = set(path_to_hash.values())
                            unique_tags = {}
                            
                            c = search_engine.db_cache.conn.cursor()
                            c.execute("SELECT hash, tags FROM tags_cache WHERE model=?", (key,))
                            db_data = c.fetchall()
                            
                            for row in db_data:
                                h, tags_json = row[0], row[1]
                                if h in valid_hashes and tags_json:
                                    tags = json.loads(tags_json)
                                    for t in tags.keys(): unique_tags[t] = unique_tags.get(t, 0) + 1
                            return unique_tags

                        # Выполняем в фоне, чтобы не заблокировать веб-сервер и не потерять соединение
                        unique_tags = await run.io_bound(process_tags, dir_val, exts, cache_key)
                                
                        if not unique_tags:
                            return ui.notify("В базе нет тегов для этой папки. Нажмите 'Индексировать'.", type='warning')
                            
                        sorted_tags = sorted(unique_tags.keys(), key=lambda x: unique_tags[x], reverse=True)
                        
                        # Ограничиваем до 8000 тегов (этого хватит для 99% базы), а редкие можно вводить вручную
                        top_tags = sorted_tags[:8000]
                        pos_tags_sel.options = top_tags
                        neg_tags_sel.options = top_tags
                        pos_tags_sel.update()
                        neg_tags_sel.update()
                        ui.notify(f"Загружено {len(sorted_tags)} уникальных тегов (в списке ТОП-8000).", type='positive')

                    ui.button('🔄 Загрузить доступные теги из БД', on_click=load_available_tags).props('outline color=pink size=sm').classes('w-full')

                    # --- БЛОК LAZY ПОИСКА ---
                    ui.label('Частичное совпадение (Lazy Search)').classes('text-sm font-bold text-green-400 mt-2')
                    lazy_tags_input = ui.input('Например: girl black hair', value=cfg.get('lazy_tags', '')).classes('w-full')

                    # --- БЛОК ПОЗИТИВНЫХ ТЕГОВ ---
                    with ui.row().classes('w-full items-center justify-between mt-2 mb-[-12px]'):
                        ui.label('Включая (Positive - AND)').classes('text-sm font-bold text-blue-400')
                        ui.button('Очистить всё', on_click=lambda: pos_tags_sel.set_value([])).props('flat dense size=sm color=red')
                    
                    # Свойство hide-selected скроет теги внутри поля
                    pos_tags_sel = ui.select([], multiple=True, with_input=True, value=cfg.get('pos_tags',[])).classes('w-full').props('hide-selected new-value-mode=add-unique')
                    pos_tags_container = ui.row().classes('w-full gap-1 mt-1')
                    
                    def remove_pos_tag(tag):
                        pos_tags_sel.set_value([t for t in pos_tags_sel.value if t != tag])

                    def update_pos_tags(e=None):
                        pos_tags_container.clear()
                        with pos_tags_container:
                            for tag in pos_tags_sel.value:
                                ui.button(f"{tag} ✖", on_click=lambda _, t=tag: remove_pos_tag(t)) \
                                    .props('dense size=sm outline color=blue-300 no-caps') \
                                    .classes('rounded-full px-2 py-0 min-h-0 text-xs bg-blue-900/30')
                    
                    pos_tags_sel.on_value_change(update_pos_tags)
                    update_pos_tags()

                    # --- БЛОК НЕГАТИВНЫХ ТЕГОВ ---
                    with ui.row().classes('w-full items-center justify-between mt-4 mb-[-12px]'):
                        ui.label('Исключая (Negative - NOT)').classes('text-sm font-bold text-pink-400')
                        ui.button('Очистить всё', on_click=lambda: neg_tags_sel.set_value([])).props('flat dense size=sm color=red')
                        
                    neg_tags_sel = ui.select([], multiple=True, with_input=True, value=cfg.get('neg_tags',[])).classes('w-full').props('hide-selected new-value-mode=add-unique')
                    neg_tags_container = ui.row().classes('w-full gap-1 mt-1')

                    def remove_neg_tag(tag):
                        neg_tags_sel.set_value([t for t in neg_tags_sel.value if t != tag])

                    def update_neg_tags(e=None):
                        neg_tags_container.clear()
                        with neg_tags_container:
                            for tag in neg_tags_sel.value:
                                ui.button(f"{tag} ✖", on_click=lambda _, t=tag: remove_neg_tag(t)) \
                                    .props('dense size=sm outline color=pink-300 no-caps') \
                                    .classes('rounded-full px-2 py-0 min-h-0 text-xs bg-pink-900/30')
                                
                    neg_tags_sel.on_value_change(update_neg_tags)
                    update_neg_tags()
                    
                    tags_threshold = ui.number('Порог уверенности (0.1 - 1.0)', value=cfg.get('tags_threshold', 0.4), format='%.2f', step=0.05).classes('w-full mt-4')
                    chk_txt_tags = ui.checkbox('Сохранять .txt файл с тегами при копировании', value=cfg.get('chk_txt_tags', True)).classes('text-sm text-gray-300 w-full')

                    with ui.expansion('Тонкие настройки', icon='tune').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-2'):
                        with ui.row().classes('w-full gap-2 px-2 pt-2'):
                            tags_batch_size = ui.number('Батч', value=cfg.get('tags_batch_size', 8), format='%.0f').classes('w-[45%]')
                            tags_video_frames = ui.number('Кадры', value=cfg.get('tags_video_frames', 4), format='%.0f').classes('w-[45%]')

                async def index_tags_action():
                    save_config({
                        'tags_dir': tags_dir.value, 'chk_img_tags': chk_img_tags.value, 'chk_vid_tags': chk_vid_tags.value,
                        'tags_model': tags_model_sel.value, 'tags_batch_size': tags_batch_size.value, 
                        'tags_video_frames': tags_video_frames.value
                    })
                    if not tags_dir.value: return ui.notify("Укажите папку!", type='warning')
                    
                    state.is_processing = True
                    search_engine.cancel_flag = False
                    btn_index_tags.disable()
                    btn_search_tags.disable()
                    
                    search_engine._unload_embedding_model()
                    aesthetic_engine.unload()
                    nsfw_engine.unload()
                    face_engine.unload()
                    
                    exts =[]
                    if chk_img_tags.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid_tags.value: exts.extend(SUPPORTED_VIDEOS)

                    def bg_task():
                        try:
                            state.add_log(f"Индексация тегов для папки: '{tags_dir.value}'")
                            tag_engine.batch_size = int(tags_batch_size.value)
                            tag_engine.video_frames = int(tags_video_frames.value)
                            tag_engine.evaluate_media(tags_dir.value, tags_model_sel.value, tuple(exts))
                            state.add_log("✅ Индексация тегов завершена!")
                        except Exception as e: state.add_log(f"❌ Ошибка: {e}")
                        finally:
                            state.status_text = "Готово!"
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(bg_task)
                    btn_index_tags.enable()
                    btn_search_tags.enable()

                async def search_tags_action():
                    save_config({
                        'tags_dir': tags_dir.value, 'pos_tags': pos_tags_sel.value, 'neg_tags': neg_tags_sel.value,
                        'tags_threshold': tags_threshold.value, 'chk_txt_tags': chk_txt_tags.value,
                        'lazy_tags': lazy_tags_input.value
                    })
                    if not tags_dir.value: return ui.notify("Укажите папку!", type='warning')
                    
                    state.is_processing = True
                    btn_search_tags.disable()
                    state.tags_base_dir = tags_dir.value
                    state.tags_results.clear()
                    state.sel_tags.clear()
                    state.tags_page = 1
                    tags_gallery_ui.refresh()
                    
                    dir_val = tags_dir.value
                    thres_val = float(tags_threshold.value)
                    pos_val = set(pos_tags_sel.value)
                    neg_val = set(neg_tags_sel.value)
                    lazy_val = lazy_tags_input.value.strip().lower()
                    cache_key = f"{tags_model_sel.value}_{int(tags_video_frames.value)}"
                    
                    exts =[]
                    if chk_img_tags.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid_tags.value: exts.extend(SUPPORTED_VIDEOS)

                    def process_search(directory, extensions, key, thres, pos, neg, lazy_str):
                        all_files = search_engine._gather_files(directory, tuple(extensions))
                        path_to_hash = search_engine.db_cache.get_or_create_hashes(all_files)
                        hash_to_path = {h: p for p, h in path_to_hash.items()}
                        
                        c = search_engine.db_cache.conn.cursor()
                        c.execute("SELECT hash, tags FROM tags_cache WHERE model=?", (key,))
                        db_data = c.fetchall()
                        
                        res = []
                        lazy_words =[w for w in lazy_str.replace(',', ' ').split() if w] if lazy_str else[]
                        
                        for row in db_data:
                            h, tags_json = row[0], row[1]
                            if h not in hash_to_path or not tags_json: continue
                            path = hash_to_path[h]
                            
                            tags = json.loads(tags_json)
                            valid = True
                            
                            # 1. Точные позитивные теги (AND)
                            for pt in pos:
                                if pt not in tags or tags[pt] < thres:
                                    valid = False; break
                            if not valid: continue
                            
                            # 2. Точные негативные теги (NOT)
                            for nt in neg:
                                if nt in tags and tags[nt] >= thres:
                                    valid = False; break
                            if not valid: continue
                            
                            # 3. Lazy Search (Частичное совпадение)
                            lazy_score = 0.0
                            if lazy_words:
                                # Оставляем теги выше порога, заменяем '_' на пробел для гибкости поиска
                                valid_tags_dict = {t.lower().replace('_', ' '): prob for t, prob in tags.items() if prob >= thres}
                                # Объединяем все теги изображения в единую строку для сверхбыстрого поиска
                                joined_tags = " | ".join(valid_tags_dict.keys())
                                
                                # Проверяем, что ВСЕ введенные слова (girl, black, hair) присутствуют в тегах
                                for w in lazy_words:
                                    if w not in joined_tags:
                                        valid = False
                                        break
                                if not valid: continue
                                
                                # Подсчитываем Score: суммируем вероятности тех тегов, внутри которых нашлись наши lazy слова
                                matched_probs =[prob for t, prob in valid_tags_dict.items() if any(w in t for w in lazy_words)]
                                if matched_probs:
                                    lazy_score = sum(matched_probs)
                            
                            # Итоговый Score для сортировки
                            if pos:
                                score = sum([tags[pt] for pt in pos]) + lazy_score
                            elif lazy_words:
                                score = lazy_score
                            else:
                                score = max(tags.values()) if tags else 0
                                
                            res.append((score, path, tags))
                            
                        res.sort(key=lambda x: x[0], reverse=True)
                        return res

                    # Вызов функции в отдельном потоке (передаем lazy_val)
                    res = await run.io_bound(process_search, dir_val, exts, cache_key, thres_val, pos_val, neg_val, lazy_val)
                    
                    state.tags_results = res
                    state.sel_tags = {p: False for s, p, t in res}
                    state.status_text = "Применение фильтров и рендеринг..."
                    state.is_processing = False
                    tags_gallery_ui.refresh()
                    btn_search_tags.enable()
                    state.status_text = "Готово!"
                    
                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10 gap-2'):
                    btn_index_tags = ui.button('🔍 Индексировать', on_click=index_tags_action).classes('w-full bg-gray-700 hover:bg-gray-600 font-bold')
                    btn_search_tags = ui.button('🎯 Искать', on_click=search_tags_action).classes('w-full bg-pink-700 hover:bg-pink-600 font-bold')

            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                await tags_gallery_ui()

        # ВКЛАДКА 6: ИНДЕКСАТОР (Кэш)
        with ui.tab_panel(tab_cache).classes('w-full h-[calc(100vh-115px)] p-8 flex flex-col items-center overflow-y-auto pb-24'):
            with ui.card().classes('w-full max-w-[600px] p-6 flex flex-col gap-4 bg-gray-900 border border-gray-800 shrink-0 mb-12 mt-4'):
                ui.label('Массовая Индексация (Предкэширование)').classes('text-xl font-bold text-blue-400')
                ui.label('Используйте это, чтобы заранее проанализировать всю папку без необходимости выполнять сам поиск. Это сохранит все нейросетевые признаки в базу данных.').classes('text-gray-400 text-sm')
                
                with ui.row().classes('w-full items-center gap-2 mt-2'):
                    cache_dir = ui.input('Папка для индексации', value=cfg.get('cache_dir', '')).classes('flex-grow')
                    ui.button(icon='folder', on_click=lambda: select_folder(cache_dir)).props('flat round dense')
                    ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache(cache_dir.value)).props('flat round dense text-color=red').tooltip('Очистить индекс файлов папки')
                
                with ui.row().classes('w-full gap-2 border-b border-gray-800 pb-4'):
                    chk_cache_img = ui.checkbox('Картинки', value=cfg.get('chk_cache_img', True))
                    chk_cache_vid = ui.checkbox('Видео', value=cfg.get('chk_cache_vid', True))
                    chk_cache_txt = ui.checkbox('Текст (только для поиска)', value=cfg.get('chk_cache_txt', False))

                ui.label('Выберите, что кэшировать:').classes('font-bold mt-2')
                
                chk_cache_search = ui.checkbox('Умный Поиск (Qwen Embeddings)', value=cfg.get('chk_cache_search', True)).classes('text-md font-bold text-blue-300')
                with ui.row().classes('w-full pl-6 pr-6 items-center gap-2').bind_visibility_from(chk_cache_search, 'value'):
                    emb_model_cache = ui.select(['Qwen/Qwen3-VL-Embedding-2B', 'Qwen/Qwen3-VL-Embedding-8B'], value=cfg.get('emb_model', 'Qwen/Qwen3-VL-Embedding-2B')).classes('flex-1')
                    cache_search_quant = ui.select(['None', '8-bit', '4-bit'], value=cfg.get('search_quant_mode', 'None'), label='Квант').classes('w-24')
                
                chk_cache_aes = ui.checkbox('Оценка Эстетики', value=cfg.get('chk_cache_aes', True)).classes('text-md font-bold text-yellow-300')
                with ui.row().classes('w-full pl-6 pr-6 items-center gap-2').bind_visibility_from(chk_cache_aes, 'value'):
                    cache_aes_quant = ui.select(['None', '8-bit', '4-bit'], value=cfg.get('aes_quant_mode', 'None'), label='Квант').classes('w-24')
                
                chk_cache_nsfw = ui.checkbox('NSFW Детектор', value=cfg.get('chk_cache_nsfw', True)).classes('text-md font-bold text-red-300')
                with ui.row().classes('w-full pl-6 pr-6 items-center gap-2').bind_visibility_from(chk_cache_nsfw, 'value'):
                    nsfw_model_cache = ui.select(['prithivMLmods/siglip2-x256-explicit-content', 'strangerguardhf/nsfw-image-detection'], value=cfg.get('nsfw_model', 'prithivMLmods/siglip2-x256-explicit-content')).classes('flex-1')
                    cache_nsfw_quant = ui.select(['None', '8-bit', '4-bit'], value=cfg.get('nsfw_quant_mode', 'None'), label='Квант').classes('w-24')

                chk_cache_tags = ui.checkbox('Тегирование (Danbooru)', value=cfg.get('chk_cache_tags', True)).classes('text-md font-bold text-pink-300')
                with ui.row().classes('w-full pl-6 pr-6 items-center gap-2').bind_visibility_from(chk_cache_tags, 'value'):
                    tags_model_cache = ui.select([
                        'SmilingWolf/wd-swinv2-tagger-v3', 'SmilingWolf/wd-convnext-tagger-v3', 
                        'SmilingWolf/wd-eva02-large-tagger-v3', 'SmilingWolf/wd-vit-tagger-v3', 
                        'Camais03/camie-tagger-v2', 'fancyfeast/joytag'
                    ], value=cfg.get('tags_model', 'SmilingWolf/wd-swinv2-tagger-v3')).classes('flex-1')

                chk_cache_face = ui.checkbox('Поиск по лицу (InsightFace)', value=cfg.get('chk_cache_face', False)).classes('text-md font-bold text-teal-300')

                with ui.expansion('Единые тонкие настройки', icon='tune').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-4'):
                    with ui.row().classes('w-full gap-4 px-4 pt-4'):
                        cache_batch_size = ui.number('Размер Батча', value=cfg.get('batch_size', 16), format='%.0f').classes('flex-1')
                        cache_video_frames = ui.number('Кадров видео', value=cfg.get('video_frames', 4), format='%.0f').classes('flex-1')
                    with ui.row().classes('w-full gap-4 px-4 pb-4'):
                        cache_max_dim = ui.number('Лимит разрешения (размер)', value=cfg.get('emb_size', 512), format='%.0f').classes('flex-1')
                
                with ui.expansion('Оптимизация памяти (ОЗУ)', icon='memory').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-2'):
                    with ui.column().classes('w-full gap-2 px-4 py-4'):
                        ui.label('Эти настройки предотвращают вылет программы (OOM) при больших папках.').classes('text-gray-400 text-xs')
                        use_ram_compression = ui.checkbox('Сжатие кэша в ОЗУ', value=cfg.get('use_ram_compression', False)).classes('text-sm text-green-400 font-bold')
                        ui.label('Блочная архитектура (Радикально спасает ОЗУ)').classes('font-bold text-sm mt-2')
                        cache_chunk_size = ui.number('Размер блока файлов (0 = выключено, рекомендуемое = 2000)', value=cfg.get('cache_chunk_size', 2000), format='%.0f').classes('w-full')

                async def run_cache_action():
                    save_config({
                        'cache_dir': cache_dir.value, 'chk_cache_img': chk_cache_img.value, 
                        'chk_cache_vid': chk_cache_vid.value, 'chk_cache_txt': chk_cache_txt.value,
                        'chk_cache_search': chk_cache_search.value, 'chk_cache_aes': chk_cache_aes.value, 
                        'chk_cache_nsfw': chk_cache_nsfw.value, 'chk_cache_face': chk_cache_face.value,
                        'chk_cache_tags': chk_cache_tags.value,
                        'search_quant_mode': cache_search_quant.value, 'aes_quant_mode': cache_aes_quant.value,
                        'nsfw_quant_mode': cache_nsfw_quant.value,
                        'use_ram_compression': use_ram_compression.value, 'cache_chunk_size': cache_chunk_size.value
                    })
                    if not cache_dir.value: return ui.notify("Укажите папку!", type='warning')
                    
                    state.is_processing = True
                    search_engine.cancel_flag = False
                    btn_cache.disable()
                    
                    def bg_task():
                        try:
                            state.add_log(f"Начало полного цикла кэширования для директории: '{cache_dir.value}'")
                            
                            # Настройка ОЗУ-кэшера (Вариант 1)
                            media_cache.enabled = True
                            media_cache.compress = use_ram_compression.value
                            chunk_size = int(cache_chunk_size.value)
                            
                            exts_search =[]
                            if chk_cache_img.value: exts_search.extend(SUPPORTED_IMAGES)
                            if chk_cache_vid.value: exts_search.extend(SUPPORTED_VIDEOS)
                            if chk_cache_txt.value: exts_search.extend(SUPPORTED_TEXTS)
                            
                            exts_media =[]
                            if chk_cache_img.value: exts_media.extend(SUPPORTED_IMAGES)
                            if chk_cache_vid.value: exts_media.extend(SUPPORTED_VIDEOS)

                            search_engine.emb_size = int(cache_max_dim.value)
                            search_engine.video_frames = int(cache_video_frames.value)
                            search_engine.quant_mode = cache_search_quant.value
                            
                            aesthetic_engine.batch_size = int(cache_batch_size.value)
                            aesthetic_engine.max_dim = int(cache_max_dim.value)
                            aesthetic_engine.video_frames = int(cache_video_frames.value)
                            aesthetic_engine.quant_mode = cache_aes_quant.value
                            
                            nsfw_engine.batch_size = int(cache_batch_size.value)
                            nsfw_engine.max_dim = int(cache_max_dim.value)
                            nsfw_engine.video_frames = int(cache_video_frames.value)
                            nsfw_engine.quant_mode = cache_nsfw_quant.value
                            
                            tag_engine.batch_size = max(1, int(cache_batch_size.value) // 2) # Теггеры едят больше VRAM, чуть урежем
                            tag_engine.video_frames = int(cache_video_frames.value)

                            face_engine.batch_size = int(cache_batch_size.value)

                            all_allowed_exts = tuple(set(exts_search + exts_media))
                            all_files_for_index = search_engine._gather_files(cache_dir.value, all_allowed_exts)
                            
                            if not all_files_for_index:
                                state.add_log("⚠️ Не найдено подходящих файлов для кэширования.")
                                return

                            if chunk_size > 0:
                                chunks =[all_files_for_index[i:i + chunk_size] for i in range(0, len(all_files_for_index), chunk_size)]
                            else:
                                chunks =[all_files_for_index]

                            state.add_log(f"Всего файлов: {len(all_files_for_index)}. Разобьем на {len(chunks)} блок(ов).")

                            for idx, chunk in enumerate(chunks):
                                if search_engine.cancel_flag: break
                                if len(chunks) > 1:
                                    state.add_log(f"🔄 === ОБРАБОТКА БЛОКА {idx+1}/{len(chunks)} ({len(chunk)} файлов) === 🔄")
                                
                                # Очистка кэша ОЗУ перед каждым новым блоком (Вариант 3)
                                media_cache.clear()

                                if chk_cache_search.value and not search_engine.cancel_flag:
                                    if len(chunks) > 1: state.add_log(f"-> Блок {idx+1}: Кэширование Умного поиска")
                                    else: state.add_log(f"-> Этап 1: Кэширование Умного поиска")
                                    nsfw_engine.unload()
                                    aesthetic_engine.unload()
                                    face_engine.unload()
                                    tag_engine.unload()
                                    search_engine.build_cache(cache_dir.value, emb_model_cache.value, int(cache_batch_size.value), tuple(exts_search), override_files=chunk)
                                    
                                if chk_cache_aes.value and not search_engine.cancel_flag:
                                    if len(chunks) > 1: state.add_log(f"-> Блок {idx+1}: Оценка Эстетики")
                                    else: state.add_log(f"-> Этап 2: Оценка Эстетики")
                                    search_engine._unload_embedding_model()
                                    nsfw_engine.unload()
                                    face_engine.unload()
                                    tag_engine.unload()
                                    aesthetic_engine.evaluate_media(cache_dir.value, tuple(exts_media), override_files=chunk)
                                    
                                if chk_cache_nsfw.value and not search_engine.cancel_flag:
                                    if len(chunks) > 1: state.add_log(f"-> Блок {idx+1}: NSFW Детектор")
                                    else: state.add_log(f"-> Этап 3: NSFW Детектор")
                                    search_engine._unload_embedding_model()
                                    aesthetic_engine.unload()
                                    face_engine.unload()
                                    tag_engine.unload()
                                    nsfw_engine.evaluate_media(cache_dir.value, nsfw_model_cache.value, tuple(exts_media), override_files=chunk)

                                if chk_cache_tags.value and not search_engine.cancel_flag:
                                    if len(chunks) > 1: state.add_log(f"-> Блок {idx+1}: Тегирование (Danbooru)")
                                    else: state.add_log(f"-> Этап 4: Тегирование (Danbooru)")
                                    search_engine._unload_embedding_model()
                                    aesthetic_engine.unload()
                                    face_engine.unload()
                                    nsfw_engine.unload()
                                    tag_engine.evaluate_media(cache_dir.value, tags_model_cache.value, tuple(exts_media), override_files=chunk)

                                if chk_cache_face.value and not search_engine.cancel_flag:
                                    if len(chunks) > 1: state.add_log(f"-> Блок {idx+1}: Кэширование Лиц (InsightFace)")
                                    else: state.add_log(f"-> Этап 5: Кэширование Лиц (InsightFace)")
                                    search_engine._unload_embedding_model()
                                    aesthetic_engine.unload()
                                    nsfw_engine.unload()
                                    tag_engine.unload()
                                    face_engine.build_cache(cache_dir.value, tuple(exts_media), override_files=chunk)

                            state.add_log("🎉 Полная индексация успешно завершена!")
                        except Exception as e: state.add_log(f"❌ Ошибка индексации: {e}")
                        finally:
                            media_cache.enabled = False
                            media_cache.compress = False
                            media_cache.clear()
                            state.is_processing = False
                            state.progress = 1.0
                            state.status_text = "Готово!"

                    await run.io_bound(bg_task)
                    btn_cache.enable()

                btn_cache = ui.button('🚀 Запустить полное кэширование', on_click=run_cache_action).classes('w-full bg-blue-600 hover:bg-blue-500 font-bold text-lg mt-4')
        
        # ВКЛАДКА: ДУБЛИКАТЫ
        with ui.tab_panel(tab_dupes).classes('w-full h-[calc(100vh-115px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('Поиск дубликатов').classes('text-lg font-bold')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto p-4 gap-2 min-h-0'):
                    with ui.row().classes('w-full items-start gap-1 flex-nowrap'):
                        dupes_dir = ui.textarea('Папки (каждая с новой строки)', value=cfg.get('dupes_dir', '')).classes('flex-grow').props('rows=3')
                        with ui.column().classes('gap-0 pt-2'):
                            ui.button(icon='create_new_folder', on_click=lambda: select_folder_multi(dupes_dir)).props('flat round dense').tooltip('Добавить папку')
                            ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache_multi(dupes_dir.value)).props('flat round dense text-color=red').tooltip('Очистить кэш')
                    
                    saved_dupes_mode = cfg.get('dupes_mode', 'Точные (Быстрый Хеш)')
                    if saved_dupes_mode == 'Похожие картинки (pHash)':  # Миграция со старого конфига
                        saved_dupes_mode = 'Похожие картинки и ВИДЕО (pHash)'
                        
                    dupes_mode = ui.select(['Точные (Быстрый Хеш)', 'Похожие картинки и ВИДЕО (pHash)'], value=saved_dupes_mode, label='Режим поиска').classes('w-full mt-2 text-lg font-bold')
                    phash_threshold = ui.number('Порог СХОЖЕСТИ (0-15, больше = шире допуск)', value=cfg.get('phash_threshold', 4), format='%.0f').classes('w-full mt-2 text-orange-400 font-bold')
                    
                    dupes_video_frames = ui.select([
                        '1 кадр (Самое начало 0%)',
                        '1 кадр (Середина 50%)',
                        '3 кадра (0%, 50%, 100%)',
                        '5 кадров (Равномерно)',
                        '10 кадров (Равномерно)'
                    ], value=cfg.get('dupes_video_frames', '3 кадра (0%, 50%, 100%)'), label='Глубина анализа видео (pHash)').classes('w-full mt-2 font-bold text-orange-300')
                    
                    with ui.row().classes('w-full gap-2 mt-2 mb-2'):
                        chk_img_dupes = ui.checkbox('Картинки', value=cfg.get('chk_img_dupes', True))
                        chk_vid_dupes = ui.checkbox('Видео', value=cfg.get('chk_vid_dupes', True))

                    def update_dupes_visibility(e=None):
                        is_phash = (dupes_mode.value != 'Точные (Быстрый Хеш)')
                        phash_threshold.set_visibility(is_phash)
                        dupes_video_frames.set_visibility(is_phash)
                        
                    dupes_mode.on_value_change(update_dupes_visibility)
                    update_dupes_visibility()

                async def run_dupes_action():
                    save_config({
                        'dupes_dir': dupes_dir.value, 'dupes_mode': dupes_mode.value,
                        'phash_threshold': phash_threshold.value, 'dupes_video_frames': dupes_video_frames.value,
                        'chk_img_dupes': chk_img_dupes.value, 'chk_vid_dupes': chk_vid_dupes.value
                    })
                    if not dupes_dir.value: return ui.notify("Укажите папки!", type='warning')
                    
                    state.is_processing = True
                    search_engine.cancel_flag = False
                    state.dupes_results.clear()
                    state.sel_dupes.clear()
                    setattr(state, 'dupes_page', 1)
                    dupes_gallery_ui.refresh()
                    btn_dupes.disable()
                    
                    exts =[]
                    if chk_img_dupes.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid_dupes.value: exts.extend(SUPPORTED_VIDEOS)

                    def bg_task():
                        try:
                            state.add_log(f"Запуск поиска дубликатов. Режим: {dupes_mode.value}")
                            if dupes_mode.value == 'Точные (Быстрый Хеш)':
                                res = dupes_engine.find_exact(dupes_dir.value, tuple(exts))
                            else:
                                res = dupes_engine.find_similar(dupes_dir.value, tuple(exts), int(phash_threshold.value), dupes_video_frames.value)
                                
                            state.dupes_results = res
                            for group in res:
                                for p in group:
                                    state.sel_dupes[p] = False
                                    
                            state.add_log(f"✅ Поиск дубликатов завершен! Найдено групп: {len(res)}")
                        except Exception as e: 
                            state.add_log(f"❌ Ошибка поиска дубликатов: {e}")
                        finally:
                            state.status_text = "Применение фильтров и рендеринг..."
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(bg_task)
                    dupes_gallery_ui.refresh()
                    btn_dupes.enable()
                    state.status_text = "Готово!"
                    
                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10'):
                    btn_dupes = ui.button('👯 Найти Дубликаты', on_click=run_dupes_action).classes('w-full bg-orange-700 hover:bg-orange-600 font-bold text-lg')

            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                await dupes_gallery_ui()

        # ВКЛАДКА: AI КЛАСТЕРИЗАЦИЯ
        with ui.tab_panel(tab_cluster).classes('w-full h-[calc(100vh-115px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('Магия Кластеризации').classes('text-lg font-bold')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto p-4 gap-2 min-h-0'):
                    ui.label('ВНИМАНИЕ: Сначала проиндексируйте файлы через "Умный поиск" или "Индексатор" (Умный поиск + Тегирование).').classes('text-xs text-orange-400 font-bold mb-2')
                    
                    with ui.row().classes('w-full items-start gap-1 flex-nowrap'):
                        cluster_dir = ui.textarea('Папки (с новой строки)', value=cfg.get('cluster_dir', '')).classes('flex-grow').props('rows=2')
                        with ui.column().classes('gap-0 pt-2'):
                            ui.button(icon='create_new_folder', on_click=lambda: select_folder_multi(cluster_dir)).props('flat round dense').tooltip('Добавить папку')
                            ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache_multi(cluster_dir.value)).props('flat round dense text-color=red').tooltip('Очистить кэш')
                    
                    cluster_algo = ui.select(['K-Means', 'DBSCAN'], value=cfg.get('cluster_algo', 'K-Means'), label='Алгоритм (K-Means - кол-во папок, DBSCAN - авто)').classes('w-full mt-2 font-bold')
                    cluster_k = ui.number('Желаемое кол-во папок (K-Means)', value=cfg.get('cluster_k', 10), format='%.0f').classes('w-full')
                    cluster_eps = ui.number('Чувствительность (DBSCAN 0.1 - 1.0)', value=cfg.get('cluster_eps', 0.4), format='%.2f', step=0.05).classes('w-full')
                    
                    with ui.row().classes('w-full gap-2 mt-2 mb-2'):
                        chk_img_cluster = ui.checkbox('Картинки', value=cfg.get('chk_img_cluster', True))
                        chk_vid_cluster = ui.checkbox('Видео', value=cfg.get('chk_vid_cluster', False))
                        
                    chk_align_domains = ui.checkbox('Слияние Картинка ↔ Видео (Сдвиг ИИ-векторов)', value=cfg.get('chk_align_domains', True)).classes('text-sm text-purple-300 font-bold mb-2').tooltip('Устраняет системную разницу в понимании нейросетью фото и видео, заставляя их попадать в одни папки')
                        
                    cluster_emb_model = ui.select(['Qwen/Qwen3-VL-Embedding-2B', 'Qwen/Qwen3-VL-Embedding-8B'], value=cfg.get('cluster_emb_model', 'Qwen/Qwen3-VL-Embedding-2B'), label='Модель (из которой брать вектора)').classes('w-full text-xs')
                    cluster_emb_size = ui.number('Разрешение при кэше', value=cfg.get('emb_size', 512), format='%.0f').classes('w-full')

                    with ui.expansion('Тонкие настройки ИИ-Инференса (Если файлов нет в БД)', icon='tune').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-2'):
                        with ui.row().classes('w-full gap-2 px-2 pt-2'):
                            cluster_batch_size = ui.number('Батч', value=cfg.get('cluster_batch_size', 16), format='%.0f').classes('w-[45%]')
                            cluster_video_frames = ui.number('Кадры видео', value=cfg.get('cluster_video_frames', 4), format='%.0f').classes('w-[45%]')
                        with ui.row().classes('w-full gap-2 px-2 pb-2'):
                            cluster_quant_mode = ui.select(['None', '8-bit', '4-bit'], value=cfg.get('cluster_quant_mode', 'None'), label='Квант').classes('w-[45%]')

                    def update_cluster_visibility(e=None):
                        is_kmeans = (cluster_algo.value == 'K-Means')
                        cluster_k.set_visibility(is_kmeans)
                        cluster_eps.set_visibility(not is_kmeans)
                        
                    cluster_algo.on_value_change(update_cluster_visibility)
                    update_cluster_visibility()

                async def execute_cluster_action(action='copy'):
                    selected_paths =[p for p, checked in state.sel_cluster.items() if checked]
                    if not selected_paths: return ui.notify('Ничего не выбрано!', type='warning')
                        
                    base_dest = await run.io_bound(pick_folder_native)
                    if not base_dest: return
                    
                    ui.notify(f"Начато {action} для кластеров...", type='info')

                    def _process_cluster_files():
                        success = 0
                        moved_paths = set()
                        
                        for cluster in state.cluster_results:
                            cluster_name = cluster["name"]
                            dest_folder = os.path.join(base_dest, cluster_name)
                            
                            for path in cluster["paths"]:
                                if state.sel_cluster.get(path):
                                    os.makedirs(dest_folder, exist_ok=True)
                                    fname = os.path.basename(path)
                                    dest = os.path.join(dest_folder, fname)
                                    
                                    # Защита, если исходный файл и цель совпадают
                                    if os.path.abspath(path) == os.path.abspath(dest):
                                        continue
                                    
                                    try:
                                        if action == 'copy':
                                            shutil.copy2(path, dest)
                                        else:
                                            shutil.move(path, dest)
                                            moved_paths.add(path)
                                        success += 1
                                    except Exception as e: state.add_log(f"Ошибка {path}: {e}")
                        return success, moved_paths
                        
                    success, moved_paths = await run.io_bound(_process_cluster_files)
                                    
                    ui.notify(f'Успешно {action}: {success} файлов', type='positive')
                    
                    if action == 'move' and moved_paths:
                        new_clusters =[]
                        for c in state.cluster_results:
                            new_paths = [item for item in c["paths"] if item not in moved_paths]
                            if new_paths: new_clusters.append({"name": c["name"], "paths": new_paths})
                        state.cluster_results = new_clusters
                        cluster_gallery_ui.refresh()

                async def run_cluster_action():
                    save_config({
                        'cluster_dir': cluster_dir.value, 'cluster_algo': cluster_algo.value,
                        'cluster_k': cluster_k.value, 'cluster_eps': cluster_eps.value,
                        'chk_img_cluster': chk_img_cluster.value, 'chk_vid_cluster': chk_vid_cluster.value,
                        'cluster_emb_model': cluster_emb_model.value, 'chk_align_domains': chk_align_domains.value
                    })
                    if not cluster_dir.value: return ui.notify("Укажите папки!", type='warning')
                    
                    state.is_processing = True
                    search_engine.cancel_flag = False
                    state.cluster_results.clear()
                    state.sel_cluster.clear()
                    setattr(state, 'cluster_page', 1)
                    cluster_gallery_ui.refresh()
                    btn_cluster.disable()
                    
                    exts =[]
                    if chk_img_cluster.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid_cluster.value: exts.extend(SUPPORTED_VIDEOS)

                    def bg_task():
                        try:
                            state.add_log(f"Начат процесс AI сортировки...")
                            res = cluster_engine.build_clusters(
                                cluster_dir.value, tuple(exts), cluster_algo.value,
                                int(cluster_k.value), float(cluster_eps.value),
                                cluster_emb_model.value, int(cluster_emb_size.value),
                                int(cluster_batch_size.value), int(cluster_video_frames.value), 
                                cluster_quant_mode.value, chk_align_domains.value
                            )
                            state.cluster_results = res
                            for cluster in res:
                                for p in cluster["paths"]: state.sel_cluster[p] = False
                            state.add_log(f"✅ Кластеризация завершена! Сформировано папок: {len(res)}")
                        except Exception as e: state.add_log(f"❌ Ошибка: {e}")
                        finally:
                            state.status_text = "Применение фильтров и рендеринг..."
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(bg_task)
                    cluster_gallery_ui.refresh()
                    btn_cluster.enable()
                    state.status_text = "Готово!"
                    
                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10'):
                    btn_cluster = ui.button('✨ Раскидать по папкам', on_click=run_cluster_action).classes('w-full bg-purple-700 hover:bg-purple-600 font-bold text-lg')

            @ui.refreshable
            async def cluster_gallery_ui():
                if not state.cluster_results:
                    ui.label("Здесь появятся сгруппированные нейросетью файлы...").classes("text-gray-400 m-4")
                    return
                
                await asyncio.sleep(0.001)

                GROUPS_PER_PAGE = int(state.groups_per_page)
                MAX_ITEMS_PER_GROUP = 30  # В свернутом виде
                ITEMS_PER_PAGE_CLUSTER = 60 # Во внутреннем развернутом виде (безопасно для DOM)
                
                # Инициализация состояния: кто развернут и на какой внутренней странице находится
                if not hasattr(state, 'expanded_clusters'):
                    state.expanded_clusters = set()
                if not hasattr(state, 'expanded_pages'):
                    state.expanded_pages = {}
                
                total_pages = max(1, (len(state.cluster_results) + GROUPS_PER_PAGE - 1) // GROUPS_PER_PAGE)
                if getattr(state, 'cluster_page', 1) > total_pages: state.cluster_page = 1
                
                def change_page(d):
                    state.cluster_page = max(1, min(total_pages, getattr(state, 'cluster_page', 1) + d))
                    cluster_gallery_ui.refresh()

                with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
                    with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                        with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                            with ui.row().classes('gap-2 items-center'):
                                ui.button('Выбрать всё', on_click=lambda: ui.timer(0, lambda: set_all('cluster', True), once=True)).props('outline color=white dense')
                                ui.button('Снять всё', on_click=lambda: ui.timer(0, lambda: set_all('cluster', False), once=True)).props('outline color=white dense')
                            with ui.row().classes('gap-2 items-center'):
                                ui.button('Копировать по папкам ✔', icon='content_copy', on_click=lambda: execute_cluster_action('copy')).props('color=purple-800 text-white font-bold dense')
                                ui.button('Переместить по папкам ✔', icon='drive_file_move', on_click=lambda: execute_cluster_action('move')).props('color=purple-600 text-white font-bold dense')
                                ui.button('УДАЛИТЬ ✔', icon='delete_forever', on_click=lambda: delete_items([p for p, c in state.sel_cluster.items() if c], 'cluster')).props('color=red-10 text-white dense')
                        
                        with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                            ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                            ui.label(f'Страница {getattr(state, "cluster_page", 1)} из {total_pages}').classes('text-gray-300 font-bold')
                            ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')
                    
                    scroll_id = 'cluster_scroll_area'
                    with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                        start_idx = (getattr(state, 'cluster_page', 1) - 1) * GROUPS_PER_PAGE
                        page_groups = state.cluster_results[start_idx : start_idx + GROUPS_PER_PAGE]
                        
                        def render_cluster_group(group):
                            cluster_name = group["name"]
                            paths = group["paths"]
                            is_expanded = {'val': False}
                            inner_page = {'val': 1}
                            
                            with ui.card().classes('w-full bg-gray-800 border border-gray-700 p-2 mb-4'):
                                with ui.row().classes('w-full justify-between items-center px-2 mb-2'):
                                    ui.label(f'📁 {cluster_name} (Всего файлов: {len(paths)})').classes('font-bold text-purple-400')
                                    
                                    with ui.row().classes('gap-2 items-center'):
                                        def select_cluster(val):
                                            for p in paths: state.sel_cluster[p] = val
                                            update_view()

                                        ui.button('Выделить группу', on_click=lambda: select_cluster(True)).props('outline size=sm color=green')
                                        ui.button('Снять выделение', on_click=lambda: select_cluster(False)).props('outline size=sm color=red')
                                        
                                        btn_toggle = ui.button('Развернуть', on_click=lambda: toggle_expand()).props('size=sm color=gray')
                                        if len(paths) <= MAX_ITEMS_PER_GROUP:
                                            btn_toggle.set_visibility(False)

                                content_container = ui.column().classes('w-full p-0 m-0')

                                def toggle_expand():
                                    is_expanded['val'] = not is_expanded['val']
                                    inner_page['val'] = 1
                                    btn_toggle.text = 'Свернуть' if is_expanded['val'] else 'Развернуть'
                                    btn_toggle._props['color'] = 'purple' if is_expanded['val'] else 'gray'
                                    btn_toggle.update()
                                    update_view()

                                def update_view():
                                    content_container.clear()
                                    with content_container:
                                        if is_expanded['val']:
                                            start_i = (inner_page['val'] - 1) * ITEMS_PER_PAGE_CLUSTER
                                            end_i = start_i + ITEMS_PER_PAGE_CLUSTER
                                            visible_group = paths[start_i:end_i]
                                            row_cls = 'w-full gap-4 pb-2 items-start flex-wrap'
                                        else:
                                            visible_group = paths[:MAX_ITEMS_PER_GROUP]
                                            row_cls = 'w-full gap-4 pb-2 items-start overflow-x-auto flex-nowrap'
                                            
                                        hidden_count = 0 if is_expanded['val'] else len(paths) - MAX_ITEMS_PER_GROUP
                                        
                                        with ui.row().classes(row_cls):
                                            for path in visible_group:
                                                safe_path = urllib.parse.quote(path)
                                                local_index = paths.index(path)
                                                
                                                with ui.column().classes('w-[200px] shrink-0 relative bg-gray-900 rounded overflow-hidden border border-gray-700 hover:border-purple-500 transition-colors'):
                                                    with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                                        ui.checkbox().bind_value(state.sel_cluster, path).on('click', lambda e, i=local_index, p=path, pts=paths: handle_shift_click(e, i, p, 'cluster', pts),['shiftKey'])
                                                    
                                                    if path.lower().endswith(SUPPORTED_VIDEOS):
                                                        ui.label('▶ ВИДЕО').classes('absolute top-2 right-2 bg-blue-600/90 text-white text-[10px] font-bold px-1.5 py-0.5 rounded z-10 pointer-events-none shadow')

                                                    with ui.context_menu():
                                                        ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                                        ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))
                                                        ui.separator()
                                                        ui.menu_item('Удалить файл', on_click=lambda p=path: delete_items([p], 'cluster')).classes('text-red-400')

                                                    ui.image(f"/thumb/{safe_path}").classes('w-full h-[150px] object-contain cursor-pointer bg-black').props('fit=contain loading="lazy"').on('click', lambda e, idx=local_index, pts=paths: open_media(idx, pts))
                                                    
                                                    with ui.column().classes('p-2 gap-0 w-full'):
                                                        ui.label(os.path.basename(path)).classes('text-gray-400 text-[10px] truncate w-full').tooltip(path)

                                            if hidden_count > 0 and not is_expanded['val']:
                                                with ui.card().classes('w-[200px] h-[190px] shrink-0 flex flex-col items-center justify-center bg-gray-900 border border-dashed border-gray-600 gap-2 p-4 cursor-pointer hover:border-purple-500 transition-colors').on('click', toggle_expand):
                                                    ui.icon('more_horiz', size='3rem').classes('text-gray-500')
                                                    ui.label(f"+ еще {hidden_count} шт.").classes('text-center font-bold text-gray-300 text-lg')
                                                    ui.label("Нажмите, чтобы развернуть").classes('text-[10px] text-center text-purple-400')

                                        if is_expanded['val'] and len(paths) > ITEMS_PER_PAGE_CLUSTER:
                                            tot_c_pages = max(1, (len(paths) + ITEMS_PER_PAGE_CLUSTER - 1) // ITEMS_PER_PAGE_CLUSTER)
                                            
                                            def change_c_page(d):
                                                inner_page['val'] = max(1, min(tot_c_pages, inner_page['val'] + d))
                                                update_view()

                                            with ui.row().classes('w-full justify-center items-center gap-4 py-2 border-t border-gray-700 mt-4'):
                                                ui.button(icon='chevron_left', on_click=lambda: change_c_page(-1)).props('flat outline color=purple size=sm')
                                                ui.label(f'Под-страница {inner_page["val"]} из {tot_c_pages}').classes('text-gray-400 text-xs font-bold')
                                                ui.button(icon='chevron_right', on_click=lambda: change_c_page(1)).props('flat outline color=purple size=sm')

                                update_view() # Первичная отрисовка

                        for group in page_groups:
                            render_cluster_group(group)

                    ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=purple-800').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')
            
            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                await cluster_gallery_ui()

    with ui.footer().classes('bg-gray-900 border-t border-gray-800 px-4 py-0 flex flex-row flex-nowrap items-center justify-between z-40 h-8 shadow-lg'):
        ui.label().bind_text_from(state, 'status_text').classes('text-blue-400 font-mono text-xs truncate max-w-[30%] shrink-0')
        ui.linear_progress(value=0, show_value=False).bind_value_from(state, 'progress').classes('flex-grow mx-4 h-1.5 rounded text-blue-600')
        
        ui.button('ПРЕРВАТЬ', icon='cancel', on_click=cancel_all_tasks) \
            .props('color=red size=sm dense outline') \
            .classes('shrink-0 py-0 min-h-0 text-xs font-bold mr-2 bg-red-900/20') \
            .bind_visibility_from(state, 'is_processing')
            
        ui.button('ЛОГИ', icon='terminal', on_click=log_drawer.toggle).props('flat text-color=white size=sm dense').classes('shrink-0 py-0 min-h-0 text-xs')

    ui.timer(0.5, update_ui_logs)

if __name__ in {"__main__", "__mp_main__"}:
    parser = argparse.ArgumentParser(description="AI Media Organizer Pro")
    parser.add_argument('--server-only', action='store_true', help='Запустить в режиме сервера (без локального окна)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='IP адрес для сервера')
    parser.add_argument('--port', type=int, default=8190, help='Порт сервера')
    
    args, unknown = parser.parse_known_args()

    if args.server_only:
        print(f"🌐 Режим сервера активирован. Откройте в браузере: http://{args.host}:{args.port}")
        # native=False отключает десктопное окно, show=False предотвращает автоматическое открытие вкладки
        ui.run(title="AI Media Organizer Pro", host=args.host, port=args.port, native=False, show=False, dark=True, reload=False, reconnect_timeout=30.0)
    else:
        # Стандартный оконный (Native) режим
        ui.run(title="AI Media Organizer Pro", port=args.port, native=True, dark=True, window_size=(1400, 900), reload=False, reconnect_timeout=30.0)
