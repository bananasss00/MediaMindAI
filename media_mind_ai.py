try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
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
    print("✅ SageAttention успешно активирован и подменил SDPA!")

except Exception as e:
    print(f"⚠️ SageAttention недоступен. Используем стандартный PyTorch SDPA.")
    print(f"   (Детали: {e})")

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
class DatabaseCache:
    def __init__(self, db_path='image_cache.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        # Включаем WAL (Write-Ahead Logging) для быстрой работы без блокировок
        self.conn.execute("PRAGMA journal_mode=WAL") 
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # Выделяем 256 МБ ОЗУ под кэш SQLite (по умолчанию там смешные крохи)
        self.conn.execute("PRAGMA cache_size=-262144") 
        # Разрешаем проецировать базу в оперативную память (до 2 ГБ)
        self.conn.execute("PRAGMA mmap_size=2147483648") 
        self.conn.execute("PRAGMA temp_store=MEMORY")

        self._init_tables()

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS emb_cache (model TEXT, path TEXT, features BLOB, PRIMARY KEY (model, path))''')
        c.execute('''CREATE TABLE IF NOT EXISTS rerank_cache_v2 (model TEXT, query TEXT, path TEXT, score REAL, PRIMARY KEY (model, query, path))''')
        c.execute('''CREATE TABLE IF NOT EXISTS aes_cache (model TEXT, path TEXT, avg_score REAL, max_score REAL, PRIMARY KEY (model, path))''')
        c.execute('''CREATE TABLE IF NOT EXISTS sim_cache (model TEXT, query TEXT, path TEXT, score REAL, PRIMARY KEY (model, query, path))''')
        c.execute('''CREATE TABLE IF NOT EXISTS nsfw_cache (model TEXT, path TEXT, top_label TEXT, danger_score REAL, details TEXT, PRIMARY KEY (model, path))''')
        
        # Создаем индексы для мгновенного поиска по пути файла (без Full Table Scan)
        c.execute('CREATE INDEX IF NOT EXISTS idx_nsfw_path ON nsfw_cache(path)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_emb_path ON emb_cache(path)')

        self.conn.commit()

    # --- NSFW ---
    def get_nsfw_score(self, model_name, path):
        c = self.conn.cursor()
        c.execute("SELECT top_label, danger_score, details FROM nsfw_cache WHERE model=? AND path=?", (model_name, path))
        return c.fetchone()

    def save_nsfw_score(self, model_name, path, top_label, danger_score, details):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO nsfw_cache (model, path, top_label, danger_score, details) VALUES (?, ?, ?, ?, ?)", 
                  (model_name, path, top_label, danger_score, json.dumps(details)))
        self.conn.commit()

    # --- Поиск и Эстетика ---
    def get_query_sims(self, model_name, query):
        c = self.conn.cursor()
        c.execute("SELECT path, score FROM sim_cache WHERE model=? AND query=?", (model_name, query))
        return {row[0]: row[1] for row in c.fetchall()}

    def save_query_sims(self, model_name, query, paths, scores):
        c = self.conn.cursor()
        data =[(model_name, query, p, s) for p, s in zip(paths, scores)]
        c.executemany("INSERT OR REPLACE INTO sim_cache (model, query, path, score) VALUES (?, ?, ?, ?)", data)
        self.conn.commit()

    def get_aesthetic_score(self, model_name, path):
        c = self.conn.cursor()
        c.execute("SELECT avg_score, max_score FROM aes_cache WHERE model=? AND path=?", (model_name, path))
        return c.fetchone()

    def save_aesthetic_score(self, model_name, path, avg_score, max_score):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO aes_cache (model, path, avg_score, max_score) VALUES (?, ?, ?, ?)", 
                  (model_name, path, avg_score, max_score))
        self.conn.commit()

    def get_image_features(self, model_name, path):
        c = self.conn.cursor()
        c.execute("SELECT features FROM emb_cache WHERE model=? AND path=?", (model_name, path))
        result = c.fetchone()
        if result is not None:
            return torch.load(io.BytesIO(result[0]), weights_only=False)
        return None

    def save_image_features(self, model_name, path, features):
        c = self.conn.cursor()
        features_bytes = io.BytesIO()
        torch.save(features, features_bytes)
        c.execute("INSERT OR REPLACE INTO emb_cache (model, path, features) VALUES (?, ?, ?)", 
                  (model_name, path, features_bytes.getvalue()))
        self.conn.commit()

    def get_rerank_score(self, model_name, query, path):
        c = self.conn.cursor()
        c.execute("SELECT score FROM rerank_cache_v2 WHERE model=? AND query=? AND path=?", (model_name, query, path))
        result = c.fetchone()
        return result[0] if result is not None else None

    def save_rerank_score(self, model_name, query, path, score):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO rerank_cache_v2 (model, query, path, score) VALUES (?, ?, ?, ?)", 
                  (model_name, query, path, score))
        self.conn.commit()

    def get_max_danger_score(self, path):
        c = self.conn.cursor()
        # Ищем файл в кэше любых NSFW-моделей и берем максимальную оценку опасности
        c.execute("SELECT MAX(danger_score) FROM nsfw_cache WHERE path=?", (path,))
        res = c.fetchone()
        return res[0] if res and res[0] is not None else -1.0  # -1.0 означает, что файла нет в базе

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

# ОПТИМИЗАЦИЯ ИНДЕКСАТОРА: Совместный кэш изображений в RAM для всех моделей в рамках одного цикла
class MediaCache:
    def __init__(self):
        self.enabled = False
        self.compress = False  # Флаг для сжатия в ОЗУ (Вариант 1)
        self.cache = {}

    def clear(self):
        self.cache.clear()
        gc.collect()

    def _compress_img(self, img):
        buf = io.BytesIO()
        # Сжимаем в JPEG прямо в памяти (качество 85% оптимально)
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _decompress_img(self, bytes_data):
        # Декодируем обратно в сырые пиксели при чтении
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
            return [self._decompress_img(b) for b in cached] if self.compress else cached
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
                self.cache[cache_key] = [self._compress_img(img) for img in resized] if self.compress else resized
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

    def _gather_files(self, dir_path, allowed_exts):
        files_list = self.files_cache.list_files(dir_path)
        if files_list is None:
            self.log(f"Индексация файловой системы: {dir_path}...")
            all_supported = SUPPORTED_IMAGES + SUPPORTED_VIDEOS + SUPPORTED_TEXTS
            files_list =[]
            for root, dirs, files in os.walk(dir_path):
                if self.cancel_flag: break
                for file in files:
                    if file.lower().endswith(all_supported):
                        files_list.append(os.path.join(root, file))
            if not self.cancel_flag:
                self.files_cache._data[dir_path] = files_list
                self.files_cache.save_cache()
                self.log(f"Найдено поддерживаемых файлов: {len(files_list)}")
        return[f for f in files_list if f.lower().endswith(allowed_exts)]

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
        """ Метод для предкэширования всех медиа-файлов в папке (без выполнения поиска) """
        self.cancel_flag = False
        files_list = self._gather_files(dir_path, allowed_exts) if override_files is None else[f for f in override_files if f.lower().endswith(allowed_exts)]
        cache_key = emb_model_name if self.emb_size == 512 else f"{emb_model_name}_{self.emb_size}"
        
        paths_to_compute =[]
        for i, fp in enumerate(files_list):
            if self.cancel_flag: break
            if self.db_cache.get_image_features(cache_key, fp) is None:
                paths_to_compute.append(fp)
                
        if not paths_to_compute or self.cancel_flag:
            self.log("Кэш эмбеддингов полностью актуален.")
            return

        # LAZY LOADING: Загружаем модель только если есть новые файлы для обработки
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
                    c_paths, c_docs = [],[]
                    c_weight = 0
                    
                    for path, doc, weight in items:
                        if c_weight + weight > batch_size and len(c_docs) > 0:
                            try:
                                feats_batch = model.encode(c_docs, batch_size=len(c_docs), convert_to_tensor=True).cpu()
                                for p, feats in zip(c_paths, feats_batch):
                                    self.db_cache.save_image_features(cache_key, p, feats)
                            except Exception as e: self.log(f"Ошибка батча эмбеддингов: {e}")
                            
                            processed_count += len(c_paths)
                            self.progress(processed_count / total, f"Кэш эмбеддингов ({processed_count}/{total})...")
                            c_paths, c_docs = [],[]
                            c_weight = 0
                            
                        c_paths.append(path)
                        c_docs.append(doc)
                        c_weight += weight
                        
                    if len(c_docs) > 0 and not self.cancel_flag:
                        try:
                            feats_batch = model.encode(c_docs, batch_size=len(c_docs), convert_to_tensor=True).cpu()
                            for p, feats in zip(c_paths, feats_batch):
                                self.db_cache.save_image_features(cache_key, p, feats)
                        except Exception as e: self.log(f"Ошибка батча эмбеддингов: {e}")
                            
                        processed_count += len(c_paths)
                        self.progress(processed_count / total, f"Кэш эмбеддингов ({processed_count}/{total})...")

    def phase1_recall(self, dir_path, raw_query, query_input, top_k, emb_model_name, batch_size, allowed_exts):
        self.cancel_flag = False
        files_list = self._gather_files(dir_path, allowed_exts)
        
        results_phase1 =[]
        cache_key = emb_model_name if self.emb_size == 512 else f"{emb_model_name}_{self.emb_size}"
        cached_sims = self.db_cache.get_query_sims(cache_key, raw_query)
        
        self.log(f"Фильтрация {len(files_list)} файлов через кэш...")
        paths_needing_sims =[]
        paths_needing_features =[]
        
        for i, file_path in enumerate(files_list):
            if self.cancel_flag: break
            if file_path in cached_sims:
                results_phase1.append((cached_sims[file_path], file_path))
            else:
                paths_needing_sims.append(file_path)
                
            if i % 500 == 0: 
                prog = 0.1 * (i / max(1, len(files_list)))
                self.progress(prog, f"Чтение кэша ({i}/{len(files_list)})...")
                
        # ⚡ Если все результаты уже в кэше — возвращаем мгновенно, не трогая VRAM!
        if not paths_needing_sims or self.cancel_flag:
            self.log("⚡ Запрос полностью закэширован! Обход загрузки модели.")
            results_phase1.sort(key=lambda x: x[0], reverse=True)
            self.progress(0.8, "Поиск завершен.") 
            return results_phase1[:top_k]

        # Иначе загружаем модель для создания вектора (эмбеддинга) запроса
        model = self._get_embedding_model(emb_model_name)
        self.log("Конвертация запроса в эмбеддинг...")
        query_emb = model.encode(query_input, convert_to_tensor=True).cpu()

        # Проверяем, есть ли уже фичи картинок для тех файлов, где нет симиларов
        sims_to_save_paths = []
        sims_to_save_scores = []
        
        for file_path in paths_needing_sims:
            if self.cancel_flag: break
            features = self.db_cache.get_image_features(cache_key, file_path)
            if features is not None:
                sim = float(util.cos_sim(query_emb, features).item())
                results_phase1.append((sim, file_path))
                
                # Собираем данные в списки вместо сохранения по одному
                sims_to_save_paths.append(file_path)
                sims_to_save_scores.append(sim)
            else: 
                paths_needing_features.append(file_path)

        # Сохраняем все вычисленные симилары ОДНИМ запросом к диску (ускорение в 100+ раз)
        if sims_to_save_paths and not self.cancel_flag:
            self.db_cache.save_query_sims(cache_key, raw_query, sims_to_save_paths, sims_to_save_scores)

        if not paths_needing_features or self.cancel_flag:
            results_phase1.sort(key=lambda x: x[0], reverse=True)
            self.progress(0.8, "Поиск завершен.") 
            return results_phase1[:top_k]

        # Если дошли сюда, значит есть файлы, для которых нужно инференсить фичи
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
                        self.db_cache.save_query_sims(cache_key, raw_query, [path], [0.0])
                
                for size_key, items in buckets.items():
                    if self.cancel_flag: break
                    c_paths, c_docs = [],[]
                    c_weight = 0
                    
                    for path, doc, weight in items:
                        if c_weight + weight > batch_size and len(c_docs) > 0:
                            try:
                                feats_batch = model.encode(c_docs, batch_size=len(c_docs), convert_to_tensor=True).cpu()
                                sims_to_save =[]
                                for p, feats in zip(c_paths, feats_batch):
                                    self.db_cache.save_image_features(cache_key, p, feats)
                                    sim = float(util.cos_sim(query_emb, feats).item())
                                    sims_to_save.append(sim)
                                    results_phase1.append((sim, p))
                                self.db_cache.save_query_sims(cache_key, raw_query, c_paths, sims_to_save)
                            except Exception as e: self.log(f"Ошибка батча: {e}")
                            
                            processed_count += len(c_paths)
                            self.progress(0.1 + 0.7 * (processed_count / total), f"Инференс ({processed_count}/{total})...")
                            c_paths, c_docs = [],[]
                            c_weight = 0
                            
                        c_paths.append(path)
                        c_docs.append(doc)
                        c_weight += weight
                        
                    if len(c_docs) > 0 and not self.cancel_flag:
                        try:
                            feats_batch = model.encode(c_docs, batch_size=len(c_docs), convert_to_tensor=True).cpu()
                            sims_to_save =[]
                            for p, feats in zip(c_paths, feats_batch):
                                self.db_cache.save_image_features(cache_key, p, feats)
                                sim = float(util.cos_sim(query_emb, feats).item())
                                sims_to_save.append(sim)
                                results_phase1.append((sim, p))
                            self.db_cache.save_query_sims(cache_key, raw_query, c_paths, sims_to_save)
                        except Exception as e: self.log(f"Ошибка батча (остаток): {e}")
                            
                        processed_count += len(c_paths)
                        self.progress(0.1 + 0.7 * (processed_count / total), f"Инференс ({processed_count}/{total})...")

        results_phase1.sort(key=lambda x: x[0], reverse=True)
        return results_phase1[:top_k]

    def phase2_rerank(self, raw_query, query_input, top_candidates, min_score, rerank_model_name):
        if not top_candidates or self.cancel_flag: return top_candidates
        cache_key = rerank_model_name if self.rerank_size == 800 else f"{rerank_model_name}_{self.rerank_size}"
        
        final_results =[]
        docs_to_compute, paths_to_compute =[],[]
        for i, (score, fp) in enumerate(top_candidates):
            if self.cancel_flag: break
            cached_score = self.db_cache.get_rerank_score(cache_key, raw_query, fp)
            if cached_score is not None:
                if cached_score >= min_score: final_results.append((cached_score, fp))
            else:
                doc, _, _ = self._load_and_prep_file(fp, 'rerank') 
                if doc is not None:
                    docs_to_compute.append(doc)
                    paths_to_compute.append(fp)
                    
        if docs_to_compute and not self.cancel_flag:
            self._unload_embedding_model() # LAZY UNLOAD: освобождаем память только если нужен Reranker
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
                    self.db_cache.save_rerank_score(cache_key, raw_query, fp, s)
                    if s >= min_score: final_results.append((s, fp))
                    
                processed += len(c_docs)
                self.progress(0.8 + 0.2 * (processed / len_total), f"Rerank ({processed}/{len_total})...")
                
            del reranker
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        final_results.sort(key=lambda x: x[0], reverse=True)
        return final_results

# ==========================================
# 3. ДВИЖКИ ЭСТЕТИКИ И NSFW
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

    def load_model(self):
        if self.model is None:
            state.add_log(f"Загрузка модели Aesthetic Predictor на {self.device}...")
            # Принудительная скачка модели в локальную папку моделей
            kwargs = {
                "low_cpu_mem_usage": True, 
                "trust_remote_code": True,
                "cache_dir": os.path.join(current_dir, "models")
            }
            if self.device == "cuda":
                kwargs["attn_implementation"] = "sdpa"
                
            self.model, self.preprocessor = convert_v2_5_from_siglip(**kwargs)
            self.model = self.model.to(self.dtype).to(self.device)
            self.model.eval()

    def unload(self):
        if self.model is not None:
            state.add_log(f"Выгрузка Aesthetic Predictor из VRAM...")
            del self.model
            del self.preprocessor
            self.model = None
            self.preprocessor = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def evaluate_media(self, directory_path, allowed_exts, override_files=None):
        all_files = self.se._gather_files(directory_path, allowed_exts) if override_files is None else[f for f in override_files if f.lower().endswith(allowed_exts)]
        image_paths =[p for p in all_files if p.lower().endswith(SUPPORTED_IMAGES)]
        video_paths =[p for p in all_files if p.lower().endswith(SUPPORTED_VIDEOS)]
        
        state.add_log(f"Найдено для оценки: {len(image_paths)} изображений, {len(video_paths)} видео.")
        results =[]
        cache_key_img = "v2_5_siglip"
        cache_key_vid = "v2_5_siglip_vid_" + str(self.video_frames)

        # Подготовка: фильтрация через кэш (LAZY LOADING)
        images_to_process =[]
        for p in image_paths:
            cached = self.db_cache.get_aesthetic_score(cache_key_img, p)
            if cached is not None:
                results.append((cached[0], p, cached[1]))
            else:
                images_to_process.append(p)
                
        videos_to_process =[]
        for p in video_paths:
            cached = self.db_cache.get_aesthetic_score(cache_key_vid, p)
            if cached is not None:
                results.append((cached[0], p, cached[1]))
            else:
                videos_to_process.append(p)
                
        # Если все есть в базе, модель даже не грузим в VRAM
        if images_to_process or videos_to_process:
            self.load_model()
        else:
            results.sort(key=lambda x: x[0], reverse=True)
            return results

        # --- ОБРАБОТКА ИЗОБРАЖЕНИЙ ---
        batch_images, batch_paths =[],[]
        for i, img_path in enumerate(images_to_process):
            if not state.is_processing: break
            state.status_text = f"Подготовка фото: {Path(img_path).name} ({i+1}/{len(images_to_process)})"
            try:
                image = media_cache.get_image(img_path, self.max_dim)
                if image:
                    batch_images.append(image)
                    batch_paths.append(img_path)
                else:
                    self.db_cache.save_aesthetic_score(cache_key_img, img_path, 0.0, 0.0)
                    results.append((0.0, img_path, 0.0))
            except Exception as e: 
                state.add_log(f"Ошибка {img_path}: {e}")
                self.db_cache.save_aesthetic_score(cache_key_img, img_path, 0.0, 0.0)
                results.append((0.0, img_path, 0.0))
                
            if len(batch_images) >= self.batch_size or (i == len(images_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(images_to_process))
                state.status_text = f"Инференс фото ({i+1}/{len(images_to_process)})..."
                try:
                    pixel_values = self.preprocessor(images=batch_images, return_tensors="pt").pixel_values.to(self.dtype).to(self.device)
                    with torch.inference_mode():
                        logits = self.model(pixel_values).logits.flatten().float().cpu().tolist()
                    for score, p in zip(logits, batch_paths):
                        self.db_cache.save_aesthetic_score(cache_key_img, p, score, score)
                        results.append((score, p, score))
                except Exception as e: state.add_log(f"Ошибка инференса: {e}")
                batch_images, batch_paths = [],[]

        # --- ОБРАБОТКА ВИДЕО ---
        batch_images, batch_frame_counts, batch_paths =[], [],[]
        for i, vid_path in enumerate(videos_to_process):
            time.sleep(0.002)
            if not state.is_processing: break
            state.status_text = f"Подготовка видео: {Path(vid_path).name} ({i+1}/{len(videos_to_process)})"
            try:
                frames = media_cache.get_video_frames(vid_path, self.max_dim, self.video_frames)
                if frames:
                    batch_images.extend(frames)
                    batch_paths.append(vid_path)
                    batch_frame_counts.append(len(frames))
                else:
                    self.db_cache.save_aesthetic_score(cache_key_vid, vid_path, 0.0, 0.0)
                    results.append((0.0, vid_path, 0.0))
            except Exception as e: 
                state.add_log(f"Ошибка чтения {vid_path}: {e}")
                self.db_cache.save_aesthetic_score(cache_key_vid, vid_path, 0.0, 0.0)
                results.append((0.0, vid_path, 0.0))
                
            if len(batch_images) >= self.batch_size or (i == len(videos_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(videos_to_process))
                state.status_text = f"Инференс видео ({i+1}/{len(videos_to_process)})..."
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
                            self.db_cache.save_aesthetic_score(cache_key_vid, path, avg_s, max_s)
                            results.append((avg_s, path, max_s))
                except Exception as e: state.add_log(f"Ошибка инференса: {e}")
                batch_images, batch_frame_counts, batch_paths = [],[],[]

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

    def load_model(self, model_name):
        if self.model is None or self.current_model_name != model_name:
            self.unload()
            state.add_log(f"Загрузка NSFW модели {model_name} на {self.device}...")
            
            local_dir = os.path.join(current_dir, "models", model_name.replace("/", "_"))
            if not os.path.exists(local_dir) or not os.listdir(local_dir):
                state.add_log(f"Скачивание модели {model_name}...")
                snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
                
                if ["strangerguardhf", "prithivmlmods"] in model_name.lower():
                    for item in os.listdir(local_dir):
                        if item.startswith("checkpoint-"):
                            chk_path = os.path.join(local_dir, item)
                            if os.path.isdir(chk_path):
                                try:
                                    shutil.rmtree(chk_path)
                                    state.add_log(f"Удален лишний чекпоинт: {item}")
                                except Exception as e:
                                    state.add_log(f"Не удалось удалить {item}: {e}")
            
            self.processor = AutoImageProcessor.from_pretrained(local_dir)
            if "siglip" in model_name.lower():
                self.model = SiglipForImageClassification.from_pretrained(local_dir)
            else:
                self.model = AutoModelForImageClassification.from_pretrained(local_dir)
            self.model.to(self.dtype).to(self.device)
            self.model.eval()
            self.current_model_name = model_name

    def unload(self):
        if self.model is not None:
            state.add_log(f"Выгрузка NSFW модели {self.current_model_name} из VRAM...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.current_model_name = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def compute_danger(self, details):
        """ Высчитывает 'вероятность опасности', игнорируя нейтральные классы для любой модели """
        safe_labels = {'safe', 'sfw', 'normal', 'general', 'neutral', 'drawing', 'safe_content', 'anime picture', 'anime'}
        return sum(prob for lbl, prob in details.items() if lbl.lower() not in safe_labels)

    def evaluate_media(self, directory_path, model_name, allowed_exts, override_files=None):
        all_files = self.se._gather_files(directory_path, allowed_exts) if override_files is None else[f for f in override_files if f.lower().endswith(allowed_exts)]
        image_paths =[p for p in all_files if p.lower().endswith(SUPPORTED_IMAGES)]
        video_paths =[p for p in all_files if p.lower().endswith(SUPPORTED_VIDEOS)]
        
        state.add_log(f"Найдено для NSFW детектора: {len(image_paths)} изображений, {len(video_paths)} видео.")
        results =[]
        cache_key = f"{model_name}_{self.video_frames}"

        # Подготовка: фильтрация через кэш (LAZY LOADING)
        images_to_process =[]
        for p in image_paths:
            cached = self.db_cache.get_nsfw_score(cache_key, p)
            if cached is not None:
                details_dict = json.loads(cached[2]) if cached[2] else {}
                results.append((cached[1], p, cached[0], details_dict))
            else:
                images_to_process.append(p)

        videos_to_process =[]
        for p in video_paths:
            cached = self.db_cache.get_nsfw_score(cache_key, p)
            if cached is not None:
                details_dict = json.loads(cached[2]) if cached[2] else {}
                results.append((cached[1], p, cached[0], details_dict))
            else:
                videos_to_process.append(p)
                
        # Если все есть в базе, модель не грузим в VRAM
        if images_to_process or videos_to_process:
            self.load_model(model_name)
        else:
            results.sort(key=lambda x: x[0], reverse=True)
            return results

        # --- ИЗОБРАЖЕНИЯ ---
        batch_images, batch_paths = [],[]
        for i, img_path in enumerate(images_to_process):
            if not state.is_processing: break
            state.status_text = f"NSFW Фото: {Path(img_path).name} ({i+1}/{len(images_to_process)})"
            try:
                image = media_cache.get_image(img_path, self.max_dim)
                if image:
                    batch_images.append(image)
                    batch_paths.append(img_path)
                else:
                    self.db_cache.save_nsfw_score(cache_key, img_path, "error", 0.0, {"error": 1.0})
                    results.append((0.0, img_path, "error", {"error": 1.0}))
            except Exception as e: 
                state.add_log(f"Ошибка {img_path}: {e}")
                self.db_cache.save_nsfw_score(cache_key, img_path, "error", 0.0, {"error": 1.0})
                results.append((0.0, img_path, "error", {"error": 1.0}))
                
            if len(batch_images) >= self.batch_size or (i == len(images_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(images_to_process))
                state.status_text = f"Инференс NSFW фото ({i+1}/{len(images_to_process)})..."
                try:
                    inputs = self.processor(images=batch_images, return_tensors="pt")
                    inputs = {k: v.to(self.dtype).to(self.device) if v.is_floating_point() else v.to(self.device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        logits = self.model(**inputs).logits
                    probs = torch.nn.functional.softmax(logits, dim=-1).cpu()
                    
                    for j, p in enumerate(batch_paths):
                        prob_dist = probs[j]
                        top_idx = prob_dist.argmax(-1).item()
                        top_label = self.model.config.id2label[top_idx]
                        details = {self.model.config.id2label[idx]: float(val) for idx, val in enumerate(prob_dist)}
                        danger = self.compute_danger(details)
                        
                        self.db_cache.save_nsfw_score(cache_key, p, top_label, danger, details)
                        results.append((danger, p, top_label, details))
                except Exception as e: state.add_log(f"Ошибка инференса: {e}")
                batch_images, batch_paths = [],[]

        # --- ВИДЕО ---
        batch_images, batch_frame_counts, batch_paths =[], [],[]
        for i, vid_path in enumerate(videos_to_process):
            time.sleep(0.002)
            if not state.is_processing: break
            state.status_text = f"NSFW Видео: {Path(vid_path).name} ({i+1}/{len(videos_to_process)})"
            try:
                frames = media_cache.get_video_frames(vid_path, self.max_dim, self.video_frames)
                if frames:
                    batch_images.extend(frames)
                    batch_paths.append(vid_path)
                    batch_frame_counts.append(len(frames))
                else:
                    self.db_cache.save_nsfw_score(cache_key, vid_path, "error", 0.0, {"error": 1.0})
                    results.append((0.0, vid_path, "error", {"error": 1.0}))
            except Exception as e: 
                state.add_log(f"Ошибка чтения {vid_path}: {e}")
                self.db_cache.save_nsfw_score(cache_key, vid_path, "error", 0.0, {"error": 1.0})
                results.append((0.0, vid_path, "error", {"error": 1.0}))

            if len(batch_images) >= self.batch_size or (i == len(videos_to_process) - 1 and batch_images):
                state.progress = (i + 1) / max(1, len(videos_to_process))
                state.status_text = f"Инференс NSFW видео ({i+1}/{len(videos_to_process)})..."
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
                        top_idx = avg_probs.argmax(-1).item()
                        top_label = self.model.config.id2label[top_idx]
                        details = {self.model.config.id2label[k]: float(val) for k, val in enumerate(avg_probs)}
                        danger = self.compute_danger(details)
                        
                        self.db_cache.save_nsfw_score(cache_key, p, top_label, danger, details)
                        results.append((danger, p, top_label, details))
                except Exception as e: state.add_log(f"Ошибка инференса: {e}")
                batch_images, batch_frame_counts, batch_paths = [], [],[]

        results.sort(key=lambda x: x[0], reverse=True)
        return results

# ==========================================
# 4. СОСТОЯНИЕ И UI УТИЛИТЫ
# ==========================================
class AppState:
    def __init__(self):
        self.search_results =[]
        self.aesthetic_results =[]
        self.nsfw_results =[]
        
        self.sel_search = {}
        self.sel_aes = {}
        self.sel_nsfw = {}
        
        self.search_page = 1
        self.aes_page = 1
        self.nsfw_page = 1
        
        self.search_base_dir = ""
        self.aes_base_dir = ""
        self.nsfw_base_dir = ""
        
        self.viewer_open = False
        self.viewer_items =[]
        self.viewer_index = 0
        
        self.is_processing = False
        self.progress = 0.0
        self.status_text = "Готов к работе"
        
        self.logs =[]
        self.full_log_history =[]
        self.current_tab = 'Search'

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
            try:
                import win32clipboard
                from PIL import Image
                import io
                img = Image.open(path).convert('RGB')
                output = io.BytesIO()
                img.save(output, 'BMP')
                data = output.getvalue()[14:]
                output.close()
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
            except ImportError:
                # Фоллбэк на PowerShell, если pywin32 не установлен
                abs_path = os.path.abspath(path).replace("'", "''")
                cmd = f"Add-Type -AssemblyName System.Windows.Forms;[System.Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile('{abs_path}'))"
                creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
                subprocess.run(['powershell', '-command', cmd], creationflags=creationflags)
        elif sys.platform == 'darwin':
            abs_path = os.path.abspath(path)
            subprocess.run(['osascript', '-e', f'set the clipboard to (read (POSIX file "{abs_path}") as JPEG picture)'])
        else:
            subprocess.run(['xclip', '-selection', 'clipboard', '-t', 'image/png', '-i', path])
        ui.notify('Картинка скопирована в буфер обмена!', type='positive')
    except Exception as e:
        ui.notify(f'Ошибка копирования в буфер: {e}', type='negative')

# ==========================================
# 5. ВЕРСТКА И ИНТЕРФЕЙС NICEGUI
# ==========================================
@ui.page('/')
def index_page():
    cfg = load_config()

    def cancel_all_tasks():
        if state.is_processing:
            search_engine.cancel()         # Флаг для Поиска и Индексатора
            state.is_processing = False    # Флаг для Эстетики и NSFW
            state.add_log("🛑 Отправлен сигнал прерывания...")
            state.status_text = "Останавливаем процессы (завершение текущего батча)..."
            ui.notify('Останавливаем выполнение...', type='warning', position='top')

    ui.colors(primary='#2563eb', secondary='#10b981', accent='#f59e0b', dark='#1e1e2f')
    ui.query('body').classes('bg-[#121212] text-white overflow-hidden m-0 p-0')

    with ui.header().classes('bg-gray-900 border-b border-gray-800 flex justify-between items-center px-4 py-2 shrink-0'):
        ui.label('🤖 AI Media Organizer Pro').classes('text-xl font-bold tracking-wider text-blue-400')
        
    with ui.tabs().classes('w-full bg-gray-900 z-10 shrink-0').bind_value(state, 'current_tab') as tabs:
        tab_search = ui.tab('Search', label='Умный Поиск', icon='search')
        tab_aesthetic = ui.tab('Aesthetic', label='Оценка Эстетики', icon='star')
        tab_nsfw = ui.tab('NSFW', label='NSFW Детектор', icon='visibility_off')
        tab_cache = ui.tab('Cache', label='Индексатор', icon='storage')

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

    # --- ДИАЛОГ ДЛЯ ПОЛНОЭКРАННОГО ПЛЕЕРА ---
    with ui.dialog().on('value', lambda e: setattr(state, 'viewer_open', e.value)) as media_dialog:
        with ui.card().classes('w-[95vw] h-[95vh] bg-transparent shadow-none p-0 flex flex-col relative items-center justify-center'):
            ui.button(icon='close', on_click=media_dialog.close).classes('absolute top-2 right-2 z-50 bg-black/60 text-white').props('flat round dense')
            
            ui.button(icon='chevron_left', on_click=lambda: change_media(-1)).classes('absolute left-2 top-1/2 -translate-y-1/2 z-50 bg-black/60 text-white text-3xl').props('flat round').tooltip('Предыдущий (←)')
            ui.button(icon='chevron_right', on_click=lambda: change_media(1)).classes('absolute right-2 top-1/2 -translate-y-1/2 z-50 bg-black/60 text-white text-3xl').props('flat round').tooltip('Следующий (→)')
            
            media_container = ui.row().classes('w-full h-full flex items-center justify-center')
            
            with ui.row().classes('absolute bottom-4 left-1/2 -translate-x-1/2 bg-black/60 px-3 py-1 rounded-full text-white flex-nowrap items-center gap-2 z-50 shadow-lg'):
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
            
        btn_viewer_select._props['icon'] = 'check_box' if is_selected else 'check_box_outline_blank'
        btn_viewer_select._props['color'] = 'green' if is_selected else 'white'
        btn_viewer_select.update()

    def toggle_selection():
        if not state.viewer_items: return
        path = state.viewer_items[state.viewer_index]
        if state.current_tab == 'Search' and path in state.sel_search:
            state.sel_search[path] = not state.sel_search[path]
            search_gallery_ui.refresh()
        elif state.current_tab == 'Aesthetic' and path in state.sel_aes:
            state.sel_aes[path] = not state.sel_aes[path]
            aesthetic_gallery_ui.refresh()
        elif state.current_tab == 'NSFW' and path in state.sel_nsfw:
            state.sel_nsfw[path] = not state.sel_nsfw[path]
            nsfw_gallery_ui.refresh()
        update_viewer_selection_ui()

    def download_current_item():
        if not state.viewer_items: return
        path = state.viewer_items[state.viewer_index]
        tab = state.current_tab.lower()
        base_dir = state.search_base_dir if tab == 'search' else (state.aes_base_dir if tab == 'aesthetic' else state.nsfw_base_dir)
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
                ui.video(f'/media/{safe_path}').classes('max-w-full max-h-full object-contain outline-none').props('autoplay controls loop')
            elif ext in SUPPORTED_IMAGES:
                ui.image(f'/media/{safe_path}').classes('max-w-full max-h-full object-contain outline-none')
            else:
                ui.icon('article', size='15rem').classes('text-gray-500')

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

    ui.keyboard(on_key=handle_keyboard, ignore=['input', 'textarea', 'select'])

    # --- ЭКСПОРТ HTML ---
    async def export_html_action(tab='search'):
        folder = await run.io_bound(pick_folder_native)
        if not folder: return
        
        html_path = os.path.join(folder, f"gallery_{tab}.html")
        html_content =[
            "<html><body style='background-color:#1e1e1e; color:white; font-family:sans-serif;'>",
            f"<h2>Экспорт результатов</h2>",
            "<div style='display:flex; flex-wrap:wrap; gap:15px;'>"
        ]
        
        items = state.search_results if tab == 'search' else (state.aesthetic_results if tab == 'aes' else state.nsfw_results)
        for item in items:
            path = item[1]
            if tab == 'search': label_text = f"Score: {item[0]:.3f}"
            elif tab == 'aes': label_text = f"★ {item[0]:.2f} (Пик: {item[2]:.2f})"
            elif tab == 'nsfw': label_text = f"🚨 Danger: {item[0]*100:.1f}% | {item[2].upper()}"
                
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
            ui.notify(f"Галерея сохранена: {html_path}", type='positive')
        except Exception as e: ui.notify(f"Ошибка экспорта: {e}", type='negative')

    # --- ПАКЕТНЫЕ ДЕЙСТВИЯ ---
    async def execute_batch(action='copy', tab='search', prepend_score=False):
        sel_dict = state.sel_search if tab == 'search' else (state.sel_aes if tab == 'aes' else state.sel_nsfw)
        selected_paths =[p for p, checked in sel_dict.items() if checked]
        if not selected_paths:
            return ui.notify('Ничего не выбрано!', type='warning')
            
        folder = await run.io_bound(pick_folder_native)
        if not folder: return
        
        base_dir = state.search_base_dir if tab == 'search' else (state.aes_base_dir if tab == 'aes' else state.nsfw_base_dir)
        success = 0
        moved_paths = set()
        
        for path in selected_paths:
            try:
                rel_path = os.path.relpath(path, base_dir)
                if rel_path.startswith('..') or os.path.isabs(rel_path): rel_path = os.path.basename(path)
            except Exception: rel_path = os.path.basename(path)
                
            rel_dir, fname = os.path.split(rel_path)

            prefix = ""
            if prepend_score:
                if tab == 'search':
                    score = next((s for s, p in state.search_results if p == path), 0)
                    prefix = f"{score:.3f}_"
                elif tab == 'aes':
                    avg_s = next((a for a, p, m in state.aesthetic_results if p == path), 0)
                    prefix = f"{avg_s:05.2f}_"
                elif tab == 'nsfw':
                    danger_s = next((d for d, p, l, dt in state.nsfw_results if p == path), 0)
                    prefix = f"{danger_s*100:05.1f}_"
                    
            dest = os.path.join(folder, rel_dir, prefix + fname)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            
            try:
                if action == 'copy': shutil.copy2(path, dest)
                else: 
                    shutil.move(path, dest)
                    moved_paths.add(path)
                success += 1
            except Exception as e: state.add_log(f"Ошибка {path}: {e}")
                
        ui.notify(f'Успешно {action}: {success} файлов', type='positive')
        
        if action == 'move' and moved_paths:
            if tab == 'search':
                state.search_results =[i for i in state.search_results if i[1] not in moved_paths]
                search_gallery_ui.refresh()
            elif tab == 'aes':
                state.aesthetic_results =[i for i in state.aesthetic_results if i[1] not in moved_paths]
                aesthetic_gallery_ui.refresh()
            elif tab == 'nsfw':
                state.nsfw_results =[i for i in state.nsfw_results if i[1] not in moved_paths]
                nsfw_gallery_ui.refresh()

    async def handle_shift_click(e, idx, path, tab):
        # Проверяем, зажат ли Shift
        is_shift = isinstance(e.args, dict) and e.args.get('shiftKey', False)
        # Ждем 50мс, чтобы NiceGUI успел обновить значение текущего чекбокса в словаре
        await asyncio.sleep(0.05) 
        
        if tab == 'search':
            sel_dict = state.sel_search
            all_p =[p for s, p in state.search_results]
        elif tab == 'aes':
            sel_dict = state.sel_aes
            all_p =[p for a, p, m in state.aesthetic_results]
        elif tab == 'nsfw':
            sel_dict = state.sel_nsfw
            all_p = [p for d, p, l, dt in state.nsfw_results]

        # Получаем индекс прошлого клика (создастся автоматически, если его еще нет)
        last_idx = getattr(state, f'last_clicked_{tab}', None)

        if not is_shift:
            # Обычный клик — запоминаем индекс
            setattr(state, f'last_clicked_{tab}', idx)
        else:
            # Shift-клик — выделяем/снимаем выделение для диапазона
            if last_idx is not None:
                start = min(idx, last_idx)
                end = max(idx, last_idx)
                # Берем то значение, которое только что получил кликнутый чекбокс
                target_val = sel_dict.get(path, True)
                
                for i in range(start, end + 1):
                    sel_dict[all_p[i]] = target_val

    def set_all(tab, value):
        if tab == 'search':
            for p in state.sel_search: state.sel_search[p] = value
            search_gallery_ui.refresh()
        elif tab == 'aes':
            for p in state.sel_aes: state.sel_aes[p] = value
            aesthetic_gallery_ui.refresh()
        elif tab == 'nsfw':
            for p in state.sel_nsfw: state.sel_nsfw[p] = value
            nsfw_gallery_ui.refresh()

    # --- КОМПОНЕНТЫ ГАЛЕРЕИ ---
    @ui.refreshable
    def search_gallery_ui():
        if not state.search_results:
            return ui.label("Здесь появятся результаты...").classes("text-gray-400 m-4")

        total_pages = max(1, (len(state.search_results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        def change_page(d):
            state.search_page = max(1, min(total_pages, state.search_page + d))
            search_gallery_ui.refresh()

        def render_pagination():
            with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                ui.label(f'Страница {state.search_page} из {total_pages}').classes('text-gray-300 font-bold')
                ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')

        with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
            # Фиксированная верхняя панель (Панель управления + Пагинатор)
            with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('Выбрать всё', on_click=lambda: set_all('search', True)).props('outline color=white dense')
                        ui.button('Снять всё', on_click=lambda: set_all('search', False)).props('outline color=white dense')
                    with ui.row().classes('gap-2 items-center'):
                        ui.button('HTML Экспорт', icon='html', on_click=lambda: export_html_action('search')).props('color=purple dense outline')
                        ui.button('Копировать ✔', icon='content_copy', on_click=lambda: execute_batch('copy', 'search', chk_prefix_search.value)).props('color=blue dense')
                        ui.button('Переместить ✔', icon='drive_file_move', on_click=lambda: execute_batch('move', 'search', chk_prefix_search.value)).props('color=red dense')
                
                render_pagination()

            # Прокручиваемая область с результатами
            scroll_id = 'search_scroll_area'
            with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                start_idx = (state.search_page - 1) * ITEMS_PER_PAGE
                page_items = state.search_results[start_idx : start_idx + ITEMS_PER_PAGE]
                all_paths =[p for s, p in state.search_results]

                with ui.grid(columns=4).classes('w-full gap-6 pb-10'):
                    for score, path in page_items:
                        safe_path = urllib.parse.quote(path)
                        global_index = all_paths.index(path) # Вычисляем индекс ЗДЕСЬ
                        
                        with ui.card().classes('bg-gray-800 border border-gray-700 hover:border-blue-500 transition-colors p-0 overflow-hidden relative'):
                            with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                ui.checkbox().bind_value(state.sel_search, path).on('click', lambda e, i=global_index, p=path: handle_shift_click(e, i, p, 'search'), ['shiftKey'])
                            
                            with ui.context_menu():
                                ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                ui.menu_item('Копировать картинку', on_click=lambda p=path: copy_image_to_clipboard(p))
                                ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))

                            if os.path.splitext(path)[1].lower() in SUPPORTED_TEXTS:
                                ui.icon('article', size='4rem').classes('w-full h-48 flex items-center justify-center bg-gray-900 cursor-pointer text-gray-500').on('click', lambda p=path: open_file_native(p))
                            else:
                                ui.image(f"/thumb/{safe_path}").classes('w-full h-48 object-contain cursor-pointer bg-black').props('fit=contain').on('click', lambda e, idx=global_index: open_media(idx, all_paths))
                            
                            with ui.row().classes('w-full justify-between items-center p-2 bg-gray-800'):
                                ui.label(f"Score: {score:.3f}").classes('text-green-400 font-bold text-sm')
                                ui.button(icon='folder', on_click=lambda p=path: reveal_file_native(p)).props('flat round dense color=white')
                            ui.label(os.path.basename(path)).classes('text-xs text-gray-400 px-2 pb-2 truncate w-full').tooltip(path)

            # Плавающая кнопка "Наверх"
            ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=blue').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')

    @ui.refreshable
    def aesthetic_gallery_ui():
        if not state.aesthetic_results:
            return ui.label("Здесь появятся топовые фото/видео...").classes("text-gray-400 m-4")

        total_pages = max(1, (len(state.aesthetic_results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        def change_page(d):
            state.aes_page = max(1, min(total_pages, state.aes_page + d))
            aesthetic_gallery_ui.refresh()

        def render_pagination():
            with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                ui.label(f'Страница {state.aes_page} из {total_pages}').classes('text-gray-300 font-bold')
                ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')

        with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
            # Фиксированная верхняя панель (Панель управления + Пагинатор)
            with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                    with ui.row().classes('gap-2'):
                        ui.button('Выбрать всё', on_click=lambda: set_all('aes', True)).props('outline color=white dense')
                        ui.button('Снять всё', on_click=lambda: set_all('aes', False)).props('outline color=white dense')
                    with ui.row().classes('gap-2'):
                        ui.button('HTML Экспорт', icon='html', on_click=lambda: export_html_action('aes')).props('color=purple dense outline')
                        ui.button('Копировать ✔', icon='content_copy', on_click=lambda: execute_batch('copy', 'aes', chk_prefix_aes.value)).props('color=yellow-800 dense')
                        ui.button('Переместить ✔', icon='drive_file_move', on_click=lambda: execute_batch('move', 'aes', chk_prefix_aes.value)).props('color=red dense')

                render_pagination()

            # Прокручиваемая область с результатами
            scroll_id = 'aes_scroll_area'
            with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                start_idx = (state.aes_page - 1) * ITEMS_PER_PAGE
                page_items = state.aesthetic_results[start_idx : start_idx + ITEMS_PER_PAGE]
                all_paths =[p for a, p, m in state.aesthetic_results]

                with ui.grid(columns=4).classes('w-full gap-6 pb-10'):
                    for avg_score, path, max_score in page_items:
                        safe_path = urllib.parse.quote(path)
                        global_index = all_paths.index(path) # Вычисляем индекс ЗДЕСЬ
                        
                        with ui.card().classes('bg-gray-800 border border-gray-700 hover:border-yellow-500 transition-colors p-0 overflow-hidden relative'):
                            with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                ui.checkbox().bind_value(state.sel_aes, path).on('click', lambda e, i=global_index, p=path: handle_shift_click(e, i, p, 'aes'),['shiftKey'])

                            with ui.context_menu():
                                ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                ui.menu_item('Копировать картинку', on_click=lambda p=path: copy_image_to_clipboard(p))
                                ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))

                            ui.image(f"/thumb/{safe_path}").classes('w-full h-48 object-contain cursor-pointer bg-black').props('fit=contain').on('click', lambda e, idx=global_index: open_media(idx, all_paths))
                            
                            with ui.row().classes('w-full justify-between items-center p-2'):
                                ui.label(f"★ {avg_score:.2f}").classes('text-yellow-400 font-bold text-lg')
                                if avg_score != max_score: ui.label(f"Пик: {max_score:.2f}").classes('text-xs text-gray-500')
                            ui.label(os.path.basename(path)).classes('text-xs text-gray-400 px-2 pb-2 truncate w-full').tooltip(path)

            # Плавающая кнопка "Наверх"
            ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=yellow-800').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')

    @ui.refreshable
    def nsfw_gallery_ui():
        if not state.nsfw_results:
            return ui.label("Здесь появятся результаты NSFW сканирования...").classes("text-gray-400 m-4")

        total_pages = max(1, (len(state.nsfw_results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
        def change_page(d):
            state.nsfw_page = max(1, min(total_pages, state.nsfw_page + d))
            nsfw_gallery_ui.refresh()

        def render_pagination():
            with ui.row().classes('w-full justify-center my-0 items-center gap-4'):
                ui.button(icon='chevron_left', on_click=lambda: change_page(-1)).props('flat outline color=white')
                ui.label(f'Страница {state.nsfw_page} из {total_pages}').classes('text-gray-300 font-bold')
                ui.button(icon='chevron_right', on_click=lambda: change_page(1)).props('flat outline color=white')

        with ui.column().classes('w-full h-full flex flex-col p-0 m-0 gap-0 relative'):
            # Фиксированная верхняя панель (Панель управления + Пагинатор)
            with ui.column().classes('w-full shrink-0 bg-gray-900 p-4 pb-2 border-b border-gray-800 z-20 gap-0 shadow-md'):
                with ui.row().classes('w-full flex justify-between items-center p-2 bg-gray-800 rounded-lg mb-2'):
                    with ui.row().classes('gap-2'):
                        ui.button('Выбрать всё', on_click=lambda: set_all('nsfw', True)).props('outline color=white dense')
                        ui.button('Снять всё', on_click=lambda: set_all('nsfw', False)).props('outline color=white dense')
                    with ui.row().classes('gap-2'):
                        ui.button('HTML Экспорт', icon='html', on_click=lambda: export_html_action('nsfw')).props('color=purple dense outline')
                        ui.button('Копировать ✔', icon='content_copy', on_click=lambda: execute_batch('copy', 'nsfw', chk_prefix_nsfw.value)).props('color=red-800 dense')
                        ui.button('Переместить ✔', icon='drive_file_move', on_click=lambda: execute_batch('move', 'nsfw', chk_prefix_nsfw.value)).props('color=red dense')

                render_pagination()

            # Прокручиваемая область с результатами
            scroll_id = 'nsfw_scroll_area'
            with ui.column().classes('w-full flex-1 overflow-y-auto p-4 relative').props(f'id="{scroll_id}"'):
                start_idx = (state.nsfw_page - 1) * ITEMS_PER_PAGE
                page_items = state.nsfw_results[start_idx : start_idx + ITEMS_PER_PAGE]
                all_paths =[p for d, p, l, dt in state.nsfw_results]

                with ui.grid(columns=4).classes('w-full gap-6 pb-10'):
                    for danger_score, path, top_label, details in page_items:
                        safe_path = urllib.parse.quote(path)
                        global_index = all_paths.index(path) # Вычисляем индекс ЗДЕСЬ
                        
                        with ui.card().classes('bg-gray-800 border border-gray-700 hover:border-red-500 transition-colors p-0 overflow-hidden relative'):
                            with ui.row().classes('absolute top-2 left-2 bg-black/60 rounded px-1 z-10'):
                                ui.checkbox().bind_value(state.sel_nsfw, path).on('click', lambda e, i=global_index, p=path: handle_shift_click(e, i, p, 'nsfw'), ['shiftKey'])

                            with ui.context_menu():
                                ui.menu_item('Скопировать путь', on_click=lambda p=path: ui.clipboard.write(p))
                                ui.menu_item('Копировать картинку', on_click=lambda p=path: copy_image_to_clipboard(p))
                                ui.menu_item('Открыть папку', on_click=lambda p=path: reveal_file_native(p))

                            ui.image(f"/thumb/{safe_path}").classes('w-full h-48 object-contain cursor-pointer bg-black').props('fit=contain').on('click', lambda e, idx=global_index: open_media(idx, all_paths))
                            
                            with ui.row().classes('w-full justify-between items-center p-2'):
                                ui.label(f"🚨 {danger_score*100:.1f}%").classes('text-red-500 font-bold text-lg')
                                ui.button(icon='troubleshoot', on_click=lambda p=path, d=details: show_nsfw_debug(p, d)).props('flat round dense color=white').tooltip('Детальный разбор категорий')
                                
                            ui.label(top_label.upper()).classes('text-xs text-gray-400 font-bold px-2')
                            ui.label(os.path.basename(path)).classes('text-xs text-gray-400 px-2 pb-2 truncate w-full').tooltip(path)

            # Плавающая кнопка "Наверх"
            ui.button(icon='keyboard_arrow_up', on_click=lambda: ui.run_javascript(f'document.getElementById("{scroll_id}").scrollTo({{top: 0, behavior: "smooth"}})')).props('round color=red-800').classes('absolute bottom-6 right-6 z-50 shadow-lg').tooltip('Наверх')


    # --- ОСНОВНАЯ РАБОЧАЯ ОБЛАСТЬ ---
    with ui.tab_panels(tabs).bind_value(state, 'current_tab').classes('w-full bg-[#121212] p-0'):
        
        # ВКЛАДКА 1: ПОИСК
        with ui.tab_panel(tab_search).classes('w-full h-[calc(100vh-180px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('Параметры поиска').classes('text-lg font-bold')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto p-4 gap-2 min-h-0'):
                    with ui.row().classes('w-full items-center gap-1 flex-nowrap'):
                        inp_dir = ui.input('Папка', value=cfg.get('inp_dir', '')).classes('flex-grow')
                        ui.button(icon='folder', on_click=lambda: select_folder(inp_dir)).props('flat round dense')
                        ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache(inp_dir.value)).props('flat round dense text-color=red').tooltip('Очистить индекс файлов папки')
                    
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
                            quant_mode = ui.select(['None', '8-bit', '4-bit'], value=cfg.get('quant_mode', 'None'), label='Квант').classes('w-[45%]')
                            min_score = ui.number('Мин Score', value=cfg.get('min_score', 0.25), format='%.2f', step=0.05).classes('w-[45%]')
                        
                        # --- ДОБАВЛЕННЫЙ БЛОК NSFW-ФИЛЬТРА ДЛЯ ПОИСКА ---
                        with ui.column().classes('w-full gap-0 px-2 pb-2 pt-2 border-t border-gray-700'):
                            search_nsfw_filter = ui.toggle(['Все', 'Только SFW', 'Только NSFW'], value=cfg.get('search_nsfw_filter', 'Все')).classes('w-full text-xs mb-1')
                            # Показываем галочку только если выбран фильтр (не "Все")
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
                        'quant_mode': quant_mode.value, 'min_score': min_score.value,
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
                            search_engine.quant_mode = quant_mode.value
                            
                            state.search_base_dir = inp_dir.value
                            q_emb, q_rank = search_engine.prepare_query(inp_query.value)
                            cands = search_engine.phase1_recall(inp_dir.value, inp_query.value, q_emb, int(top_k.value), emb_model.value, int(batch_size.value), tuple(exts))
                            
                            if use_reranker.value: 
                                cands = search_engine.phase2_rerank(inp_query.value, q_rank, cands, float(min_score.value), rerank_model.value)

                            # --- ЛОГИКА ФИЛЬТРАЦИИ NSFW ---
                            if search_nsfw_filter.value != 'Все':
                                filtered_cands =[]
                                for score, path in cands:
                                    danger = search_engine.db_cache.get_max_danger_score(path)
                                    
                                    if danger == -1.0: # Файла нет в базе
                                        if search_strict_nsfw.value: continue # Строгий режим: отсекаем
                                        else:
                                            # Мягкий режим: неизвестное считаем за SFW (а в NSFW-поиск не пускаем)
                                            if search_nsfw_filter.value == 'Только NSFW': continue 
                                    else:
                                        is_nsfw = danger >= 0.45 # Порог опасности (можно поменять, 0.45 = 45%)
                                        if search_nsfw_filter.value == 'Только SFW' and is_nsfw: continue
                                        if search_nsfw_filter.value == 'Только NSFW' and not is_nsfw: continue
                                        
                                    filtered_cands.append((score, path))
                                cands = filtered_cands
                            # -----------------------------
                                
                            state.search_results = cands
                            state.sel_search = {path: False for _, path in cands}
                            state.add_log("✅ Поиск успешно завершен!")
                        except Exception as e: state.add_log(f"❌ Ошибка поиска: {e}")
                        finally:
                            state.status_text = "Готово!"
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(task)
                    search_gallery_ui.refresh()
                    btn_search.enable()

                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10'):
                    btn_search = ui.button('🚀 Искать', on_click=run_search_action).classes('w-full bg-blue-600 hover:bg-blue-500 font-bold')

            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                search_gallery_ui()

        # ВКЛАДКА 2: ЭСТЕТИКА
        with ui.tab_panel(tab_aesthetic).classes('w-full h-[calc(100vh-180px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('Оценка Эстетики').classes('text-lg font-bold')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto p-4 gap-2 min-h-0'):
                    with ui.row().classes('w-full items-center gap-1 flex-nowrap'):
                        rate_dir = ui.input('Папка', value=cfg.get('rate_dir', '')).classes('flex-grow')
                        ui.button(icon='folder', on_click=lambda: select_folder(rate_dir)).props('flat round dense')
                        ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache(rate_dir.value)).props('flat round dense text-color=red').tooltip('Очистить индекс файлов папки')
                        
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
                            
                        # --- ДОБАВЛЕННЫЙ БЛОК NSFW-ФИЛЬТРА ДЛЯ ЭСТЕТИКИ ---
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
                        'chk_prefix_aes': chk_prefix_aes.value,
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
                    
                    exts =[]
                    if chk_img_aes.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid_aes.value: exts.extend(SUPPORTED_VIDEOS)

                    def bg_task():
                        try:
                            state.add_log(f"Запуск оценки эстетики для папки: '{rate_dir.value}'")
                            aesthetic_engine.batch_size = int(aes_batch_size.value)
                            aesthetic_engine.video_frames = int(aes_video_frames.value)
                            aesthetic_engine.max_dim = int(aes_max_dim.value)
                            
                            state.aes_base_dir = rate_dir.value
                            n = int(top_n_rate.value)
                            
                            res = aesthetic_engine.evaluate_media(rate_dir.value, tuple(exts))

                            # --- ЛОГИКА ФИЛЬТРАЦИИ NSFW ---
                            if aes_nsfw_filter.value != 'Все':
                                filtered_res = []
                                for item in res:
                                    path = item[1] # В эстетике структура: (avg_score, path, max_score)
                                    danger = aesthetic_engine.db_cache.get_max_danger_score(path)
                                    
                                    if danger == -1.0: # Нет в базе
                                        if aes_strict_nsfw.value: continue
                                        else:
                                            if aes_nsfw_filter.value == 'Только NSFW': continue
                                    else:
                                        is_nsfw = danger >= 0.45
                                        if aes_nsfw_filter.value == 'Только SFW' and is_nsfw: continue
                                        if aes_nsfw_filter.value == 'Только NSFW' and not is_nsfw: continue
                                        
                                    filtered_res.append(item)
                                res = filtered_res
                            # -----------------------------
                                
                            state.aesthetic_results = res[:n]
                            state.sel_aes = {path: False for _, path, _ in state.aesthetic_results}
                            state.add_log("✅ Оценка эстетики завершена!")
                        except Exception as e: state.add_log(f"❌ Ошибка: {e}")
                        finally:
                            state.status_text = "Готово!"
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(bg_task)
                    aesthetic_gallery_ui.refresh()
                    btn_rate.enable()
                    
                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10'):
                    btn_rate = ui.button('✨ Оценить', on_click=run_aesthetic_action).classes('w-full bg-yellow-600 hover:bg-yellow-500 font-bold text-lg')

            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                aesthetic_gallery_ui()

        # ВКЛАДКА 3: NSFW
        with ui.tab_panel(tab_nsfw).classes('w-full h-[calc(100vh-180px)] p-4 flex flex-row flex-nowrap items-stretch gap-4'):
            with ui.column().classes('w-[350px] shrink-0 bg-gray-900 rounded-xl border border-gray-800 shadow-lg flex flex-col overflow-hidden p-0 gap-0'):
                with ui.row().classes('w-full p-4 pb-2 shrink-0 border-b border-gray-800 bg-gray-900 z-10'):
                    ui.label('NSFW Детектор').classes('text-lg font-bold')
                
                with ui.column().classes('w-full flex-1 overflow-y-auto p-4 gap-2 min-h-0'):
                    with ui.row().classes('w-full items-center gap-1 flex-nowrap'):
                        nsfw_dir = ui.input('Папка', value=cfg.get('nsfw_dir', '')).classes('flex-grow')
                        ui.button(icon='folder', on_click=lambda: select_folder(nsfw_dir)).props('flat round dense')
                        ui.button(icon='delete_sweep', on_click=lambda: clear_folder_cache(nsfw_dir.value)).props('flat round dense text-color=red').tooltip('Очистить индекс файлов папки')
                        
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
                
                async def run_nsfw_action():
                    save_config({
                        'nsfw_dir': nsfw_dir.value, 'chk_img_nsfw': chk_img_nsfw.value, 'chk_vid_nsfw': chk_vid_nsfw.value,
                        'nsfw_model': nsfw_model_sel.value, 'top_n_nsfw': top_n_nsfw.value, 
                        'nsfw_batch_size': nsfw_batch_size.value, 'nsfw_video_frames': nsfw_video_frames.value, 
                        'nsfw_max_dim': nsfw_max_dim.value, 'chk_prefix_nsfw': chk_prefix_nsfw.value
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
                    
                    exts =[]
                    if chk_img_nsfw.value: exts.extend(SUPPORTED_IMAGES)
                    if chk_vid_nsfw.value: exts.extend(SUPPORTED_VIDEOS)

                    def bg_task():
                        try:
                            state.add_log(f"Запуск NSFW сканирования для папки: '{nsfw_dir.value}'")
                            nsfw_engine.batch_size = int(nsfw_batch_size.value)
                            nsfw_engine.video_frames = int(nsfw_video_frames.value)
                            nsfw_engine.max_dim = int(nsfw_max_dim.value)
                            
                            state.nsfw_base_dir = nsfw_dir.value
                            n = int(top_n_nsfw.value)
                            
                            res = nsfw_engine.evaluate_media(nsfw_dir.value, nsfw_model_sel.value, tuple(exts))
                                
                            state.nsfw_results = res[:n]
                            state.sel_nsfw = {path: False for _, path, _, _ in state.nsfw_results}
                            state.add_log("✅ NSFW сканирование завершено!")
                        except Exception as e: state.add_log(f"❌ Ошибка: {e}")
                        finally:
                            state.status_text = "Готово!"
                            state.progress = 1.0
                            state.is_processing = False

                    await run.io_bound(bg_task)
                    nsfw_gallery_ui.refresh()
                    btn_nsfw.enable()
                    
                with ui.row().classes('w-full p-4 pt-2 shrink-0 border-t border-gray-800 bg-gray-900 z-10'):
                    btn_nsfw = ui.button('🚨 Анализ', on_click=run_nsfw_action).classes('w-full bg-red-800 hover:bg-red-700 font-bold text-lg')

            with ui.column().classes('flex-1 w-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden h-full relative p-0'):
                nsfw_gallery_ui()

        # ВКЛАДКА 4: ИНДЕКСАТОР (Кэш)
        with ui.tab_panel(tab_cache).classes('w-full h-[calc(100vh-180px)] p-8 flex flex-col items-center overflow-y-auto pb-24'):
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
                emb_model_cache = ui.select(['Qwen/Qwen3-VL-Embedding-2B', 'Qwen/Qwen3-VL-Embedding-8B'], value=cfg.get('emb_model', 'Qwen/Qwen3-VL-Embedding-2B')).classes('w-full pl-6').bind_visibility_from(chk_cache_search, 'value')
                
                chk_cache_aes = ui.checkbox('Оценка Эстетики', value=cfg.get('chk_cache_aes', True)).classes('text-md font-bold text-yellow-300')
                
                chk_cache_nsfw = ui.checkbox('NSFW Детектор', value=cfg.get('chk_cache_nsfw', True)).classes('text-md font-bold text-red-300')
                nsfw_model_cache = ui.select(['prithivMLmods/siglip2-x256-explicit-content', 'strangerguardhf/nsfw-image-detection'], value=cfg.get('nsfw_model', 'prithivMLmods/siglip2-x256-explicit-content')).classes('w-full pl-6').bind_visibility_from(chk_cache_nsfw, 'value')

                with ui.expansion('Единые тонкие настройки', icon='tune').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-4'):
                    with ui.row().classes('w-full gap-4 px-4 pt-4'):
                        cache_batch_size = ui.number('Размер Батча', value=cfg.get('batch_size', 16), format='%.0f').classes('flex-1')
                        cache_video_frames = ui.number('Кадров видео', value=cfg.get('video_frames', 4), format='%.0f').classes('flex-1')
                    with ui.row().classes('w-full gap-4 px-4 pb-4'):
                        cache_max_dim = ui.number('Лимит разрешения (размер)', value=cfg.get('emb_size', 512), format='%.0f').classes('flex-1')
                
                # --- НОВЫЙ БЛОК НАСТРОЕК ОПТИМИЗАЦИИ ПАМЯТИ ---
                with ui.expansion('Оптимизация памяти (ОЗУ)', icon='memory').classes('w-full bg-gray-800/50 rounded-lg border border-gray-700 mt-2'):
                    with ui.column().classes('w-full gap-2 px-4 py-4'):
                        ui.label('Эти настройки предотвращают вылет программы (OOM) при больших папках.').classes('text-gray-400 text-xs')
                        
                        use_ram_compression = ui.checkbox('Сжатие кэша в ОЗУ (Экономит до 90% памяти ценой 5-10% нагрузки на CPU)', value=cfg.get('use_ram_compression', False)).classes('text-sm text-green-400 font-bold')
                        
                        ui.label('Блочная архитектура (Радикально спасает ОЗУ)').classes('font-bold text-sm mt-2')
                        cache_chunk_size = ui.number('Размер блока файлов (0 = выключено, рекомендуемое = 2000)', value=cfg.get('cache_chunk_size', 2000), format='%.0f').classes('w-full')
                # --------------------------------------------------

                async def run_cache_action():
                    save_config({
                        'cache_dir': cache_dir.value, 'chk_cache_img': chk_cache_img.value, 
                        'chk_cache_vid': chk_cache_vid.value, 'chk_cache_txt': chk_cache_txt.value,
                        'chk_cache_search': chk_cache_search.value, 'chk_cache_aes': chk_cache_aes.value, 
                        'chk_cache_nsfw': chk_cache_nsfw.value,
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
                            
                            aesthetic_engine.batch_size = int(cache_batch_size.value)
                            aesthetic_engine.max_dim = int(cache_max_dim.value)
                            aesthetic_engine.video_frames = int(cache_video_frames.value)
                            
                            nsfw_engine.batch_size = int(cache_batch_size.value)
                            nsfw_engine.max_dim = int(cache_max_dim.value)
                            nsfw_engine.video_frames = int(cache_video_frames.value)

                            # --- НОВАЯ ЛОГИКА БЛОЧНОЙ ОБРАБОТКИ (Вариант 3) ---
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
                                    search_engine.build_cache(cache_dir.value, emb_model_cache.value, int(cache_batch_size.value), tuple(exts_search), override_files=chunk)
                                    
                                if chk_cache_aes.value and not search_engine.cancel_flag:
                                    if len(chunks) > 1: state.add_log(f"-> Блок {idx+1}: Оценка Эстетики")
                                    else: state.add_log(f"-> Этап 2: Оценка Эстетики")
                                    search_engine._unload_embedding_model()
                                    nsfw_engine.unload()
                                    aesthetic_engine.evaluate_media(cache_dir.value, tuple(exts_media), override_files=chunk)
                                    
                                if chk_cache_nsfw.value and not search_engine.cancel_flag:
                                    if len(chunks) > 1: state.add_log(f"-> Блок {idx+1}: NSFW Детектор")
                                    else: state.add_log(f"-> Этап 3: NSFW Детектор")
                                    search_engine._unload_embedding_model()
                                    aesthetic_engine.unload()
                                    nsfw_engine.evaluate_media(cache_dir.value, nsfw_model_cache.value, tuple(exts_media), override_files=chunk)
                                    
                            state.add_log("🎉 Полная индексация успешно завершена!")
                        except Exception as e: state.add_log(f"❌ Ошибка индексации: {e}")
                        finally:
                            # Полная очистка по завершению
                            media_cache.enabled = False
                            media_cache.compress = False
                            media_cache.clear()
                            
                            state.is_processing = False
                            state.progress = 1.0
                            state.status_text = "Готово!"

                    await run.io_bound(bg_task)
                    btn_cache.enable()

                btn_cache = ui.button('🚀 Запустить полное кэширование', on_click=run_cache_action).classes('w-full bg-blue-600 hover:bg-blue-500 font-bold text-lg mt-4')


    # Компактный подвал (Status bar) - строго в одну линию
    with ui.footer().classes('bg-gray-900 border-t border-gray-800 px-4 py-0 flex flex-row flex-nowrap items-center justify-between z-40 h-8 shadow-lg'):
        ui.label().bind_text_from(state, 'status_text').classes('text-blue-400 font-mono text-xs truncate max-w-[30%] shrink-0')
        ui.linear_progress(value=0, show_value=False).bind_value_from(state, 'progress').classes('flex-grow mx-4 h-1.5 rounded text-blue-600')
        
        # --- КНОПКА ОТМЕНЫ (показывается только когда что-то запущено) ---
        ui.button('ПРЕРВАТЬ', icon='cancel', on_click=cancel_all_tasks) \
            .props('color=red size=sm dense outline') \
            .classes('shrink-0 py-0 min-h-0 text-xs font-bold mr-2 bg-red-900/20') \
            .bind_visibility_from(state, 'is_processing')
        # ----------------------------------------------------------------
            
        ui.button('ЛОГИ', icon='terminal', on_click=log_drawer.toggle).props('flat text-color=white size=sm dense').classes('shrink-0 py-0 min-h-0 text-xs')

    ui.timer(0.5, update_ui_logs)

if __name__ in {"__main__", "__mp_main__"}:
    parser = argparse.ArgumentParser(description="AI Media Organizer Pro")
    parser.add_argument('--server-only', action='store_true', help='Запустить в режиме сервера (без локального окна)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='IP адрес для сервера')
    parser.add_argument('--port', type=int, default=8080, help='Порт сервера')
    
    # Считываем аргументы
    args, unknown = parser.parse_known_args()

    if args.server_only:
        print(f"🌐 Режим сервера активирован. Откройте в браузере: http://{args.host}:{args.port}")
        # native=False отключает десктопное окно, show=False предотвращает автоматическое открытие вкладки
        ui.run(title="AI Media Organizer Pro", host=args.host, port=args.port, native=False, show=False, dark=True, reload=False)
    else:
        # Стандартный оконный (Native) режим
        ui.run(title="AI Media Organizer Pro", port=args.port, native=True, dark=True, window_size=(1400, 900), reload=False)
