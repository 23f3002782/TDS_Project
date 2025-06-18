import os
import json
import numpy as np
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

CHUNK_SIZE = 800
OVERLAP = 0.25
MIN_CHUNK_SIZE = 50
EMBEDDINGS_DIR = "embeddings"
PROCESSED_DIR = "processed_documents"
MODEL = "nomic-embed-text:latest"
OLLAMA_URL = "http://localhost:11434"

# Simple chunker with overlap

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: float = OVERLAP) -> List[str]:
    step = int(chunk_size * (1 - overlap))
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) >= MIN_CHUNK_SIZE:
            chunks.append(chunk)
    return chunks

def extract_yaml_frontmatter(text: str) -> str:
    """Extract YAML frontmatter if present, else return empty string."""
    match = re.match(r'^(---\n.*?\n---\n)', text, flags=re.DOTALL)
    return match.group(1) if match else ''

def chunk_text_with_frontmatter(text: str, chunk_size: int = CHUNK_SIZE, overlap: float = OVERLAP) -> List[str]:
    # Extract YAML frontmatter if present
    frontmatter = extract_yaml_frontmatter(text)
    if frontmatter:
        # Remove frontmatter from text for normal chunking
        text_wo_frontmatter = text[len(frontmatter):]
        # First chunk: frontmatter + first chunk of content
        step = int(chunk_size * (1 - overlap))
        content_chunks = []
        for i in range(0, len(text_wo_frontmatter), step):
            chunk = text_wo_frontmatter[i:i+chunk_size]
            if len(chunk.strip()) >= MIN_CHUNK_SIZE:
                content_chunks.append(chunk)
        if content_chunks:
            # Prepend frontmatter to the first chunk
            content_chunks[0] = frontmatter + content_chunks[0]
        else:
            content_chunks = [frontmatter]
        return content_chunks
    else:
        # No frontmatter, chunk as usual
        return chunk_text(text, chunk_size, overlap)

def preprocess_text(text: str) -> str:
    # Remove YAML frontmatter
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL | re.MULTILINE)
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def generate_embedding(text: str) -> Optional[List[float]]:
    payload = json.dumps({"model": MODEL, "prompt": text})
    try:
        result = subprocess.run([
            'curl', '-s', f'{OLLAMA_URL}/api/embeddings', '-d', payload
        ], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Ollama error: {result.stderr}")
            return None
        response = json.loads(result.stdout)
        return response.get('embedding', None)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def chunk_text_semantic(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Chunk text by paragraphs, grouping them until chunk_size is reached.
    Always includes YAML frontmatter (if present) in the first chunk.
    """
    # Extract YAML frontmatter if present
    frontmatter = extract_yaml_frontmatter(text)
    if frontmatter:
        text_wo_frontmatter = text[len(frontmatter):]
    else:
        text_wo_frontmatter = text

    # Split into paragraphs (double newline or single newline between non-empty lines)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text_wo_frontmatter) if p.strip()]
    chunks = []
    current_chunk = frontmatter if frontmatter else ""
    current_len = len(current_chunk)

    for para in paragraphs:
        if current_len + len(para) + 2 <= chunk_size or not current_chunk:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
            current_len = len(current_chunk)
        else:
            # Save current chunk and start new one
            if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(current_chunk)
            current_chunk = para
            current_len = len(current_chunk)
    # Add last chunk
    if current_chunk and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk)
    return chunks

def extract_source_url(text: str) -> str:
    """Extract source_url from YAML frontmatter if present."""
    match = re.search(r'source_url:\s*["\']([^"\']+)["\']', text)
    return match.group(1) if match else None

# Replace all calls to chunk_text_with_frontmatter with chunk_text_semantic
def process_file(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    source_url = extract_source_url(text)
    chunks = chunk_text_semantic(text)
    embeddings = []
    chunk_texts = []
    source_urls = []
    for chunk in chunks:
        emb = generate_embedding(chunk)
        if emb:
            embeddings.append(emb)
            chunk_texts.append(chunk)
            source_urls.append(source_url)
    return {
        'file_path': file_path,
        'embeddings': embeddings,
        'texts': chunk_texts,
        'source_urls': source_urls
    }

def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    all_embeddings = []
    all_texts = []
    file_indices = []
    chunk_indices = []
    file_paths = []
    all_source_urls = []
    idx = 0
    # Process all .md files in both subfolders
    for subdir in ['discourse', 'tds_course']:
        folder = os.path.join(PROCESSED_DIR, subdir)
        if not os.path.exists(folder):
            continue
        md_files = list(Path(folder).glob('*.md'))
        for mdfile in tqdm(md_files, desc=f"Processing {subdir}"):
            result = process_file(str(mdfile))
            for i, emb in enumerate(result['embeddings']):
                all_embeddings.append(emb)
                all_texts.append(result['texts'][i])
                file_indices.append(idx)
                chunk_indices.append(i)
                file_paths.append(str(mdfile))
                all_source_urls.append(result['source_urls'][i])
            idx += 1
    # Save as npz
    np.savez_compressed(
        os.path.join(EMBEDDINGS_DIR, 'all_embeddings'),
        embeddings=np.array(all_embeddings, dtype=np.float32),
        texts=np.array(all_texts, dtype=object),
        file_indices=np.array(file_indices),
        chunk_indices=np.array(chunk_indices),
        file_paths=np.array(file_paths, dtype=object),
        source_urls=np.array(all_source_urls, dtype=object)
    )
    print(f"Saved {len(all_embeddings)} embeddings to {EMBEDDINGS_DIR}/all_embeddings.npz")

if __name__ == "__main__":
    main()
