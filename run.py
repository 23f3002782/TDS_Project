import os
import json
import base64
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import io
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

EMBEDDINGS_FILE = "embeddings/all_embeddings.npz"
OLLAMA_URL = "http://localhost:11434"
MODEL = "nomic-embed-text:latest"
GEMINI_API_KEY = "AIzaSyA7nj6so88kbL-i4ztf7TjGcnHb_54xxck"

# Configure the Gemini client
client = genai.Client(api_key="AIzaSyA7nj6so88kbL-i4ztf7TjGcnHb_54xxck")

async def lifespan(app: FastAPI):
    app.state.index = EmbeddingIndex(EMBEDDINGS_FILE)
    yield

app = FastAPI(lifespan=lifespan)

# Enable CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
    question: str
    image: Optional[str] = None

# Load embeddings at startup
class EmbeddingIndex:
    def __init__(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.texts = data['texts']
        self.file_paths = data['file_paths']
        self.chunk_indices = data['chunk_indices']
        # Load optional metadata arrays if present
        self.source_urls = data['source_urls'] if 'source_urls' in data else None
        self.section_titles = data['section_titles'] if 'section_titles' in data else None
    def search(self, query, top_k=15):
        query_emb = generate_embedding(query)
        if query_emb is None:
            return []
        sims = cosine_similarity(np.array(query_emb).reshape(1, -1), self.embeddings)[0]
        top_idx = np.argsort(sims)[-top_k:][::-1]
        results = []
        for i in top_idx:
            if sims[i] > 0.1:
                result = {
                    'text': str(self.texts[i]),
                    'file_path': str(self.file_paths[i]),
                    'chunk_index': int(self.chunk_indices[i]),
                    'score': float(sims[i])
                }
                if self.source_urls is not None:
                    result['source_url'] = str(self.source_urls[i]) if self.source_urls[i] else None
                if self.section_titles is not None:
                    result['section_title'] = str(self.section_titles[i]) if self.section_titles[i] else None
                results.append(result)
        return results

def generate_embedding(text: str):
    payload = json.dumps({"model": MODEL, "prompt": text})
    try:
        result = subprocess.run([
            'curl', '-s', f'{OLLAMA_URL}/api/embeddings', '-d', payload
        ], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        response = json.loads(result.stdout)
        return response.get('embedding', None)
    except Exception:
        return None

def extract_links(search_results: List[dict]) -> List[dict]:
    """
    Extract links from search results using source_url if present (from course or discourse files).
    Fallback to discourse extraction logic if source_url is missing.
    """
    links = []
    for result in search_results:
        file_path = result.get('file_path', '')
        text_content = result.get('text', '')
        # Try to get source_url from embedding metadata (if present)
        source_url = result.get('source_url')
        # Try to get section title if present (for link text)
        section_title = result.get('section_title')
        filename = Path(file_path).name
        # If source_url is present, use it directly
        if source_url:
            # Use section title, or fallback to filename
            link_text = section_title or filename.replace('.md', '').replace('-', ' ').title()
            links.append({'url': source_url, 'text': link_text})
            continue
        # Fallback: discourse logic (legacy)
        filename_lower = filename.lower()
        is_discourse_file = ('tds-kb-' in filename_lower and filename_lower.endswith('.md')) or 'discourse' in file_path.lower()
        if is_discourse_file:
            discourse_url = None
            title = None
            yaml_url_match = re.search(r'discourse_url:\s*["\']([^"\']+)["\']', text_content)
            yaml_title_match = re.search(r'title:\s*["\']([^"\']+)["\']', text_content)
            if yaml_url_match:
                discourse_url = yaml_url_match.group(1)
            if yaml_title_match:
                title = yaml_title_match.group(1)
            if not discourse_url:
                topic_match = re.search(r'tds-kb-(\d+)\.md', filename_lower)
                if topic_match:
                    topic_id = topic_match.group(1)
                    discourse_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}"
            if not discourse_url:
                url_pattern = r'https://discourse\\.onlinedegree\\.iitm\\.ac\\.in/t/[^\\s\\)\"\']+?(?=[\\s\\)\"\']\\B|\\Z)'
                urls_in_content = re.findall(url_pattern, text_content)
                if urls_in_content:
                    discourse_url = urls_in_content[0]
            if discourse_url:
                link_text = title or filename.replace('.md', '').replace('-', ' ').title()
                links.append({'url': discourse_url, 'text': link_text})
    # Remove duplicates and limit to top 3
    unique_links = []
    seen_urls = set()
    for link in links:
        if link['url'] not in seen_urls:
            unique_links.append(link)
            seen_urls.add(link['url'])
            if len(unique_links) >= 5:
                break
    return unique_links

def process_base64_image(base64_string: str):
    try:
        # Remove data URL prefix if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Create PIL Image from bytes
        image = Image.open(BytesIO(image_bytes))
        
        # Verify image format is supported
        if image.format not in ['PNG', 'JPEG', 'WEBP', 'HEIC', 'HEIF']:
            raise ValueError(f"Unsupported image format: {image.format}")
            
        # Check if image needs resizing (optimize for token usage)
        # Gemini uses 258 tokens if both dimensions <= 384 pixels
        # Larger images are tiled into 768x768 pixel tiles
        max_dimension = max(image.size)
        if max_dimension > 768:
            # Resize maintaining aspect ratio
            ratio = 768.0 / max_dimension
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
        # Ensure image is in correct orientation
        if hasattr(image, '_getexif'):  # Check if image has EXIF data
            try:
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(274)  # 274 is the orientation tag
                    if orientation:
                        # Rotate image according to EXIF orientation
                        if orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)
            except Exception as e:
                print(f"Error processing EXIF data: {e}")
        
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def process_base64_image_for_gemini(base64_string: str) -> bytes:
    """
    Decodes a base64 image string and returns PNG bytes suitable for Gemini API.
    """
    try:
        # Remove data URL prefix if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # Optionally resize or process as needed
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.post("/api/")
async def answer_api(req: QARequest):
    question = req.question.strip()
    image = req.image

    search_query = question
    if image:
        try:
            image_bytes = process_base64_image_for_gemini(image)
            if image_bytes:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        {"role": "user", "parts": [
                            {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
                            {"text": "Describe this image in one sentence, focusing on technical or educational content if present."}
                        ]}
                    ]
                )
                if response:
                    image_desc = response.text
                    search_query = f"{question} {image_desc}"
                    print(f"Image description: {image_desc}")  # Debug log
            else:
                print("Failed to process image")
        except Exception as e:
            print(f"Error processing image for search: {e}")
    
    # Search with combined query
    results = app.state.index.search(search_query, top_k=8)
    context = "\n\n".join([r['text'] for r in results])
    
    if not context:
        answer = "I cannot find relevant information in the available course materials to answer your question. Please check the course content or ask your instructor for clarification."
        links = []
        return JSONResponse({"answer": answer, "links": links})

    # Debug: Print search results
    print("Search results found:", len(results))
    print("First result text sample:", results[0]['text'][:200] if results else "No results")
    
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a Virtual Teaching Assistant for IIT Madras online degree program. "
                    "Provide clear, accurate answers based on course materials. "
                    "Format responses in a structured, easy-to-read manner. "
                    "If information is not in the provided context, state that clearly. "
                    "Do not speculate or provide information beyond the course materials. "
                    "Be concise but thorough."
                )
            ),
            contents=prompt
        )
        answer = response.text if response else "No answer generated."
    except Exception as e:
        answer = f"Error generating answer: {e}"
    
    # Extract and debug links
    links = extract_links(results)
    print("Links found:", len(links))
    print("Links:", links)
        
    return JSONResponse({"answer": answer, "links": links})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="localhost", port=5000, reload=True)
