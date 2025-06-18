import os
import re
import json
import requests
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import time
from urllib.parse import urlparse
from google import genai
from PIL import Image
import io

class MarkdownImageProcessor:
    def __init__(self, api_key: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the processor with Gemini API key and chunking parameters.
        
        Args:
            api_key: Google Gemini API key
            chunk_size: Target size for text chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
        """
        self.client = genai.Client(api_key=api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def extract_images_from_markdown(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract image URLs and their alt text from markdown content.
        
        Returns:
            List of tuples: (alt_text, image_url)
        """
        # Pattern to match ![alt](url) format
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(pattern, content)
        return matches
    
    def download_image(self, url: str) -> str:
        """
        Download image from URL to temporary file.
        
        Returns:
            Path to temporary file
        """
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file extension from URL or content type
            parsed_url = urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1]
            if not ext:
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                elif 'gif' in content_type:
                    ext = '.gif'
                elif 'webp' in content_type:
                    ext = '.webp'
                else:
                    ext = '.jpg'  # default
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            print(f"Error downloading image {url}: {e}")
            return None
    
    def convert_image_to_webp(self, image_path: str) -> str:
        """
        Convert image to WebP format.
        
        Args:
            image_path: Path to the original image file
            
        Returns:
            Path to the converted WebP file
        """
        try:
            # Open the original image
            with Image.open(image_path) as img:
                # Convert image to RGB mode if not already in that mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as WebP format
                webp_path = f"{os.path.splitext(image_path)[0]}.webp"
                img.save(webp_path, 'webp')
                
            return webp_path
            
        except Exception as e:
            print(f"Error converting image to WebP {image_path}: {e}")
            return image_path  # Return original path on error
    
    def convert_webp_to_png(self, webp_path: str) -> str:
        """
        Convert WebP image to PNG format.
        
        Args:
            webp_path: Path to the WebP image file
            
        Returns:
            Path to the converted PNG file
        """
        try:
            # Open the WebP image
            with Image.open(webp_path) as img:
                # Create a new temporary file for PNG
                png_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                png_path = png_file.name
                png_file.close()
                
                # Convert and save as PNG
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    background.convert('RGB').save(png_path, 'PNG')
                else:
                    img.convert('RGB').save(png_path, 'PNG')
                
                return png_path
        except Exception as e:
            print(f"Error converting WebP to PNG: {e}")
            return None

    def generate_image_description(self, image_path: str, context: str = "") -> str:
        """
        Generate description for image using Gemini API.
        
        Args:
            image_path: Path to image file
            context: Additional context about the image
            
        Returns:
            Generated description
        """
        try:
            # Upload image to Gemini
            my_file = self.client.files.upload(file=image_path)
            
            # Create prompt with context
            prompt = f"""Analyze this image and provide a detailed description that would be useful for search and understanding. 
            
Context: {context if context else 'This image is from an educational document.'}

Please describe:
1. What the image shows (charts, diagrams, screenshots, etc.)
2. Key elements, text, or data visible
3. The purpose or educational value
4. Any specific technical details

Keep the description factual and comprehensive."""

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[my_file, prompt],
            )
            
            # Clean up uploaded file from Gemini
            try:
                self.client.files.delete(my_file.name)
            except:
                pass
                
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating description for {image_path}: {e}")
            return f"[Image description unavailable: {str(e)}]"
    
    def create_text_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for better embedding.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('!', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('?', start, end)
                if sentence_end == -1:
                    # Look for paragraph breaks
                    sentence_end = text.rfind('\n\n', start, end)
                if sentence_end == -1:
                    # Look for any line break
                    sentence_end = text.rfind('\n', start, end)
                
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            if start >= len(text):
                break
        
        return chunks
    
    def process_markdown_file(self, file_path: str, output_dir: str = None) -> Dict:
        """
        Process a single markdown file: extract images, generate descriptions, create chunks.
        
        Args:
            file_path: Path to markdown file
            output_dir: Directory to save processed file (if None, overwrites original)
            
        Returns:
            Dictionary with processing results
        """
        print(f"Processing: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract images
        images = self.extract_images_from_markdown(content)
        print(f"Found {len(images)} images")
        
        # Process each image
        processed_content = content
        image_descriptions = {}
        
        for alt_text, image_url in images:
            print(f"Processing image: {image_url}")
            
            # Download image
            temp_image_path = self.download_image(image_url)
            if not temp_image_path:
                continue
            
            try:
                # If the image is WebP, convert it to PNG first
                if temp_image_path and temp_image_path.lower().endswith('.webp'):
                    png_path = self.convert_webp_to_png(temp_image_path)
                    if png_path:
                        # Clean up the WebP file
                        try:
                            os.unlink(temp_image_path)
                        except:
                            pass
                        temp_image_path = png_path

                # Generate description
                description = self.generate_image_description(
                    temp_image_path, 
                    context=f"Alt text: {alt_text}" if alt_text else ""
                )
                
                image_descriptions[image_url] = description
                
                # Replace image in markdown with description
                new_text = f"""**[Image Description]**: {description}

*Original image: ![{alt_text}]({image_url})*"""
                
                processed_content = processed_content.replace(
                    f"![{alt_text}]({image_url})", 
                    new_text
                )
                
                print(f"Generated description for {image_url[:50]}...")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except:
                    pass
            
            # Be nice to the API
            time.sleep(1)
        
        # Create chunks
        chunks = self.create_text_chunks(processed_content)
        
        # Save processed file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(file_path))
        else:
            output_path = file_path
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        return {
            'file_path': file_path,
            'output_path': output_path,
            'images_processed': len(image_descriptions),
            'image_descriptions': image_descriptions,
            'chunks': chunks,
            'chunk_count': len(chunks)
        }
    
    def process_directory(self, input_dir: str, output_dir: str = None) -> List[Dict]:
        """
        Process all markdown files in a directory.
        
        Args:
            input_dir: Directory containing markdown files
            output_dir: Directory to save processed files (if None, overwrites originals)
            
        Returns:
            List of processing results for each file
        """
        results = []
        
        for file_path in Path(input_dir).glob("*.md"):
            try:
                result = self.process_markdown_file(str(file_path), output_dir)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append({
                    'file_path': str(file_path),
                    'error': str(e)
                })
        
        return results

def main():
    # Configuration
    GOOGLE_API_KEY = "AIzaSyBJGNOZw43-1YerD3bNS04vnPrx_iJS_CE"  # Replace with your actual API key
    TDS_COURSE_DIR = "tds_course_md"  # Replace with your TDS course directory
    DISCOURSE_DIR = "discourse_threads_md"  # Replace with your discourse directory
    OUTPUT_DIR = "processed_documents"  # Output directory for processed files
    
    # Initialize processor
    processor = MarkdownImageProcessor(
        api_key=GOOGLE_API_KEY,
        chunk_size=1000,  # Adjust based on your needs
        chunk_overlap=100
    )
    
    # Process both directories
    all_results = []
    
    # Check if output folders already exist and skip processing if they do
    tds_output_dir = os.path.join(OUTPUT_DIR, "tds_course")
    discourse_output_dir = os.path.join(OUTPUT_DIR, "discourse")

    if os.path.exists(tds_output_dir):
        print("=" * 50)
        print("TDS Course Documents already processed. Skipping.")
        print("=" * 50)
    else:
        print("=" * 50)
        print("Processing TDS Course Documents")
        print("=" * 50)
        tds_results = processor.process_directory(
            TDS_COURSE_DIR, 
            tds_output_dir
        )
        all_results.extend(tds_results)

    if os.path.exists(discourse_output_dir):
        print("=" * 50)
        print("Discourse Documents already processed. Skipping.")
        print("=" * 50)
    else:
        print("=" * 50)
        print("Processing Discourse Documents")
        print("=" * 50)
        discourse_results = processor.process_directory(
            DISCOURSE_DIR, 
            discourse_output_dir
        )
        all_results.extend(discourse_results)
    
    # Save processing summary
    summary = {
        'total_files_processed': len(all_results),
        'total_images_processed': sum(r.get('images_processed', 0) for r in all_results),
        'total_chunks_created': sum(r.get('chunk_count', 0) for r in all_results),
        'files': all_results
    }
    
    with open(os.path.join(OUTPUT_DIR, "processing_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Files processed: {summary['total_files_processed']}")
    print(f"Images processed: {summary['total_images_processed']}")
    print(f"Total chunks created: {summary['total_chunks_created']}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()