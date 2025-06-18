# TDS_Project

## How to Use This Repository

Follow these steps to set up and use the Virtual TA pipeline:

1. **Download Course Markdown Files**

   - Run `md_download.py` to download all course markdown files and add YAML frontmatter with web links.

   ```bash
   python md_download.py
   ```

2. **Download Discourse Threads**

   - Run `discourse_scrapper.py` to fetch and save Discourse forum threads as markdown files.

   ```bash
   python discourse_scrapper.py
   ```

3. **Generate Image Descriptions**

   - Run `img_description_generator.py` to process markdown files, generate image descriptions using Gemini, and save processed files.

   ```bash
   python img_description_generator.py
   ```

4. **Generate Embeddings**

   - Run `embed.py` (or `final_embed.py` if using the final version) to chunk the processed markdown files and generate embeddings for semantic search.

   ```bash
   python embed.py
   # or
   python final_embed.py
   ```

5. **Start the API Server**
   - Run `run.py` to start the FastAPI server for question answering and link retrieval.
   ```bash
   python run.py
   ```

---

- For more details on each step, refer to the respective script files in the repository.
- Ensure all dependencies in `requirements.txt` are installed before starting.
- The API will be available at `http://localhost:5000/api/` by default.
