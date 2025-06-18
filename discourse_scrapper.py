import os
import json
import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_JSON = "/c/courses/tds-kb/34.json"
OUTPUT_DIR = "discourse_threads_md"

def load_cookie(path="cookie.txt"):
    with open(path, "r") as f:
        return f.read().strip()

def html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text("\n").strip()

def extract_image_urls(html: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    
    # Method 1: Look for lightbox links (preferred - gets original full-size images)
    lightbox_links = soup.find_all("a", class_="lightbox")
    for link in lightbox_links:
        href = link.get("href")
        if href and "/uploads/" in href and "/original/" in href:
            if href.startswith("/"):
                href = BASE_URL + href
            urls.append(href)
    
    # Method 2: Fallback to img tags if no lightbox found
    if not urls:
        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            if src.startswith("/"):
                src = BASE_URL + src
            
            # Filter to only include uploaded content images
            # Exclude avatars and emojis
            if "/uploads/" in src and ("/original/" in src or "/optimized/" in src):
                urls.append(src)
    
    return urls

def get_all_topics(cookie: str) -> list:
    headers = {"Cookie": cookie, "User-Agent": "Mozilla/5.0"}
    topics = []
    page = 1
    while True:
        url = f"{BASE_URL}{CATEGORY_JSON}?page={page}"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"→ Stopped at page {page}: HTTP {resp.status_code}")
            break
        data = resp.json().get("topic_list", {})
        batch = data.get("topics", [])
        if not batch:
            print("→ No more topics.")
            break
        topics.extend(batch)
        print(f"Fetched page {page}, {len(batch)} topics.")
        page += 1
        time.sleep(1)
    return topics

def filter_topics(topics: list,
                  start_str="2025-01-01T00:00:00+00:00",
                  end_str=  "2025-04-30T23:59:59+00:00") -> list:
    start = datetime.fromisoformat(start_str)
    end   = datetime.fromisoformat(end_str)
    out = []
    for t in topics:
        dt = datetime.fromisoformat(t["created_at"].replace("Z","+00:00"))
        if start <= dt <= end:
            out.append(t)
    print(f"Filtered down to {len(out)} topics between {start.date()} and {end.date()}.")
    return out

import requests
import time

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"

def fetch_posts_for_topic(topic_id: int, cookie: str, batch_size: int = 20):
    """
    Returns a sorted list of all post objects for the given topic.
    """
    session = requests.Session()
    session.headers.update({
        "Cookie": cookie,
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    })

    # 1) Initial fetch
    url_topic = f"{BASE_URL}/t/{topic_id}.json"
    r = session.get(url_topic)
    r.raise_for_status()
    data = r.json()
    initial_posts = data["post_stream"]["posts"]
    stream_ids      = data["post_stream"]["stream"]  # all IDs

    # 2) Determine which ones we still need
    fetched_ids = {post["id"] for post in initial_posts}
    missing_ids = [pid for pid in stream_ids if pid not in fetched_ids]

    # 3) Batch‑fetch missing posts
    extra_posts = []
    for i in range(0, len(missing_ids), batch_size):
        batch = missing_ids[i : i + batch_size]
        params = [("post_ids[]", pid) for pid in batch]
        params.append(("include_suggested", "true"))
        url_posts = f"{BASE_URL}/t/{topic_id}/posts.json"
        r2 = session.get(url_posts, params=params)
        r2.raise_for_status()
        batch_posts = r2.json()["post_stream"]["posts"]
        extra_posts.extend(batch_posts)
        time.sleep(0.5)  # stay polite

    # 4) Combine & sort by post_number
    all_posts = initial_posts + extra_posts
    all_posts.sort(key=lambda p: p["post_number"])
    return all_posts


def render_topic_to_md(topic_meta: dict, posts: list, outdir: str):
    """Renders one .md file per topic."""
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"tds-kb-{topic_meta['id']}.md")
    with open(fname, "w", encoding="utf-8") as f:
        # Front matter
        f.write("---\n")
        f.write(f"topic_id: {topic_meta['id']}\n")
        f.write(f"title: \"{topic_meta['fancy_title']}\"\n")
        f.write(f"discourse_url: \"{BASE_URL}/t/{topic_meta['slug']}/{topic_meta['id']}\"\n")
        f.write("---\n\n")
        # Posts
        for post in posts:
            num = post["post_number"]
            author = post["username"]
            ts = post["created_at"]
            text = html_to_text(post["cooked"])
            images = extract_image_urls(post["cooked"])
            label = "Q" if num == 1 else "A"
            f.write(f"### {label}{num} by {author} ({ts})\n\n")
            for line in text.split("\n"):
                f.write(f"> {line}\n")
            f.write("\n")
            if images:
                for url in images:
                    f.write(f"![image]({url})\n\n")
            f.write("---\n\n")
    print(f"  • Wrote {fname}")

def main():
    cookie = load_cookie()
    
    # 1. Fetch & save raw topic list
    all_topics_file = "all_topics.json"
    if os.path.exists(all_topics_file):
        print("Loading existing all_topics.json...")
        with open(all_topics_file, "r") as f:
            all_topics = json.load(f)
    else:
        print("Fetching all topics...")
        all_topics = get_all_topics(cookie)
        with open(all_topics_file, "w") as f:
            json.dump(all_topics, f, indent=4)

    # 2. Filter by date and save filtered topics
    filtered_topics_file = "filtered_topics.json"
    if os.path.exists(filtered_topics_file):
        print("Loading existing filtered_topics.json...")
        with open(filtered_topics_file, "r") as f:
            filtered = json.load(f)
    else:
        print("Filtering topics by date...")
        filtered = filter_topics(all_topics)
        with open(filtered_topics_file, "w") as f:
            json.dump(filtered, f, indent=4)

    # 3. Fetch posts & render MD
    # You can skip any of these steps by commenting them out
    for topic in filtered:
        tid = topic["id"]
        output_file = os.path.join(OUTPUT_DIR, f"tds-kb-{tid}.md")
        
        if os.path.exists(output_file):
            print(f"Skipping topic {tid} - already exists")
            continue
            
        print(f"Processing topic {tid}...")
        posts = fetch_posts_for_topic(tid, cookie)
        render_topic_to_md(topic, posts, OUTPUT_DIR)
        time.sleep(0.5)
    
    print("\n✅ All done — Markdown files in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
