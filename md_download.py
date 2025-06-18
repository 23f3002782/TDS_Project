import os, requests

API_URL = "https://api.github.com/repos/sanand0/tools-in-data-science-public/git/trees/tds-2025-01"

def fetch_json():
    r = requests.get(API_URL).json()
    return [f['path'] for f in r['tree'] if f['path'].endswith('.md')]

def download(paths):
    os.makedirs("tds_course_md", exist_ok=True)
    for p in paths:
        url = f"https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/{p}"
        r = requests.get(url); r.raise_for_status()
        content = r.text
        # Insert YAML frontmatter with web link if not present
        web_link = f"https://tds.s-anand.net/#/{p[:-3]}"
        if not content.lstrip().startswith('---'):
            # Add YAML frontmatter at the top
            yaml = f"---\nsource_url: \"{web_link}\"\n---\n\n"
            content = yaml + content
        out_path = os.path.join("tds_course_md", os.path.basename(p))
        with open(out_path, "w") as f:
            f.write(content)

if __name__ == "__main__":
    files = fetch_json()
    print("Files:", files)
    download(files)
