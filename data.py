import json
import requests
from bs4 import BeautifulSoup
import time

# Load URLs
with open('urls.json', 'r') as f:
    urls = json.load(f)

print(f"Loaded {len(urls)} URLs")

# Simple function to get page content
def get_page_content(url):
    try:
        response = requests.get(url, timeout=20)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        
        # Get clean text
        text = soup.get_text()
        clean_text = ' '.join(text.split())
        
        return clean_text
    except:
        return f"Error loading page: {url}"

# Process all URLs and create markdown
markdown_content = "# Manappuram Website Content\n\n"

for i, url in enumerate(urls):
    print(f"Processing {i+1}/{len(urls)}: {url}")
    
    content = get_page_content(url)
    
    markdown_content += f"## Source: {url}\n\n"
    markdown_content += f"{content}\n\n"
    markdown_content += "---\n\n"
    
    time.sleep(2)  

# Save to file
with open('manappuram_content.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print("Content saved to manappuram_content.md")