import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import random
import time

with open('./savant.html', "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")
    
rows = soup.select("tbody tr")
pitches = []

for row in rows:
    cells = row.find_all("td")

    # Pitch type
    label_el = row.select_one(".search-pitch-label")
    pitch_type = label_el.get_text(strip=True) if label_el else "UNK"

    # Video link → play_id UUID
    link = row.select_one('a[href*="sporty-videos"]')
    if not link:
        continue
    m = re.search(r"playId=([a-f0-9\-]+)", link.get("href", ""))
    if not m:
        continue
    play_id = m.group(1)

    date   = cells[0].get_text(strip=True)
    batter = cells[5].get_text(strip=True).strip()
    result = cells[12].get_text(strip=True)

    pitches.append({
        "play_id":    play_id,
        "pitch_type": pitch_type,
        "date":       date,
        "batter":     batter,
        "result":     result,
    })

def get_video_url(play_id: str) -> str | None:
    page_url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
    resp = requests.get(page_url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    match = re.search(r"https://sporty-clips\.mlb\.com/[\w=]+\.mp4", resp.text)
    return match.group(0) if match else None

def download_clip(play_id: str, out_path: str):
    video_url = get_video_url(play_id)
    if not video_url:
        print(f"No video found for {play_id}")
        return
    r = requests.get(video_url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved {out_path}")

random_sample = random.sample(pitches, k=200)

for x in random_sample:
    download_clip(x['play_id'], './data/videos/pitch_{}.mp4'.format(x['play_id']))
    time.sleep(random.randint(1,7))
