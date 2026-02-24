import os, sys, subprocess, tempfile, json, re, time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

def pip_install(pkg):
    subprocess.call([sys.executable, "-m", "pip", "install", pkg])

try:
    import yt_dlp
except:
    pip_install("yt-dlp")
    import yt_dlp

try:
    import whisper
except:
    pip_install("openai-whisper")
    import whisper

try:
    from transformers import pipeline
except:
    pip_install("transformers sentencepiece torch")
    from transformers import pipeline

def ask(prompt, default=None):
    x = input(f"{prompt} " + (f"[{default}] " if default else ": "))
    return x.strip() if x.strip() != "" else default

url = ask("Enter YouTube URL:")
clips_required = int(ask("How many context reels?", "3"))
min_len = int(ask("Minimum seconds per clip", "25"))
max_len = int(ask("Maximum seconds per clip", "45"))
fps = int(ask("FPS output", "30"))
out_dir = ask("Output folder", "reels_output")

Path(out_dir).mkdir(exist_ok=True)
tmp = Path(tempfile.mkdtemp())

# === Download video (yt-dlp) ===
print("\nDownloading video...")
ydl_opts = {"format": "bestvideo+bestaudio/best", "outtmpl": str(tmp / "video.%(ext)s")}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)
video_file = list(tmp.glob("video.*"))[0]

# === Try captions ===
transcript_file = tmp / "transcript.txt"
print("\nExtracting transcript...")
try:
    sub_opts = {
        "skip_download": True,
        "writeauto subtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "outtmpl": str(tmp / "subs"),
    }
    with yt_dlp.YoutubeDL(sub_opts) as ydl:
        ydl.extract_info(url, download=False)

    vtt = list(tmp.glob("subs.en.vtt"))[0]
    raw = Path(vtt).read_text(encoding="utf-8")
    raw = re.sub(r"WEBVTT.*?\n\n", "", raw, flags=re.S)
    raw = re.sub(r"\d+\n\d\d:.*? --> .*?\n", "", raw)
    raw = raw.replace("\n\n", " ").replace("\n", " ")
    transcript_file.write_text(raw, encoding="utf-8")
except:
    print("No captions — running Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(str(video_file))
    transcript_file.write_text(result["text"], encoding="utf-8")

txt = transcript_file.read_text(encoding="utf-8")
sentences = re.split(r'(?<=[.?!]) +', txt)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# === Sentence embeddings for context grouping ===
emb = pipeline("feature-extraction", model="sentence-transformers/all-mpnet-base-v2")

sentence_data = []
for s in sentences:
    if len(s) > 5:
        sentence_data.append({"s": s, "vec": emb(s)[0][0]})

# === Group into meaningful segments =
segments = []
chunk = []
current_len = 0

for s in sentence_data:
    approx = max(4, len(s["s"].split()) * 0.6)
    if current_len + approx > max_len:
        if current_len >= min_len:
            segments.append(chunk)
        chunk = []
        current_len = 0
    chunk.append(s["s"])
    current_len += approx

if chunk and current_len >= min_len:
    segments.append(chunk)

if len(segments) == 0:
    segments = [sentences[:int(len(sentences) * 0.2)]]

chosen = segments[:clips_required]

# === Estimate timestamps (word count → seconds) ===
def ts(sentence, full):
    idx = full.find(sentence)
    if idx < 0:
        return 0
    return len(full[:idx].split()) * 0.6

results = []

print("\nGenerating reels...\n")

# get video duration using ffprobe
def get_duration(path):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "{path}"'
    out = subprocess.check_output(cmd, shell=True).decode().strip()
    return float(out)

duration = get_duration(str(video_file))

for i, seg in enumerate(chosen):
    start = ts(seg[0], txt)
    end = ts(seg[-1], txt) + max_len
    start = max(0, start)
    end = min(duration, end)

    out_path = Path(out_dir) / f"reel_{i+1}.mp4"

    cmd = f'ffmpeg -y -i "{video_file}" -vf "crop=in_w*9/16:in_h,scale=1080:1920" -ss {start} -to {end} -r {fps} "{out_path}"'
    subprocess.call(cmd, shell=True)

    results.append(str(out_path))

print("\nReels generated:")
for r in results:
    print(r)

print("\n✅ Done.\n")
