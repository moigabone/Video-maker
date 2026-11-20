# AI Video Editor

Monteur vidéo automatique piloté par IA : on lui donne des rushs, une musique et un brief texte ;
il rend un MP4 monté sur le rythme, avec effets motion design.

1. **Analyse des rushs** — métadonnées (ffprobe), proxies 720p, planche contact envoyée à Claude Vision
   (description, énergie, meilleurs passages), cache par empreinte SHA-256.
2. **Analyse musicale** — librosa : BPM, beats, temps forts, sections intro / montée / climax / fin.
3. **Montage par Claude** — génère `timeline.json`, puis un validateur déterministe cale chaque coupe
   sur le beat le plus proche (≤ 120 ms), sans trou noir ni chevauchement, et répare les erreurs.
4. **Rendu Remotion** — plans plein écran avec zoom punch, mosaïque 3x3 révélée sur les beats,
   HUD caméra (REC, ISO, timecode), flash obturateur ou light leak chaud, titres animés (dont titre « split »).

## Démarrer (sur un ordinateur)

Prérequis : Python ≥ 3.11 + [uv](https://docs.astral.sh/uv/), Node ≥ 20, ffmpeg/ffprobe.

```bash
uv sync && (cd renderer && npm install)
cp .env.example .env                      # puis renseigner ANTHROPIC_API_KEY
# déposer les vidéos dans input/videos/ et la musique dans input/audio/
cp brief.example.txt brief.txt            # puis écrire son brief
uv run python main.py run --input ./input --brief ./brief.txt --output ./output/final.mp4
```

Test complet sans clé API (vidéos synthétiques + vrai rendu) : `uv run pytest -v`.

## Style appris depuis des montages d'exemple

Le monteur imite un *style* appris sur des montages d'exemple « dans le même délire » : rythme des coupes
(en beats), longueur de l'accroche, titres, transitions, étalonnage. Le style **`cinematic-travel`** regroupe
tous les exemples fournis : travel edit 16:9 au grand-angle, accroche avec titre en deux mots écartés
(« REC | JAPAN »), puis un plan par beat, des light leaks chauds et une fin calme ou en boucle.

```bash
uv run python main.py run --input ./input --brief ./brief.txt --style cinematic-travel --format 16:9
uv run python main.py learn-style --name cinematic-travel --video nouvel_exemple.mp4   # enrichir le style
uv run python main.py styles                                                          # lister les styles
```

Architecture, commandes détaillées et conventions : voir [CLAUDE.md](CLAUDE.md).
