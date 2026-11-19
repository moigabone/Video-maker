# AI Video Editor

Monteur vidéo automatique piloté par IA : on lui donne des rushs, une musique et un brief texte ;
il rend un MP4 monté sur le rythme, avec effets motion design.

1. **Analyse des rushs** — métadonnées (ffprobe), proxies 720p, planche contact envoyée à Claude Vision
   (description, énergie, meilleurs passages), cache par empreinte SHA-256.
2. **Analyse musicale** — librosa : BPM, beats, temps forts, sections intro / montée / climax / fin.
3. **Montage par Claude** — génère `timeline.json`, puis un validateur déterministe cale chaque coupe
   sur le beat le plus proche (≤ 120 ms), sans trou noir ni chevauchement, et répare les erreurs.
4. **Rendu Remotion** — plans plein écran avec zoom punch, mosaïque 3x3 révélée sur les beats,
   HUD caméra (REC, ISO, timecode), flash obturateur, titres animés.

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

Architecture, commandes détaillées et conventions : voir [CLAUDE.md](CLAUDE.md).
