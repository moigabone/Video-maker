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

## Styles appris depuis des montages d'exemple

Le monteur peut imiter le *style* d'un montage existant : rythme des coupes (en beats), longueur de
l'accroche, titres, transitions, étalonnage. Deux styles sont fournis :

| Style | Esprit |
|---|---|
| `rec-japan` | ville de nuit, néons, rafale de plans d'½ à 1 beat, light leaks chauds, retour en boucle au plan d'ouverture |
| `touch-grass` | nature lumineuse, POV en accroche longue, un plan par beat, cuts secs, fin sur un plan calme |

```bash
uv run python main.py run --input ./input --brief ./brief.txt --style rec-japan --format 16:9
uv run python main.py learn-style --video mon_exemple.mp4 --name mon-style   # apprendre un nouveau style
uv run python main.py styles                                                 # lister les styles
```

Architecture, commandes détaillées et conventions : voir [CLAUDE.md](CLAUDE.md).
