# AI Video Editor — pipeline de montage automatique piloté par IA

Éditeur agentique : analyse des rushs (ffprobe + VLM Claude sur planches contact) → analyse musicale
(librosa : BPM, beats, onsets, sections d'énergie) → génération de `timeline.json` par LLM (Claude,
Structured Outputs) → **validateur déterministe** → rendu motion graphics Remotion (React/TS) en MP4.

## Commandes

```bash
uv sync                                   # deps Python (.venv)
(cd renderer && npm install)              # deps Remotion

uv run python main.py analyze --input ./input [--lut ./luts/dlogm.cube] [--skip-vision] [--force-vision]
uv run python main.py direct  --brief ./brief.txt --format 9:16 [--fps 30] [--duration 30] [--offline] [--style rec-japan]
uv run python main.py validate                       # re-valide workspace/timeline.json
uv run python main.py preview                        # Remotion Studio sur la timeline courante (proxies)
uv run python main.py render  --quality high --output ./output/final.mp4 [--no-conform]
uv run python main.py run     --input ./input --brief ./brief.txt --output ./output/final.mp4 [--style touch-grass]
uv run python main.py learn-style --video exemple.mp4 --name mon-style [--end 14.5] [--skip-vision]
uv run python main.py styles                         # liste les styles appris (styles/*.json)

uv run pytest -v                                     # smoke test e2e (sans clé API, ~1 min, rendu Remotion réel → tests/smoke_output.mp4)
(cd renderer && npx tsc --noEmit)                    # typecheck du renderer
```

Option globale : `--workspace DIR` (défaut `./workspace`). `--offline` = brouillon heuristique sans API.

## Architecture

```
main.py                 CLI typer + rich (analyze/direct/validate/preview/render/run)
src/config.py           chemins, .env, modèles (VISION_MODEL / DIRECTOR_MODEL), constantes
src/schemas.py          Pydantic v2 strict (extra=forbid) : MediaCatalog, AudioAnalysis, Timeline, TimelineDraft
src/ffmpeg_utils.py     ffprobe JSON, run ffmpeg, échappement filtre lut3d
src/ingest.py           scan input/videos + input/audio, SHA-256, ffprobe, proxies 720p30 (LUT optionnelle)
src/audio_analyzer.py   librosa → workspace/audio_analysis.json
src/video_indexer.py    keyframes (intervalle 1–1.5s ou scenedetect) → planche contact → Claude Vision → cache SHA-256
src/llm.py              client Anthropic + appel structuré (stream, output_config.format, fallbacks serveur)
src/director.py         brief + catalogue + audio → TimelineDraft (LLM) → Timeline → validate_timeline()
src/style_learner.py    montage d'exemple → StyleProfile (coupes, BPM, alignement beats, look, lecture Claude) → styles/
src/render.py           conform des originaux (+LUT) / proxies → render_public (hard links) + render_props.json → node
renderer/               projet Remotion 4 (TS + React 19)
  src/Root.tsx            <Composition id="Main"> + calculateMetadata (format/fps/durée depuis les props)
  src/MainComposition.tsx scènes + <Audio> (audio_start_offset) + calques d'overlays
  src/components/         SingleClip, NineGridScene, CameraHudOverlay, ShutterFlash, KineticTitle
  scripts/render.mjs      bundle + selectComposition + renderMedia (h264/aac), progression JSON sur stdout
styles/                 styles appris versionnés (rec-japan, touch-grass : exemples fournis par l'utilisateur)
tests/test_smoke_e2e.py fixtures ffmpeg → ingest → audio → validateur (mock) → rendu Remotion → ffprobe → learn-style
```

## Fichiers du workspace (`workspace/`, gitignoré)

| Fichier | Produit par | Contenu |
|---|---|---|
| `media_catalog.json` | analyze | `MediaCatalog` : vidéos (original + proxy + analyse VLM), audios, `analysis_cache` par SHA-256 |
| `audio_analysis.json` | analyze | `AudioAnalysis` : bpm, beats, downbeats, strong_beats, onsets, sections, courbe RMS |
| `timeline.json` | direct | `Timeline` validée (source de vérité du montage) |
| `timeline.draft.json` | direct | sortie brute du LLM (`TimelineDraft`) pour debug |
| `validation_report.json` | direct/validate | corrections appliquées par le validateur (snaps, clamps, substitutions) |
| `proxies/` `contact_sheets/` `conform/` | analyze / render | médias dérivés |
| `render_public/` `render_props.json` | render/preview | dossier public Remotion (hard links : le serveur Remotion refuse les symlinks) + props |

## Schémas clés (src/schemas.py)

- Temps toujours en **secondes (float)** côté Python ; le renderer convertit en frames (`Math.round(t*fps)`).
- `Timeline` : `format` (`9:16`→1080x1920, `16:9`→1920x1080), `fps` (30|60), `duration_sec`, `audio`
  (`AudioTrackRef` avec `audio_start_offset` = position dans la musique au temps 0), `scenes`, `overlays`.
- `scenes` : union discriminée par `type` — `single` (`clip`, `zoom_punch`) | `nine_grid` (`cells` ×9, `reveal` `all|beats`, `reveal_times`).
- `ClipRef` : `clip_id`, `in_sec`, `out_sec`, `speed`, `path` (original absolu, rempli par le validateur).
  Invariant : `out_sec - in_sec == (end - start) * speed`.
- `overlays` : `camera_hud` | `shutter_flash` (`at`, `frames` 2–4, `color` white|black|warm = light leak) |
  `kinetic_title` (`position` center|top|bottom|lower_third|split = deux mots écartés gauche/droite).
- `StyleProfile` (styles/*.json) : `pacing` (plans, plan médian en s et en beats, accroche, % de coupes sur
  beat/demi-beat), `look` (luminance, contraste, saturation, chaleur, flashs), lecture qualitative
  (`summary`, `structure`, `title_style`, `transitions`, `color_grade`, `editing_rules`) injectée dans le
  prompt du directeur via `--style` (section « STYLE REFERENCE »). `qualitative_source` : claude|manual|metrics.
- `TimelineDraft` : projection *plate* de Timeline envoyée au LLM (tous champs requis, aucune union) pour
  rester dans les limites des Structured Outputs ; convertie par `draft_to_timeline()`.

## Invariants garantis par `validate_timeline()` (ne jamais les casser)

1. Chaque `clip_id`/fichier existe (sinon substitution + warning dans le rapport).
2. `0 <= in_sec < out_sec <= clip_duration` (marge d'une frame source).
3. Scènes contiguës : première à 0.0, `scene[i].end == scene[i+1].start`, dernière = `duration_sec` → ni trou noir ni chevauchement.
4. Chaque cut est snappé au beat/onset le plus proche si |Δ| ≤ 120 ms (priorité strong_beats/downbeats > beats > onsets), puis quantifié à la frame.
5. Overlays bornés à `[0, duration_sec]`, flashs snappés sur cut/strong beat.

## Conventions

- Python ≥ 3.11, typage complet, `from __future__ import annotations`, Pydantic v2 `extra="forbid"`.
- Toute nouvelle option de montage = champ dans `schemas.py` + type miroir dans `renderer/src/types.ts` + règle dans le validateur.
- Les binaires externes passent par `src/ffmpeg_utils.py` (jamais de `shell=True`).
- Modèles par défaut : `claude-opus-5-5` (surchargeables via `VISION_MODEL` / `DIRECTOR_MODEL`) ; les appels
  utilisent `output_config.format` (JSON schema via `anthropic.transform_schema`) + fallbacks serveur (`fallbacks="default"`).
- Le rendu final utilise les **originaux** (conform des segments utilisés, LUT appliquée à ce moment) ; la preview utilise les **proxies**.
- Remotion : toutes les dépendances `@remotion/*` à la même version exacte. Navigateur : `REMOTION_BROWSER_EXECUTABLE`
  sinon auto-détection (`/opt/pw-browsers`) sinon téléchargement Remotion.
- Messages CLI en français ; code et identifiants en anglais.

## Styles (montages d'exemple)

- Ne jamais versionner les vidéos d'exemple (droits d'auteur, taille) : seules les fiches `styles/*.json` le sont.
- `learn-style` exclut automatiquement l'écran de fin TikTok/Reels (frames finales sombres et fixes) ; `--end` force la fin utile.
- Un style reproduit un *esprit* (rythme, titres, transitions, structure), jamais le contenu du montage source.
- Les fiches `rec-japan` et `touch-grass` ont des mesures automatiques et une lecture qualitative rédigée à la main (`manual`).

## Pièges connus

- Le serveur de fichiers Remotion refuse les **symlinks** (404) : `render_public/media` contient des hard links (repli copie).
- `anthropic` ≥ 1.x embarque **httpx2** (pas httpx) : les tests simulent Claude via `httpx2.MockTransport` (`tests/conftest.py`).
- Le validateur cale `audio_start_offset` sur le beat le plus proche (≤ 120 ms) : la durée peut perdre quelques ms en fin de musique.
- Les tests partagent `tests/workspace` et dépendent de l'ordre du fichier : lancer la suite complète, pas un test isolé.
- Les avertissements « Memory reported by CGroup » de Remotion en conteneur sont sans conséquence.
