#!/usr/bin/env node
// Rendu programmatique Remotion : bundle → selectComposition → renderMedia (H.264 + AAC).
// Usage : node scripts/render.mjs --props render_props.json --public-dir DIR --output out.mp4 [--quality high]
//         [--browser /path/chrome-headless-shell] [--concurrency N]
// Émet des lignes JSON {"event":"progress"|"log", ...} sur stdout (lues par src/render.py).

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {bundle} from '@remotion/bundler';
import {renderMedia, selectComposition} from '@remotion/renderer';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const QUALITY = {
  draft: {crf: 28, jpegQuality: 70, scale: 0.5, x264Preset: 'veryfast'},
  medium: {crf: 23, jpegQuality: 85, scale: 1, x264Preset: 'fast'},
  high: {crf: 18, jpegQuality: 95, scale: 1, x264Preset: 'medium'},
};

const parseArgs = (argv) => {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const [k, inline] = a.slice(2).split('=', 2);
    out[k] = inline ?? argv[i + 1];
    if (inline === undefined) i++;
  }
  return out;
};

const emit = (obj) => process.stdout.write(JSON.stringify(obj) + '\n');
const log = (message) => emit({event: 'log', message});

let lastPct = -1;
let lastStage = '';
const progress = (stage, value) => {
  const pct = Math.floor(value * 100);
  if (pct === lastPct && stage === lastStage) return;
  lastPct = pct;
  lastStage = stage;
  emit({event: 'progress', stage, progress: Math.min(1, Math.max(0, value))});
};

const main = async () => {
  const args = parseArgs(process.argv.slice(2));
  if (!args.props || !args.output) {
    throw new Error('Arguments requis : --props <render_props.json> --output <out.mp4>');
  }
  const preset = QUALITY[args.quality ?? 'high'];
  if (!preset) throw new Error(`Qualité inconnue : ${args.quality}`);
  const inputProps = JSON.parse(fs.readFileSync(path.resolve(args.props), 'utf8'));
  const publicDir = args['public-dir'] ? path.resolve(args['public-dir']) : path.join(ROOT, 'public');
  const browserExecutable = args.browser || process.env.REMOTION_BROWSER_EXECUTABLE || null;
  const outputLocation = path.resolve(args.output);
  const concurrency = args.concurrency ? Number(args.concurrency) : Math.max(1, Math.min(8, Math.floor(os.cpus().length / 2)));

  log(`bundle de ${path.join(ROOT, 'src/index.ts')} (public: ${publicDir})`);
  const serveUrl = await bundle({
    entryPoint: path.join(ROOT, 'src/index.ts'),
    publicDir,
    symlinkPublicDir: true,
    onProgress: (p) => progress('bundling', p / 100),
  });

  const composition = await selectComposition({
    serveUrl,
    id: 'Main',
    inputProps,
    browserExecutable,
    logLevel: 'warn',
  });
  log(`composition ${composition.width}x${composition.height} @ ${composition.fps}fps, ${composition.durationInFrames} frames`);

  await renderMedia({
    serveUrl,
    composition,
    inputProps,
    codec: 'h264',
    audioCodec: 'aac',
    pixelFormat: 'yuv420p',
    imageFormat: 'jpeg',
    jpegQuality: preset.jpegQuality,
    crf: preset.crf,
    x264Preset: preset.x264Preset,
    scale: preset.scale,
    concurrency,
    browserExecutable,
    outputLocation,
    overwrite: true,
    enforceAudioTrack: true,
    logLevel: 'warn',
    timeoutInMilliseconds: 120000,
    onProgress: ({progress: p, stitchStage}) => progress(stitchStage === 'muxing' ? 'muxing' : 'rendering', p),
  });
  progress('done', 1);
  log(`MP4 écrit : ${outputLocation}`);
};

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
