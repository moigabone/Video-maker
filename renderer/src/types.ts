// Miroir TypeScript de src/schemas.py (Timeline) + champs ajoutés par src/render.py (`src`).

export type ClipRef = {
  clip_id: string;
  in_sec: number;
  out_sec: number;
  speed: number;
  path?: string | null;
  /** Chemin relatif au dossier public Remotion (ajouté par prepare_render). */
  src?: string;
};

export type SectionLabel = 'intro' | 'buildup' | 'climax' | 'outro';

type SceneBase = {
  start: number;
  end: number;
  section?: SectionLabel | null;
  note?: string | null;
};

export type SingleSceneData = SceneBase & {
  type: 'single';
  clip: ClipRef;
  zoom_punch: boolean;
};

export type NineGridSceneData = SceneBase & {
  type: 'nine_grid';
  cells: ClipRef[];
  reveal: 'all' | 'beats';
  reveal_times: number[];
  border_px: number;
};

export type Scene = SingleSceneData | NineGridSceneData;

export type CameraHudData = {
  type: 'camera_hud';
  start: number;
  end: number;
  aperture: string;
  iso: string;
  shutter: string;
  white_balance: string;
  resolution_label: string;
  show_rec: boolean;
  show_timecode: boolean;
  timecode_offset_sec: number;
};

export type ShutterFlashData = {
  type: 'shutter_flash';
  at: number;
  frames: number;
  color: 'white' | 'black';
};

export type KineticTitleData = {
  type: 'kinetic_title';
  start: number;
  end: number;
  text: string;
  subtitle?: string | null;
  position: 'center' | 'top' | 'bottom' | 'lower_third';
  animation: 'spring_up' | 'scale_in' | 'letter_stagger' | 'fade';
  style: 'minimal' | 'bold' | 'outline';
  color: string;
};

export type Overlay = CameraHudData | ShutterFlashData | KineticTitleData;

export type AudioTrack = {
  audio_id?: string | null;
  path?: string | null;
  audio_start_offset: number;
  volume: number;
  fade_in_sec: number;
  fade_out_sec: number;
  src?: string;
};

export type Timeline = {
  version: number;
  title: string;
  format: '9:16' | '16:9';
  fps: 30 | 60;
  width: number;
  height: number;
  duration_sec: number;
  audio: AudioTrack | null;
  scenes: Scene[];
  overlays: Overlay[];
  created_at?: string;
  generator?: string | null;
};

export type RenderProps = {
  timeline: Timeline;
  previewMode?: boolean;
};

export const toFrame = (seconds: number, fps: number): number => Math.round(seconds * fps);

/** Unité de mise à l'échelle : 1 = 1px pour un petit côté de 1080px. */
export const unit = (width: number, height: number): number => Math.min(width, height) / 1080;

export const FONT_SANS = "Inter, 'Inter Display', 'Helvetica Neue', Helvetica, Arial, sans-serif";
export const FONT_MONO = "'JetBrains Mono', 'DejaVu Sans Mono', 'Liberation Mono', Menlo, monospace";
