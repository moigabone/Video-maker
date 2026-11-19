import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import type {CameraHudData} from '../types';
import {FONT_MONO, unit} from '../types';

const pad2 = (n: number) => String(Math.floor(n)).padStart(2, '0');

/** Timecode SMPTE HH:MM:SS:FF (non drop-frame) à partir d'un numéro de frame absolu. */
export const toTimecode = (frame: number, fps: number): string => {
  const f = Math.max(0, Math.floor(frame));
  const ff = f % fps;
  const totalSec = Math.floor(f / fps);
  return `${pad2(totalSec / 3600)}:${pad2((totalSec % 3600) / 60)}:${pad2(totalSec % 60)}:${pad2(ff)}`;
};

/**
 * Calque vectoriel SVG type viseur caméra : REC clignotant, coins autofocus, réglages d'exposition,
 * timecode dynamique réel (frame de la composition), cadre et vignette légère.
 */
export const CameraHudOverlay: React.FC<{overlay: CameraHudData; startFrame: number; durationInFrames: number}> = ({
  overlay,
  startFrame,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const {fps, width: W, height: H} = useVideoConfig();
  const u = unit(W, H);
  const fadeFrames = Math.min(4, Math.floor(durationInFrames / 2));
  const opacity = fadeFrames > 0
    ? interpolate(frame, [0, fadeFrames, durationInFrames - fadeFrames, durationInFrames], [0, 1, 1, 0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      })
    : 1;
  const recOn = Math.floor((startFrame + frame) / Math.round(fps / 2)) % 2 === 0;
  const tc = toTimecode(startFrame + frame + Math.round((overlay.timecode_offset_sec ?? 0) * fps), fps);

  const m = 56 * u; // marge du cadre
  const stroke = Math.max(1.5, 2.2 * u);
  const fs = 30 * u;
  const white = 'rgba(255,255,255,0.92)';
  const shadow = 'drop-shadow(0 0 2px rgba(0,0,0,0.65))';

  // Coins du cadre et de la zone autofocus centrale
  const corner = (x: number, y: number, dx: number, dy: number, len: number) =>
    `M ${x + dx * len} ${y} L ${x} ${y} L ${x} ${y + dy * len}`;
  const afW = Math.min(W, H) * 0.34;
  const afH = afW * 0.72;
  const afX = (W - afW) / 2;
  const afY = (H - afH) / 2;
  const afLen = 34 * u;
  const frameLen = 70 * u;

  // Échelle d'exposition (-2 … +2) avec un curseur qui respire légèrement
  const scaleW = 360 * u;
  const scaleX = (W - scaleW) / 2;
  const scaleY = H - m - 92 * u;
  const cursor = Math.sin((startFrame + frame) / fps * 1.3) * 0.18;

  return (
    <AbsoluteFill style={{opacity, pointerEvents: 'none'}}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{position: 'absolute', inset: 0, filter: shadow}}>
        <defs>
          <radialGradient id="hud-vignette" cx="50%" cy="50%" r="75%">
            <stop offset="60%" stopColor="#000" stopOpacity={0} />
            <stop offset="100%" stopColor="#000" stopOpacity={0.42} />
          </radialGradient>
        </defs>
        <rect x={0} y={0} width={W} height={H} fill="url(#hud-vignette)" />

        {/* Cadre fin + coins */}
        <rect x={m} y={m} width={W - 2 * m} height={H - 2 * m} fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth={stroke * 0.6} />
        <g stroke={white} strokeWidth={stroke * 1.4} fill="none" strokeLinecap="square">
          <path d={corner(m, m, 1, 1, frameLen)} />
          <path d={corner(W - m, m, -1, 1, frameLen)} />
          <path d={corner(m, H - m, 1, -1, frameLen)} />
          <path d={corner(W - m, H - m, -1, -1, frameLen)} />
        </g>

        {/* Zone autofocus */}
        <g stroke={white} strokeWidth={stroke} fill="none">
          <path d={corner(afX, afY, 1, 1, afLen)} />
          <path d={corner(afX + afW, afY, -1, 1, afLen)} />
          <path d={corner(afX, afY + afH, 1, -1, afLen)} />
          <path d={corner(afX + afW, afY + afH, -1, -1, afLen)} />
          <line x1={W / 2 - 14 * u} y1={H / 2} x2={W / 2 + 14 * u} y2={H / 2} />
          <line x1={W / 2} y1={H / 2 - 14 * u} x2={W / 2} y2={H / 2 + 14 * u} />
        </g>

        {/* REC */}
        {overlay.show_rec ? (
          <g>
            <circle cx={m + 44 * u} cy={m + 48 * u} r={13 * u} fill="#FF2D2D" opacity={recOn ? 1 : 0.15} />
            <text x={m + 68 * u} y={m + 58 * u} fill={white} fontFamily={FONT_MONO} fontSize={fs} fontWeight={700} letterSpacing={2 * u}>
              REC
            </text>
          </g>
        ) : null}

        {/* Timecode */}
        {overlay.show_timecode ? (
          <text x={W / 2} y={m + 58 * u} fill={white} fontFamily={FONT_MONO} fontSize={fs} textAnchor="middle" letterSpacing={1.5 * u}>
            {tc}
          </text>
        ) : null}

        {/* Résolution + batterie */}
        <g>
          <text x={W - m - 110 * u} y={m + 58 * u} fill={white} fontFamily={FONT_MONO} fontSize={fs * 0.82} textAnchor="end">
            {overlay.resolution_label}
          </text>
          <rect x={W - m - 96 * u} y={m + 34 * u} width={58 * u} height={28 * u} rx={4 * u} fill="none" stroke={white} strokeWidth={stroke} />
          <rect x={W - m - 37 * u} y={m + 42 * u} width={6 * u} height={12 * u} fill={white} />
          <rect x={W - m - 91 * u} y={m + 39 * u} width={36 * u} height={18 * u} fill={white} />
        </g>

        {/* Échelle d'exposition */}
        <g stroke={white} strokeWidth={stroke * 0.8}>
          <line x1={scaleX} y1={scaleY} x2={scaleX + scaleW} y2={scaleY} />
          {[-2, -1, 0, 1, 2].map((v) => {
            const x = scaleX + ((v + 2) / 4) * scaleW;
            return <line key={v} x1={x} y1={scaleY - (v === 0 ? 16 : 10) * u} x2={x} y2={scaleY} />;
          })}
        </g>
        <path
          d={`M ${scaleX + (0.5 + cursor / 4) * scaleW - 9 * u} ${scaleY + 16 * u} l ${9 * u} ${-12 * u} l ${9 * u} ${12 * u} z`}
          fill={white}
        />

        {/* Réglages d'exposition */}
        <g fontFamily={FONT_MONO} fontSize={fs} fill={white} fontWeight={600}>
          <text x={m + 34 * u} y={H - m - 30 * u}>{overlay.shutter}</text>
          <text x={W / 2 - 120 * u} y={H - m - 30 * u} textAnchor="middle">{overlay.aperture}</text>
          <text x={W / 2 + 120 * u} y={H - m - 30 * u} textAnchor="middle">{overlay.iso}</text>
          <text x={W - m - 34 * u} y={H - m - 30 * u} textAnchor="end">{overlay.white_balance}</text>
        </g>
      </svg>
    </AbsoluteFill>
  );
};
