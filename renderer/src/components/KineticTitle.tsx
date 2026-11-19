import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import type {KineticTitleData} from '../types';
import {FONT_SANS, unit} from '../types';

const STYLE_PRESETS: Record<KineticTitleData['style'], (u: number, color: string) => React.CSSProperties> = {
  minimal: (u, color) => ({fontSize: 76 * u, fontWeight: 300, letterSpacing: '0.22em', textTransform: 'uppercase', color}),
  bold: (u, color) => ({fontSize: 132 * u, fontWeight: 800, letterSpacing: '-0.02em', textTransform: 'uppercase', color,
    textShadow: `0 ${6 * u}px ${24 * u}px rgba(0,0,0,0.45)`}),
  outline: (u, color) => ({fontSize: 132 * u, fontWeight: 800, letterSpacing: '0.02em', textTransform: 'uppercase',
    color: 'transparent', WebkitTextStroke: `${3 * u}px ${color}`}),
};

const POSITION: Record<KineticTitleData['position'], React.CSSProperties> = {
  center: {justifyContent: 'center', alignItems: 'center', textAlign: 'center'},
  top: {justifyContent: 'flex-start', alignItems: 'center', textAlign: 'center', paddingTop: '16%'},
  bottom: {justifyContent: 'flex-end', alignItems: 'center', textAlign: 'center', paddingBottom: '18%'},
  lower_third: {justifyContent: 'flex-end', alignItems: 'flex-start', textAlign: 'left', paddingBottom: '24%', paddingLeft: '8%'},
};

/** Titre minimaliste animé (spring / interpolate) avec sortie en fondu. */
export const KineticTitle: React.FC<{overlay: KineticTitleData; durationInFrames: number}> = ({overlay, durationInFrames}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const u = unit(width, height);
  const enter = spring({frame, fps, durationInFrames: Math.min(20, durationInFrames), config: {damping: 200, stiffness: 140, mass: 0.7}});
  const exitStart = Math.max(1, durationInFrames - Math.min(8, Math.floor(durationInFrames / 3)));
  const exit = interpolate(frame, [exitStart, durationInFrames], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const base = STYLE_PRESETS[overlay.style](u, overlay.color || '#FFFFFF');

  let transform = 'none';
  let opacity = enter * exit;
  let letterSpacing = base.letterSpacing;
  switch (overlay.animation) {
    case 'spring_up':
      transform = `translateY(${(1 - enter) * 70 * u}px)`;
      break;
    case 'scale_in':
      transform = `scale(${0.82 + 0.18 * enter})`;
      letterSpacing = `${interpolate(enter, [0, 1], [0.5, 0.18])}em`;
      break;
    case 'fade':
      opacity = interpolate(frame, [0, 10], [0, 1], {extrapolateRight: 'clamp'}) * exit;
      break;
    case 'letter_stagger':
      opacity = exit;
      break;
  }

  const text =
    overlay.animation === 'letter_stagger'
      ? Array.from(overlay.text).map((ch, i) => {
          const s = spring({frame: frame - i * 2, fps, durationInFrames: 14, config: {damping: 18, stiffness: 200, mass: 0.5}});
          return (
            <span key={i} style={{display: 'inline-block', opacity: s, transform: `translateY(${(1 - s) * 40 * u}px)`, whiteSpace: 'pre'}}>
              {ch}
            </span>
          );
        })
      : overlay.text;

  const subtitleOpacity = interpolate(frame, [6, 18], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}) * exit;

  return (
    <AbsoluteFill style={{display: 'flex', flexDirection: 'column', fontFamily: FONT_SANS, padding: `0 ${64 * u}px`, ...POSITION[overlay.position]}}>
      <div style={{transform, opacity}}>
        {overlay.position === 'lower_third' ? (
          <div style={{width: 90 * u * enter, height: 6 * u, backgroundColor: overlay.color || '#FFF', marginBottom: 18 * u}} />
        ) : null}
        <div style={{...base, letterSpacing, lineHeight: 1.05}}>{text}</div>
      </div>
      {overlay.subtitle ? (
        <div style={{marginTop: 22 * u, fontSize: 34 * u, fontWeight: 400, letterSpacing: '0.3em', textTransform: 'uppercase',
          color: overlay.color || '#FFF', opacity: subtitleOpacity * 0.85}}>
          {overlay.subtitle}
        </div>
      ) : null}
    </AbsoluteFill>
  );
};
