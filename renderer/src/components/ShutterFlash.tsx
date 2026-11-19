import React from 'react';
import {AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';

/**
 * Flash obturateur (2 à 4 frames) : plein cadre blanc ou noir qui décroît très vite, avec deux
 * lames horizontales qui se rouvrent pour évoquer le rideau de l'obturateur.
 * `warm` : light leak orangé (brûlure de pellicule) qui balaie le cadre en se dissipant.
 */
export const ShutterFlash: React.FC<{color: 'white' | 'black' | 'warm'; frames: number}> = ({color, frames}) => {
  const frame = useCurrentFrame();
  const {height} = useVideoConfig();
  const n = Math.max(2, Math.min(4, frames));
  const t = interpolate(frame, [0, n], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const opacity = interpolate(t, [0, 1], [1, 0], {easing: Easing.out(Easing.quad)});

  if (color === 'warm') {
    const x = interpolate(t, [0, 1], [30, 70]);
    return (
      <AbsoluteFill style={{pointerEvents: 'none', mixBlendMode: 'screen'}}>
        <AbsoluteFill
          style={{
            opacity: Math.min(1, opacity * 1.15),
            background: `radial-gradient(ellipse 85% 120% at ${x}% 45%, rgba(255,248,230,1) 0%, rgba(255,190,90,0.95) 30%, rgba(255,110,40,0.75) 58%, rgba(200,40,10,0.35) 80%, rgba(0,0,0,0) 100%)`,
          }}
        />
      </AbsoluteFill>
    );
  }

  const blade = interpolate(t, [0, 1], [0.5, 0], {easing: Easing.out(Easing.cubic)}) * height;
  const fill = color === 'black' ? '#000' : '#FFF';
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <AbsoluteFill style={{backgroundColor: fill, opacity}} />
      <div style={{position: 'absolute', left: 0, right: 0, top: 0, height: blade, backgroundColor: '#000', opacity: opacity * 0.55}} />
      <div style={{position: 'absolute', left: 0, right: 0, bottom: 0, height: blade, backgroundColor: '#000', opacity: opacity * 0.55}} />
    </AbsoluteFill>
  );
};
