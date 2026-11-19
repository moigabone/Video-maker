import React from 'react';
import {AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';

/**
 * Flash obturateur (2 à 4 frames) : plein cadre blanc ou noir qui décroît très vite, avec deux
 * lames horizontales qui se rouvrent pour évoquer le rideau de l'obturateur.
 */
export const ShutterFlash: React.FC<{color: 'white' | 'black'; frames: number}> = ({color, frames}) => {
  const frame = useCurrentFrame();
  const {height} = useVideoConfig();
  const n = Math.max(2, Math.min(4, frames));
  const t = interpolate(frame, [0, n], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const opacity = interpolate(t, [0, 1], [1, 0], {easing: Easing.out(Easing.quad)});
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
