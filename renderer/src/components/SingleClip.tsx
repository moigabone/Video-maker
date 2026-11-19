import React from 'react';
import {AbsoluteFill, Easing, interpolate, OffthreadVideo, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import type {ClipRef, SingleSceneData} from '../types';
import {toFrame} from '../types';

/** Lecture d'un segment de rush en plein cadre (object-fit: cover), muet : la bande-son vient de la musique. */
export const ClipVideo: React.FC<{clip: ClipRef; style?: React.CSSProperties}> = ({clip, style}) => {
  const {fps} = useVideoConfig();
  if (!clip.src) {
    return <AbsoluteFill style={{backgroundColor: '#111'}} />;
  }
  return (
    <OffthreadVideo
      src={staticFile(clip.src)}
      trimBefore={toFrame(clip.in_sec, fps)}
      playbackRate={clip.speed || 1}
      muted
      pauseWhenBuffering
      style={{width: '100%', height: '100%', objectFit: 'cover', ...style}}
    />
  );
};

/** Plan plein écran avec option zoom_punch : micro punch-in 1.00 → 1.05 sur 4 frames à l'entrée. */
export const SingleClip: React.FC<{scene: SingleSceneData}> = ({scene}) => {
  const frame = useCurrentFrame();
  const scale = scene.zoom_punch
    ? interpolate(frame, [0, 4], [1, 1.05], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
        easing: Easing.out(Easing.cubic),
      })
    : 1;
  return (
    <AbsoluteFill style={{overflow: 'hidden', backgroundColor: '#000'}}>
      <AbsoluteFill style={{transform: `scale(${scale})`, transformOrigin: '50% 50%'}}>
        <ClipVideo clip={scene.clip} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
