import React from 'react';
import {AbsoluteFill, Sequence, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import type {ClipRef, NineGridSceneData} from '../types';
import {toFrame, unit} from '../types';
import {ClipVideo} from './SingleClip';

const Cell: React.FC<{clip: ClipRef; revealFrame: number; index: number; animate: boolean}> = ({clip, revealFrame, index, animate}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const local = frame - revealFrame;
  const pop = animate ? spring({frame: local, fps, durationInFrames: 8, config: {damping: 14, stiffness: 220, mass: 0.5}}) : 1;
  const visible = local >= 0;
  return (
    <div style={{position: 'relative', overflow: 'hidden', backgroundColor: '#000'}}>
      {visible ? (
        <div style={{position: 'absolute', inset: 0, transform: `scale(${0.86 + 0.14 * pop})`, opacity: Math.min(1, 0.2 + pop)}}>
          {/* La vidéo démarre à in_sec au moment où la case apparaît */}
          <Sequence from={revealFrame} layout="none" name={`cell ${index + 1}`}>
            <ClipVideo clip={clip} />
          </Sequence>
        </div>
      ) : null}
    </div>
  );
};

/** Mosaïque 3x3 à fines bordures noires ; les cases apparaissent ensemble ou une par beat (`reveal_times`). */
export const NineGridScene: React.FC<{scene: NineGridSceneData}> = ({scene}) => {
  const {fps, width, height} = useVideoConfig();
  const cells = scene.cells.slice(0, 9);
  while (cells.length < 9 && scene.cells.length > 0) cells.push(scene.cells[cells.length % scene.cells.length]);
  const border = Math.round((scene.border_px ?? 6) * unit(width, height));
  const byBeats = scene.reveal === 'beats';
  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#000',
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gridTemplateRows: 'repeat(3, 1fr)',
        gap: border,
        padding: border,
      }}
    >
      {cells.map((clip, i) => (
        <Cell
          key={i}
          index={i}
          clip={clip}
          animate={byBeats}
          revealFrame={byBeats ? toFrame(scene.reveal_times[i] ?? 0, fps) : 0}
        />
      ))}
    </AbsoluteFill>
  );
};
