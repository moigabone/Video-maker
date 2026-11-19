import React from 'react';
import {CalculateMetadataFunction, Composition} from 'remotion';
import {MainComposition} from './MainComposition';
import type {RenderProps} from './types';

// Props par défaut (sans média) pour ouvrir Studio sans timeline : la vraie timeline arrive via --props.
const defaultProps: RenderProps = {
  previewMode: true,
  timeline: {
    version: 1,
    title: 'AI Video Editor',
    format: '9:16',
    fps: 30,
    width: 1080,
    height: 1920,
    duration_sec: 4,
    audio: null,
    scenes: [],
    overlays: [
      {
        type: 'camera_hud',
        start: 0,
        end: 4,
        aperture: 'F 4.0',
        iso: 'ISO 800',
        shutter: '1/1000',
        white_balance: '5600K',
        resolution_label: '4K 60P',
        show_rec: true,
        show_timecode: true,
        timecode_offset_sec: 0,
      },
      {
        type: 'kinetic_title',
        start: 0.3,
        end: 3.7,
        text: 'Aucune timeline',
        subtitle: 'python main.py preview',
        position: 'center',
        animation: 'spring_up',
        style: 'minimal',
        color: '#FFFFFF',
      },
      {type: 'shutter_flash', at: 0, frames: 3, color: 'white'},
    ],
  },
};

const calculateMetadata: CalculateMetadataFunction<RenderProps> = ({props}) => {
  const t = props.timeline;
  const fps = t.fps === 60 ? 60 : 30;
  const [width, height] = t.format === '16:9' ? [1920, 1080] : [1080, 1920];
  return {
    fps,
    width,
    height,
    durationInFrames: Math.max(1, Math.round(t.duration_sec * fps)),
  };
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Main"
      component={MainComposition}
      durationInFrames={120}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={defaultProps}
      calculateMetadata={calculateMetadata}
    />
  );
};
