import React from 'react';
import {AbsoluteFill, Audio, interpolate, Sequence, staticFile, useVideoConfig} from 'remotion';
import {CameraHudOverlay} from './components/CameraHudOverlay';
import {KineticTitle} from './components/KineticTitle';
import {NineGridScene} from './components/NineGridScene';
import {ShutterFlash} from './components/ShutterFlash';
import {SingleClip} from './components/SingleClip';
import {AudioTrack, RenderProps, toFrame} from './types';

const MusicTrack: React.FC<{audio: AudioTrack; totalFrames: number}> = ({audio, totalFrames}) => {
  const {fps} = useVideoConfig();
  if (!audio.src) return null;
  const vol = audio.volume ?? 1;
  const fadeIn = Math.max(0, Math.round((audio.fade_in_sec ?? 0) * fps));
  const fadeOut = Math.max(0, Math.round((audio.fade_out_sec ?? 0) * fps));
  const volume = (f: number): number => {
    let v = vol;
    if (fadeIn > 0) v *= interpolate(f, [0, fadeIn], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
    if (fadeOut > 0) {
      v *= interpolate(f, [totalFrames - fadeOut, totalFrames], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
    }
    return v;
  };
  return <Audio src={staticFile(audio.src)} trimBefore={toFrame(audio.audio_start_offset, fps)} volume={volume} />;
};

export const MainComposition: React.FC<RenderProps> = ({timeline}) => {
  const {fps, durationInFrames} = useVideoConfig();
  const f = (t: number) => toFrame(t, fps);

  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      {/* Piste vidéo : scènes contiguës (garanties par le validateur Python) */}
      {timeline.scenes.map((scene, i) => {
        const from = f(scene.start);
        const duration = f(scene.end) - from;
        if (duration <= 0) return null;
        return (
          <Sequence key={`scene-${i}`} from={from} durationInFrames={duration} premountFor={Math.round(fps / 2)}
            name={`${scene.type} #${i}`}>
            {scene.type === 'single' ? <SingleClip scene={scene} /> : <NineGridScene scene={scene} />}
          </Sequence>
        );
      })}

      {/* Musique synchronisée */}
      {timeline.audio ? <MusicTrack audio={timeline.audio} totalFrames={durationInFrames} /> : null}

      {/* Calques d'effets : HUD → titres → flashs (au-dessus de tout) */}
      {timeline.overlays.map((ov, i) => {
        if (ov.type !== 'camera_hud') return null;
        const from = f(ov.start);
        const duration = Math.max(1, f(ov.end) - from);
        return (
          <Sequence key={`hud-${i}`} from={from} durationInFrames={duration} name="camera_hud">
            <CameraHudOverlay overlay={ov} startFrame={from} durationInFrames={duration} />
          </Sequence>
        );
      })}
      {timeline.overlays.map((ov, i) => {
        if (ov.type !== 'kinetic_title') return null;
        const from = f(ov.start);
        const duration = Math.max(1, f(ov.end) - from);
        return (
          <Sequence key={`title-${i}`} from={from} durationInFrames={duration} name={`title: ${ov.text}`}>
            <KineticTitle overlay={ov} durationInFrames={duration} />
          </Sequence>
        );
      })}
      {timeline.overlays.map((ov, i) => {
        if (ov.type !== 'shutter_flash') return null;
        const frames = Math.min(4, Math.max(2, Math.round(ov.frames)));
        return (
          <Sequence key={`flash-${i}`} from={f(ov.at)} durationInFrames={frames} name="shutter_flash">
            <ShutterFlash color={ov.color} frames={frames} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
