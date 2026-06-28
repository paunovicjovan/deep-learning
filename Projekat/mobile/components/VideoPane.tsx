import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';
import { useVideoPlayer, VideoView } from 'expo-video';
import * as ImagePicker from 'expo-image-picker';
import * as VideoThumbnails from 'expo-video-thumbnails';
import * as FileSystem from 'expo-file-system/legacy';
import type { ConnStatus } from '../lib/types';

// Demo mode: pick a video, play it, and sample frames from it for the backend.
// Like CameraPane but the frame comes from a thumbnail at the current playback
// time. Playback and sampling are decoupled so the video stays smooth.

// JPEG quality for sampled frames (backend downscales to 640 wide anyway).
const THUMB_QUALITY = 0.6;
// Sampling cadence in ms.
const TARGET_INTERVAL_MS = 50;
// How many frames may await a backend reply at once (pipelining).
const MAX_IN_FLIGHT = 3;
// A frame with no reply after this long counts as lost (prevents deadlock).
const REPLY_TIMEOUT_MS = 3000;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

interface Props {
  status: ConnStatus;
  sendFrame: (data: string | ArrayBuffer) => void;
  /** Lets the parent route the socket's per-reply event to our pacing gate. */
  registerOnResult: (cb: () => void) => void;
  /** Re-handshakes the socket so the backend tracker starts fresh per video. */
  onReset?: () => void;
}

export function VideoPane({ status, sendFrame, registerOnResult, onReset }: Props) {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [picking, setPicking] = useState(false);
  // Send timestamps of frames still awaiting a backend reply.
  const pendingRef = useRef<number[]>([]);
  // Latest video URI for the sampling loop without re-creating it on each pick.
  const videoUriRef = useRef<string | null>(null);
  useEffect(() => { videoUriRef.current = videoUri; }, [videoUri]);

  const player = useVideoPlayer(null, (p) => {
    p.loop = true;
    p.muted = true;
  });

  useEffect(() => {
    registerOnResult(() => {
      pendingRef.current.shift();
    });
  }, [registerOnResult]);

  const pickVideo = useCallback(async () => {
    try {
      setPicking(true);
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) return;
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['videos'],
      });
      if (result.canceled) return;
      const uri = result.assets[0].uri;
      setVideoUri(uri);
      // replaceAsync swaps the source on the existing player, then play.
      await player.replaceAsync({ uri });
      player.play();
      // Re-handshake so the backend tracker starts clean for this video.
      onReset?.();
    } catch (e) {
      console.warn('[video] pick failed:', e);
    } finally {
      setPicking(false);
    }
  }, [player, onReset]);

  // Sampling loop. Same pacing as CameraPane but the frame comes from a video
  // thumbnail at the player's current time.
  useEffect(() => {
    if (status !== 'ready' || !videoUri) return;
    let active = true;
    pendingRef.current = [];

    (async () => {
      while (active) {
        const tickStart = Date.now();
        try {
          // Treat long-unanswered frames as lost so the pipeline never stalls.
          while (
            pendingRef.current.length &&
            tickStart - pendingRef.current[0] > REPLY_TIMEOUT_MS
          ) {
            pendingRef.current.shift();
          }

          const src = videoUriRef.current;
          if (src && pendingRef.current.length < MAX_IN_FLIGHT) {
            // player.currentTime is in seconds; getThumbnailAsync wants ms.
            const timeMs = Math.max(0, Math.round((player.currentTime ?? 0) * 1000));
            const { uri } = await VideoThumbnails.getThumbnailAsync(src, {
              time: timeMs,
              quality: THUMB_QUALITY,
            });
            if (!active) break;

            const b64 = await FileSystem.readAsStringAsync(uri, {
              encoding: FileSystem.EncodingType.Base64,
            });
            // Thumbnails are written to cache; clean up so they don't accumulate.
            FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => {});
            if (!active) break;

            pendingRef.current.push(Date.now());
            sendFrame(b64);
          }
        } catch (e) {
          console.warn('[video] sample loop error:', e);
        }

        const elapsed = Date.now() - tickStart;
        await sleep(Math.max(10, TARGET_INTERVAL_MS - elapsed));
      }
    })();

    return () => {
      active = false;
      pendingRef.current = [];
    };
  }, [status, videoUri, sendFrame, player]);

  return (
    <View style={styles.fill}>
      {videoUri ? (
        <VideoView
          player={player}
          style={StyleSheet.absoluteFill}
          contentFit="contain"
          nativeControls={false}
        />
      ) : (
        <Text style={styles.msg}>No video loaded</Text>
      )}

      <View style={styles.overlay}>
        <Pressable style={styles.pickBtn} onPress={pickVideo} disabled={picking}>
          {picking ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.pickText}>{videoUri ? 'Change video' : 'Pick video'}</Text>
          )}
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#000', justifyContent: 'center', alignItems: 'center' },
  msg: { color: '#fff', padding: 16, textAlign: 'center' },
  overlay: { position: 'absolute', bottom: 16, alignSelf: 'center' },
  pickBtn: {
    backgroundColor: 'rgba(10,132,255,0.9)',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 20,
    minWidth: 140,
    alignItems: 'center',
  },
  pickText: { color: '#fff', fontWeight: '700', fontSize: 14 },
});
