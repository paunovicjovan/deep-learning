import React, { useEffect, useRef } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import * as FileSystem from 'expo-file-system/legacy';
import type { ConnStatus } from '../lib/types';

// JPEG quality for snapshots (backend downscales to 640 wide anyway).
const SNAPSHOT_QUALITY = 80;
// Target capture cadence in ms. Lower = more frames, higher = smoother preview.
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
}

export function CameraPane({ status, sendFrame, registerOnResult }: Props) {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const camRef = useRef<Camera>(null);
  // Send timestamps of frames still awaiting a backend reply.
  const pendingRef = useRef<number[]>([]);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    registerOnResult(() => {
      pendingRef.current.shift();
    });
  }, [registerOnResult]);

  // Streaming loop. Captures frames on a fixed cadence; up to MAX_IN_FLIGHT may
  // await a backend reply so capture overlaps with backend processing.
  useEffect(() => {
    if (status !== 'ready') return;
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

          const cam = camRef.current;
          if (cam && pendingRef.current.length < MAX_IN_FLIGHT) {
            const snap = await cam.takeSnapshot({ quality: SNAPSHOT_QUALITY });
            if (!active) break;

            const uri = snap.path.startsWith('file://') ? snap.path : `file://${snap.path}`;
            const b64 = await FileSystem.readAsStringAsync(uri, {
              encoding: FileSystem.EncodingType.Base64,
            });
            // Snapshots are written to cache; clean up so they don't accumulate.
            FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => {});
            if (!active) break;

            pendingRef.current.push(Date.now());
            sendFrame(b64);
          }
        } catch (e) {
          // e.g. snapshot attempted before the camera session is fully started
          console.warn('[camera] stream loop error:', e);
        }

        const elapsed = Date.now() - tickStart;
        await sleep(Math.max(10, TARGET_INTERVAL_MS - elapsed));
      }
    })();

    return () => {
      active = false;
      pendingRef.current = [];
    };
  }, [status, sendFrame]);

  if (!hasPermission) {
    return (
      <View style={styles.fill}>
        <Text style={styles.msg}>Camera permission required</Text>
      </View>
    );
  }
  if (device == null) {
    return (
      <View style={styles.fill}>
        <Text style={styles.msg}>No camera device found</Text>
      </View>
    );
  }

  return (
    <View style={styles.fill}>
      <Camera
        ref={camRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        resizeMode="cover"
        // texture-view is required for takeSnapshot and renders the preview correctly.
        androidPreviewViewType="texture-view"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#000', justifyContent: 'center', alignItems: 'center' },
  msg: { color: '#fff', padding: 16, textAlign: 'center' },
});
