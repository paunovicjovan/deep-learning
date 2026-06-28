import React, { useCallback, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { createAudioPlayer } from 'expo-audio';
import type { AudioPlayer } from 'expo-audio';
import { CameraPane } from '../components/CameraPane';
import { VideoPane } from '../components/VideoPane';
import { HistoryList } from '../components/HistoryList';
import { useDetectionSocket } from '../lib/useDetectionSocket';
import type { ConnStatus, Detection } from '../lib/types';

type Mode = 'live' | 'demo';

const SOUND_MAP: Record<string, any> = {
  i2: require('../assets/sounds/i2.mp3'),
  i4: require('../assets/sounds/i4.mp3'),
  i5: require('../assets/sounds/i5.mp3'),
  il100: require('../assets/sounds/il100.mp3'),
  il110: require('../assets/sounds/il110.mp3'),
  il50: require('../assets/sounds/il50.mp3'),
  il60: require('../assets/sounds/il60.mp3'),
  il70: require('../assets/sounds/il70.mp3'),
  il80: require('../assets/sounds/il80.mp3'),
  il90: require('../assets/sounds/il90.mp3'),
  ip: require('../assets/sounds/ip.mp3'),
  p1: require('../assets/sounds/p1.mp3'),
  p10: require('../assets/sounds/p10.mp3'),
  p11: require('../assets/sounds/p11.mp3'),
  p12: require('../assets/sounds/p12.mp3'),
  p13: require('../assets/sounds/p13.mp3'),
  p19: require('../assets/sounds/p19.mp3'),
  p23: require('../assets/sounds/p23.mp3'),
  p26: require('../assets/sounds/p26.mp3'),
  p27: require('../assets/sounds/p27.mp3'),
  p3: require('../assets/sounds/p3.mp3'),
  p5: require('../assets/sounds/p5.mp3'),
  p6: require('../assets/sounds/p6.mp3'),
  pa: require('../assets/sounds/pa.mp3'),
  pb: require('../assets/sounds/pb.mp3'),
  pbp: require('../assets/sounds/pbp.mp3'),
  pg: require('../assets/sounds/pg.mp3'),
  ph: require('../assets/sounds/ph.mp3'),
  pl10: require('../assets/sounds/pl10.mp3'),
  pl100: require('../assets/sounds/pl100.mp3'),
  pl110: require('../assets/sounds/pl110.mp3'),
  pl120: require('../assets/sounds/pl120.mp3'),
  pl15: require('../assets/sounds/pl15.mp3'),
  pl20: require('../assets/sounds/pl20.mp3'),
  pl25: require('../assets/sounds/pl25.mp3'),
  pl30: require('../assets/sounds/pl30.mp3'),
  pl35: require('../assets/sounds/pl35.mp3'),
  pl40: require('../assets/sounds/pl40.mp3'),
  pl5: require('../assets/sounds/pl5.mp3'),
  pl50: require('../assets/sounds/pl50.mp3'),
  pl60: require('../assets/sounds/pl60.mp3'),
  pl65: require('../assets/sounds/pl65.mp3'),
  pl70: require('../assets/sounds/pl70.mp3'),
  pl80: require('../assets/sounds/pl80.mp3'),
  pl90: require('../assets/sounds/pl90.mp3'),
  pm: require('../assets/sounds/pm.mp3'),
  pn: require('../assets/sounds/pn.mp3'),
  pne: require('../assets/sounds/pne.mp3'),
  pr: require('../assets/sounds/pr.mp3'),
  w: require('../assets/sounds/w.mp3'),
};

// Configured in .env (EXPO_PUBLIC_BACKEND_URL). Restart Metro after changing it.
const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL ?? '';

const STATUS_COLOR: Record<ConnStatus, string> = {
  idle: '#888',
  connecting: '#e0a800',
  ready: '#2ecc71',
  error: '#e74c3c',
  closed: '#888',
};

const STATUS_LABEL: Record<ConnStatus, string> = {
  idle: 'Not connected',
  connecting: 'Connecting…',
  ready: 'Streaming',
  error: 'Connection error',
  closed: 'Disconnected',
};

export default function DetectorScreen() {
  const playerRef = useRef<AudioPlayer | null>(null);
  const queueRef = useRef<any[]>([]);
  const isPlayingRef = useRef(false);
  const onResultRef = useRef<() => void>(() => { });
  const [mode, setMode] = useState<Mode>('live');

  // Play queued sounds one at a time, waiting for each to finish so they don't overlap.
  const playNext = useCallback(() => {
    if (isPlayingRef.current) return;
    const source = queueRef.current.shift();
    if (!source) return;

    isPlayingRef.current = true;
    try {
      const p = createAudioPlayer(source);
      playerRef.current = p;

      const sub = p.addListener('playbackStatusUpdate', (status) => {
        if (!status.didJustFinish) return;
        sub.remove();
        try { p.remove(); } catch { }
        if (playerRef.current === p) playerRef.current = null;
        isPlayingRef.current = false;
        playNext();
      });

      p.play();
    } catch {
      // If a sound fails to start, don't wedge the queue.
      isPlayingRef.current = false;
      playNext();
    }
  }, []);

  const playSignSound = useCallback((d: Detection) => {
    const source = SOUND_MAP[d.class_name];
    if (!source) return;
    queueRef.current.push(source);
    playNext();
  }, [playNext]);

  const socket = useDetectionSocket({
    onNewSign: playSignSound,
    onResult: () => onResultRef.current(),
  });

  const connected = socket.status === 'ready' || socket.status === 'connecting';

  // Reconnecting re-runs the handshake, which resets the backend tracker.
  const resetSession = useCallback(() => {
    if (BACKEND_URL) socket.connect(BACKEND_URL);
  }, [socket]);

  return (
    <View style={styles.root}>
      {/* LEFT column: video on top, controls (buttons) in a strip below it */}
      <View style={styles.left}>
        <View style={styles.video}>
          {mode === 'live' ? (
            <CameraPane
              status={socket.status}
              sendFrame={socket.sendFrame}
              registerOnResult={(cb) => {
                onResultRef.current = cb;
              }}
            />
          ) : (  
            <VideoPane
              status={socket.status}
              sendFrame={socket.sendFrame}
              registerOnResult={(cb) => {
                onResultRef.current = cb;
              }}
              onReset={resetSession}
            />
         )} 
        </View>

        {/* CONTROLS: mode toggle + connection status/button, directly under the video */}
        <View style={styles.controls}>
          <View style={styles.modeRow}>
            {(['live', 'demo'] as Mode[]).map((m) => (
              <Pressable
                key={m}
                style={[styles.modeBtn, mode === m && styles.modeBtnActive]}
                onPress={() => setMode(m)}
              >
                <Text style={[styles.modeText, mode === m && styles.modeTextActive]}>
                  {m === 'live' ? 'Live' : 'Demo video'}
                </Text>
              </Pressable>
            ))}
          </View>

          <View style={styles.statusRow}>
            <View style={styles.header}>
              <View style={[styles.dot, { backgroundColor: STATUS_COLOR[socket.status] }]} />
              <Text style={styles.headerText}>{STATUS_LABEL[socket.status]}</Text>
            </View>

            {connected ? (
              <Pressable style={[styles.btn, styles.btnStop]} onPress={socket.disconnect}>
                <Text style={styles.btnText}>Disconnect</Text>
              </Pressable>
            ) : (
              <Pressable
                style={[styles.btn, !BACKEND_URL && styles.btnDisabled]}
                disabled={!BACKEND_URL}
                onPress={() => socket.connect(BACKEND_URL)}
              >
                <Text style={styles.btnText}>Connect</Text>
              </Pressable>
            )}
          </View>

          {socket.status === 'ready' ? (
            <Text style={styles.url} numberOfLines={1}>
              {`sent: ${socket.framesSent}`}
            </Text>
          ) : null}
        </View>
      </View>

      {/* RIGHT column: detection history, big sign cards stacked full height */}
      <View style={styles.listWrap}>
        <HistoryList data={socket.history} onClear={socket.clearHistory} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, flexDirection: 'row', backgroundColor: '#000' },
  // LEFT column: video stacked above the controls strip.
  left: { flex: 3, flexDirection: 'column' },
  // Video fills the column above the buttons.
  video: { flex: 1, backgroundColor: '#000' },
  controls: { backgroundColor: '#0c0c0c', paddingHorizontal: 12, paddingVertical: 10, gap: 8 },
  modeRow: { flexDirection: 'row', gap: 6 },
  modeBtn: {
    flex: 1,
    paddingVertical: 6,
    borderRadius: 6,
    backgroundColor: '#1c1c1e',
    alignItems: 'center',
  },
  modeBtnActive: { backgroundColor: '#0a84ff' },
  modeText: { color: '#888', fontSize: 13, fontWeight: '600' },
  modeTextActive: { color: '#fff' },
  statusRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  header: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  headerText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  url: { color: '#666', fontSize: 11, fontFamily: 'monospace' },
  btn: {
    backgroundColor: '#0a84ff',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 22,
    alignItems: 'center',
  },
  btnStop: { backgroundColor: '#e74c3c' },
  btnDisabled: { backgroundColor: '#333' },
  btnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  // RIGHT column: the at-a-glance detected-signs feed, full height.
  listWrap: { flex: 2, backgroundColor: '#0c0c0c', paddingHorizontal: 12, paddingVertical: 12 },
});
