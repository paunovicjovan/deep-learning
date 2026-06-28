import { useCallback, useEffect, useRef, useState } from 'react';
import type { ConnStatus, Detection } from './types';


interface UseDetectionSocketArgs {
  /** Called once for each NEW sign the backend reports (already deduped). */
  onNewSign?: (d: Detection) => void;
  /** Called once per server reply, used to clear the in-flight gate. */
  onResult?: () => void;
}

export function useDetectionSocket({ onNewSign, onResult }: UseDetectionSocketArgs = {}) {
  const [status, setStatus] = useState<ConnStatus>('idle');
  const [history, setHistory] = useState<Detection[]>([]);
  const [framesSent, setFramesSent] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const readyRef = useRef(false);

  // Keep latest callbacks in refs so connect() doesn't need to be recreated.
  const onNewSignRef = useRef(onNewSign);
  const onResultRef = useRef(onResult);
  useEffect(() => { onNewSignRef.current = onNewSign; }, [onNewSign]);
  useEffect(() => { onResultRef.current = onResult; }, [onResult]);

  const disconnect = useCallback(() => {
    readyRef.current = false;
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      // Detach handlers before close() so onclose won't fire, then set status here.
      ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
      try { ws.close(); } catch {}
      setStatus('closed');
    }
  }, []);

  const connect = useCallback((host: string) => {
    disconnect();
    setStatus('connecting');
    const ws = new WebSocket(`ws://${host}/ws/detect`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      // Handshake: frames will arrive as base64-encoded JPEG text messages.
      ws.send(JSON.stringify({ format: 'jpeg' }));
    };

    ws.onmessage = (e) => {
      if (typeof e.data !== 'string') return; // we only expect JSON back
      let msg: any;
      try { msg = JSON.parse(e.data); } catch { return; }

      if (msg.type === 'ready') {
        readyRef.current = true;
        setFramesSent(0);
        setStatus('ready');
      } else if (msg.type === 'error') {
        console.warn('[socket] server error:', msg.message);
        setStatus('error');
      } else if (msg.type === 'result') {
        if (msg.warning) console.warn('[socket] server warning:', msg.warning);
        onResultRef.current?.();
        const dets: Detection[] = msg.detections ?? [];
        if (dets.length) {
          // Image updates patch an existing entry's crop; new signs are prepended.
          const updates = dets.filter((d) => d.is_update);
          const fresh = dets.filter((d) => !d.is_update);

          setHistory((prev) => {
            let next = prev;
            if (updates.length) {
              next = next.map((item) => {
                const u = updates.find(
                  (d) => d.track_id === item.track_id && d.timestamp === item.timestamp,
                );
                return u ? { ...item, image_b64: u.image_b64 } : item;
              });
            }
            // reverse so within a batch the latest fresh sign ends up first
            return fresh.length ? [...[...fresh].reverse(), ...next] : next;
          });

          fresh.forEach((d) => onNewSignRef.current?.(d));
        }
      }
    };

    ws.onerror = () => setStatus('error');
    ws.onclose = () => { readyRef.current = false; setStatus('closed'); };
  }, [disconnect]);

  const sendFrame = useCallback((data: string | ArrayBuffer | ArrayBufferView) => {
    const ws = wsRef.current;
    if (ws && readyRef.current && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(data as any);
        setFramesSent((n) => n + 1);
      } catch (e) {
        console.warn('[socket] send failed:', e);
      }
    }
  }, []);

  const clearHistory = useCallback(() => setHistory([]), []);

  // Clean up on unmount.
  useEffect(() => disconnect, [disconnect]);

  return { status, history, framesSent, connect, disconnect, sendFrame, clearHistory };
}
