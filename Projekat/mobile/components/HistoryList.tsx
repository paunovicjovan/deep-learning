import React from 'react';
import { FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import type { Detection } from '../lib/types';

function timeLabel(ts: number) {
  try {
    return new Date(ts * 1000).toLocaleTimeString();
  } catch {
    return '';
  }
}

export function HistoryList({ data, onClear }: { data: Detection[]; onClear: () => void }) {
  return (
    <View style={styles.fill}>
      <FlatList
        data={data}
        // track_id isn't unique across sessions (IDs restart at 1), so key on timestamp + id.
        keyExtractor={(item) => `${item.timestamp}-${item.track_id}`}
        contentContainerStyle={data.length === 0 ? styles.emptyWrap : styles.listContent}
        ListEmptyComponent={<Text style={styles.empty}>No signs detected yet…</Text>}
        renderItem={({ item }) => (
          <View style={styles.item}>
            {item.image_b64 ? (
              <Image
                style={styles.thumb}
                source={{ uri: `data:image/jpeg;base64,${item.image_b64}` }}
                resizeMode="cover"
              />
            ) : (
              <View style={[styles.thumb, styles.thumbPlaceholder]}>
                <Text style={styles.thumbPlaceholderText}>?</Text>
              </View>
            )}
            <View style={styles.itemBody}>
              <Text style={styles.desc} numberOfLines={2}>
                {item.description}
              </Text>
              <View style={styles.metaRow}>
                <Text style={styles.cls}>{item.class_name}</Text>
                <Text style={styles.conf}>{Math.round(item.confidence * 100)}%</Text>
                <Text style={styles.time}>{timeLabel(item.timestamp)}</Text>
              </View>
            </View>
          </View>
        )}
      />
      {data.length > 0 && (
        <Pressable style={styles.clear} onPress={onClear}>
          <Text style={styles.clearText}>Clear ({data.length})</Text>
        </Pressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  listContent: { paddingBottom: 8 },
  emptyWrap: { flexGrow: 1, justifyContent: 'center', alignItems: 'center' },
  empty: { color: '#888', fontSize: 14 },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1c1c1e',
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 14,
    marginBottom: 10,
    gap: 16,
  },
  thumb: {
    width: 104,
    height: 104,
    borderRadius: 12,
    backgroundColor: '#000',
  },
  thumbPlaceholder: { alignItems: 'center', justifyContent: 'center' },
  thumbPlaceholderText: { color: '#555', fontSize: 44, fontWeight: '700' },
  itemBody: { flex: 1 },
  desc: { color: '#fff', fontSize: 22, fontWeight: '700' },
  metaRow: { flexDirection: 'row', alignItems: 'center', marginTop: 8, gap: 12 },
  cls: { color: '#5ac8fa', fontSize: 15, fontFamily: 'monospace' },
  conf: { color: '#2ecc71', fontSize: 15, fontWeight: '600' },
  time: { color: '#888', fontSize: 14, marginLeft: 'auto' },
  clear: {
    paddingVertical: 8,
    alignItems: 'center',
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#333',
  },
  clearText: { color: '#e74c3c', fontSize: 13 },
});
