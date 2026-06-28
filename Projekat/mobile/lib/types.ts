export interface Detection {
  track_id: number;
  class_name: string;
  description: string;
  confidence: number;
  bbox: number[]; // normalized [x1, y1, x2, y2]
  frame_number: number;
  timestamp: number; // seconds since epoch (from the backend)
  image_b64?: string | null; // base64 JPEG of the sign cropped from the frame
  // True if this is a clearer crop for a sign already in history, not a new one.
  is_update?: boolean;
}

export type ConnStatus = 'idle' | 'connecting' | 'ready' | 'error' | 'closed';
