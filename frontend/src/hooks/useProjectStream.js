import { useEffect, useRef } from 'react';
import { WS_BASE_URL } from '../services/api.js';

export function useProjectStream({ threadId, enabled, onMessage, onOpen, onClose }) {
  const handlersRef = useRef({ onMessage, onOpen, onClose });

  useEffect(() => {
    handlersRef.current = { onMessage, onOpen, onClose };
  }, [onMessage, onOpen, onClose]);

  useEffect(() => {
    if (!enabled || !threadId) return undefined;

    const socket = new WebSocket(`${WS_BASE_URL}/ws/stream/${encodeURIComponent(threadId)}`);

    socket.onopen = () => handlersRef.current.onOpen?.();
    socket.onmessage = (event) => {
      try {
        handlersRef.current.onMessage?.(JSON.parse(event.data));
      } catch (error) {
        handlersRef.current.onMessage?.({ type: 'error', data: error.message });
      }
    };
    socket.onclose = () => handlersRef.current.onClose?.();

    return () => socket.close();
  }, [threadId, enabled]);
}
