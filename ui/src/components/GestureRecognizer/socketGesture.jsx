import { useState, useEffect, useRef, useCallback } from 'react';
import { Hands } from '@mediapipe/hands';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { HAND_CONNECTIONS } from '@mediapipe/hands';

export default function useGestureRecognition() {
  const [active, setActive] = useState(false);
  const [gesture, setGesture] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handsRef = useRef(null);

  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(0);

  const socketRef = useRef(null);
  const animationRef = useRef(null);

  // Khởi tạo MediaPipe Hands
  const initHands = useCallback(() => {
    const hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7
    });

    hands.onResults(handleResults);
    handsRef.current = hands;
  }, []);

  // Tách landmarks thành left_hand và right_hand arrays
  const extractLandmarks = useCallback((results) => {
    const defaultHand = Array(21).fill({ x: 0, y: 0, z: 0 });
    let leftHand = defaultHand;
    let rightHand = defaultHand;

    if (
      results.multiHandLandmarks &&
      results.multiHandedness &&
      results.multiHandLandmarks.length === results.multiHandedness.length
    ) {
      results.multiHandedness.forEach((h, i) => {
        if (h.label === 'Left') {
          leftHand = results.multiHandLandmarks[i];
        } else if (h.label === 'Right') {
          rightHand = results.multiHandLandmarks[i];
        }
      });
    }

    const leftHandData = leftHand.map((p) => ({
      x: p.x,
      y: p.y,
      z: p.z
    }));
    const rightHandData = rightHand.map((p) => ({
      x: p.x,
      y: p.y,
      z: p.z
    }));

    return { leftHandData, rightHandData };
  }, []);

  // Xử lý kết quả từ MediaPipe và gửi về backend
  const handleResults = useCallback(
    (results) => {
      // Cập nhật FPS
      frameCountRef.current++;
      const now = performance.now();
      if (now >= lastFpsUpdateRef.current + 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsUpdateRef.current = now;
      }

      // Vẽ lên canvas
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (canvas && video) {
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

        if (results.multiHandLandmarks) {
          for (const landmarks of results.multiHandLandmarks) {
            drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 1 });
            drawConnectors(
              ctx,
              landmarks,
              HAND_CONNECTIONS,
              { color: '#00FF00', lineWidth: 2 }
            );
          }
        }
        ctx.restore();
      }

      // Gửi dữ liệu sang backend
      if (
        socketRef.current &&
        socketRef.current.readyState === WebSocket.OPEN
      ) {
        const { leftHandData, rightHandData } = extractLandmarks(results);

        socketRef.current.send(
          JSON.stringify({
            left_hand: leftHandData,
            right_hand: rightHandData
          })
        );
      }
    },
    [extractLandmarks]
  );

  // Bật camera và bắt khung hình
  const startCamera = useCallback(async () => {
    try {
      const video = videoRef.current;
      if (!video) return;

      video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });

      await new Promise((r) => (video.onloadedmetadata = r));
      video.play();
      setActive(true);

      const processFrame = async () => {
        if (video.readyState >= 2 && handsRef.current) {
          await handsRef.current.send({ image: video });
        }
        animationRef.current = requestAnimationFrame(processFrame);
      };
      animationRef.current = requestAnimationFrame(processFrame);
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Could not access camera. Please check permissions.');
    }
  }, []);

  // Tắt camera
  const stopCamera = useCallback(() => {
    const video = videoRef.current;
    if (video && video.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    setActive(false);
  }, []);

  // Khởi tạo MediaPipe khi mount
  useEffect(() => {
    initHands();
    return () => {
      stopCamera();
    };
  }, [initHands, stopCamera]);

  // Mở/đóng WebSocket khi active thay đổi
  useEffect(() => {
    if (active) {
      const socket = new WebSocket('ws://localhost:8000/gesture');
      socket.onopen = () => console.log('WebSocket connected');
      socket.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.gesture) setGesture(data.gesture);
          if (data.confidence != null)
            setConfidence(Math.round(data.confidence * 100));
        } catch (err) {
          console.error('WS message parse error:', err);
        }
      };
      socket.onerror = (e) => {
        console.error('WebSocket error:', e);
        setError('WebSocket connection error');
      };
      socket.onclose = () => console.log('WebSocket disconnected');

      socketRef.current = socket;
    } else {
      socketRef.current?.close();
      socketRef.current = null;
    }

    return () => {
      socketRef.current?.close();
    };
  }, [active]);

  return {
    active,
    gesture,
    confidence,
    fps,
    error,
    videoRef,
    canvasRef,
    startCamera,
    stopCamera
  };
}