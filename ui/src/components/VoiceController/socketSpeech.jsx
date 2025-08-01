import { useState, useEffect, useRef } from 'react';

export default function socketSpeech() {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [conversation, setConversation] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [volume, setVolume] = useState(0);
  const [error, setError] = useState('');

  const audioStreamRef = useRef(null);
  const audioAnalyserRef = useRef(null);
  const audioProcessorRef = useRef(null);
  const socketRef = useRef(null);
  const frameIdRef = useRef(0);

  const audioBufferRef = useRef(new Int16Array(0));
  const sampleRate = 16000;
  const bytesPerSecond = sampleRate * 4; // 2 bytes/sample (16-bit)

  useEffect(() => {
    if (listening) {
      const socket = new WebSocket("ws://localhost:8000/ws");
      socket.binaryType = "arraybuffer";

      socket.onopen = () => console.log('WebSocket was connected');
      socket.onclose = () => console.log('WebSocket was disconnected');
      socket.onerror = e => {
        console.error('WebSocket error:', e);
        setError("WebSocket connection error");
      };
      socket.onmessage = e => {
        try {
          const data = JSON.parse(e.data);
          console.log('Received:', data);

          if (data.is_final) {
            setConversation(prev => prev + data.transcript + ' ');
            setTranscript('');
            setIsSpeaking(false);
          }
        } catch (err) {
          console.error('Error parsing message:', err);
        }
      };

      socketRef.current = socket;
      audioBufferRef.current = new Int16Array(0);
    }
    else {
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    }

    return () => {
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [listening]);

  useEffect(() => {
    if (!listening) {
      audioStreamRef.current?.getTracks().forEach(t => t.stop());
      audioProcessorRef.current?.disconnect();
      audioAnalyserRef.current?.audioContext?.close();
      cancelAnimationFrame(frameIdRef.current);
      setVolume(0);
      return;
    }

    navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: sampleRate,
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: true,
        autoGainControl: true
      }
    }).then(stream => {
      audioStreamRef.current = stream;

      const audioContext = new (window.AudioContext || window.AudioContext)({
        sampleRate: sampleRate
      });

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      const bufferSize = 2048;
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

      processor.onaudioprocess = event => {
        const inputData = event.inputBuffer.getChannelData(0);
        const rms = Math.sqrt(inputData.reduce((sum, val) => sum + val * val, 0) / inputData.length);

        if (rms > 0.02) {
          // Float32 → Int16
          const int16Buffer = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            let s = Math.max(-1, Math.min(1, inputData[i]));
            // int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            int16Buffer[i] = Math.max(-32768, Math.min(32767, Math.round(s * 32767)));
          }

          // Thêm vào buffer tích lũy
          const newBuffer = new Int16Array(audioBufferRef.current.length + int16Buffer.length);
          newBuffer.set(audioBufferRef.current);
          newBuffer.set(int16Buffer, audioBufferRef.current.length);
          audioBufferRef.current = newBuffer;

          // Kiểm tra nếu đủ 2 giây thì gửi
          if (audioBufferRef.current.length >= bytesPerSecond) {
            if (socketRef.current?.readyState === WebSocket.OPEN) {
              console.log(`Gửi 2 giây âm thanh: ${audioBufferRef.current.length * 2} bytes`);
              socketRef.current.send(audioBufferRef.current.buffer);

              audioBufferRef.current = new Int16Array(0);
            }
          }
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      audioProcessorRef.current = processor;
      audioAnalyserRef.current = { audioContext, analyser };

      // Vẽ volume
      const floatData = new Float32Array(analyser.fftSize);
      const tick = () => {
        analyser.getFloatTimeDomainData(floatData);

        let sum = 0;
        for (let v of floatData) sum += v * v;
        const rms = Math.sqrt(sum / floatData.length);

        const scaled = Math.min(1, rms * 30);
        setVolume(scaled);

        frameIdRef.current = requestAnimationFrame(tick);
      };
      frameIdRef.current = requestAnimationFrame(tick);

      setIsSpeaking(true);
      console.log('Audio processing started');
    })
      .catch(err => {
        console.error('Error accessing microphone:', err);
        setError(`Microphone access denied: ${err.message}`);
      });

    return () => {
      audioStreamRef.current?.getTracks().forEach(t => t.stop());
      audioProcessorRef.current?.disconnect();
      audioAnalyserRef.current?.audioContext?.close();
      cancelAnimationFrame(frameIdRef.current);
    };
  }, [listening]);

  const startListening = () => {
    setError('');
    setTranscript('');
    setConversation('');
    setListening(true);
    console.log('Starting to listen...');
  };

  const stopListening = () => {
    setListening(false);
    setIsSpeaking(false);

    // Gửi phần âm thanh còn lại khi dừng
    if (audioBufferRef.current.length > 0 && socketRef.current?.readyState === WebSocket.OPEN) {
      console.log(`Gửi âm thanh cuối: ${audioBufferRef.current.length * 2} bytes`);
      socketRef.current.send(audioBufferRef.current.buffer);
      audioBufferRef.current = new Int16Array(0);
    }

    console.log('Stopped listening');
  };

  return {
    listening,
    isSpeaking,
    conversation,
    volume,
    error,
    startListening,
    stopListening
  };
}
