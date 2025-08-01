import { useState, useEffect, useRef } from 'react';

export default function useSpeech() {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [volume, setVolume] = useState(0);
  const [error, setError] = useState('');
  const recognitionRef = useRef(null);
  const audioAnalyserRef = useRef(null);
  const audioStreamRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError('Trình duyệt không hỗ trợ Speech Recognition');
      return;
    }
    const recog = new SpeechRecognition();
    recog.continuous = true;
    recog.interimResults = true;
    recog.lang = 'vi-VN';
    recog.onresult = (e) => {
      const text = Array.from(e.results).map(r => r[0].transcript).join('');
      setTranscript(text);
      setIsSpeaking(e.results.some(r => !r.isFinal));
    };
    recog.onerror = (e) => setError(e.error);
    recog.onend = () => listening && recog.start();

    recognitionRef.current = recog;
  }, []);

  useEffect(() => {
    if (!listening) {
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach(t => t.stop());
        audioAnalyserRef.current?.audioContext?.close();
        audioStreamRef.current = null;
      }
      setVolume(0);
      return;
    }

    let audioContext, analyser, dataArray, frameId;
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      audioStreamRef.current = stream;
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      dataArray = new Uint8Array(analyser.frequencyBinCount);
      audioAnalyserRef.current = { audioContext, analyser };

      const tick = () => {
        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;
        dataArray.forEach(v => {
          const x = v - 128;
          sum += x * x;
        });
        const rms = Math.sqrt(sum / dataArray.length) / 128;
        setVolume(rms);
        frameId = requestAnimationFrame(tick);
      };
      tick();
    }).catch(err => setError(err.message));

    return () => {
      cancelAnimationFrame(frameId);
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach(t => t.stop());
        audioAnalyserRef.current?.audioContext?.close();
      }
    };
  }, [listening]);

  const startListening = () => {
    setError('');
    setListening(true);
    recognitionRef.current?.start();
  };

  const stopListening = () => {
    setListening(false);
    recognitionRef.current?.stop();
  };

  return { listening, isSpeaking, transcript, volume, startListening, stopListening, error };
}
