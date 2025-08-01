import React, { useState, useRef } from 'react';
import axios from 'axios';

export default function useSpeechRequest() {
    const [text, setText] = useState('');
    const [audioUrl, setAudioUrl] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [playbackRate, setPlaybackRate] = useState(1.0);
    const audioRef = useRef(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!text.trim()) {
            setError('Vui lòng nhập văn bản');
            return;
        }

        setIsLoading(true);
        setError('');

        try {
            const response = await axios.post(
                'http://localhost:8000/generate-speech/',
                {
                    text,
                    speed: playbackRate
                },
                {
                    responseType: 'blob', 
                }
            );

            const url = URL.createObjectURL(response.data);
            setAudioUrl(url);
            
            if (audioRef.current) {
                audioRef.current.playbackRate = playbackRate;
                // Thêm setTimeout để đảm bảo URL đã được cập nhật
                setTimeout(() => audioRef.current.play(), 100);
            }
        } catch (err) {
            console.error('Error generating speech:', err);
            setError('Không thể tạo giọng nói. Vui lòng thử lại.');
        } finally {
            setIsLoading(false);
        }
    };

    const handlePlaybackRateChange = (e) => {
        const rate = parseFloat(e.target.value);
        setPlaybackRate(rate);
        
        if (audioRef.current) {
            audioRef.current.playbackRate = rate;
        }
    };

    return {
        text,
        setText,
        audioUrl,
        isLoading,
        error,
        playbackRate,
        audioRef,
        handleSubmit,
        handlePlaybackRateChange 
    };
}