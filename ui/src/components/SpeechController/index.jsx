import React from "react";
import useSpeechRequest from "./request"

export default function SpeechController() {
    const {
        text,
        setText,
        audioUrl,
        isLoading,
        error,
        playbackRate,
        audioRef,
        handleSubmit,
        handlePlaybackRateChange 
    } = useSpeechRequest();

    return (
        <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
            <h1 className="text-2xl font-bold mb-6 text-center">Chuyển văn bản thành giọng nói</h1>

            <form onSubmit={handleSubmit} className="mb-6">
                <div className="mb-4">
                    <label htmlFor="text" className="block text-gray-700 font-bold mb-2">
                        Nhập văn bản:
                    </label>
                    <textarea
                        id="text"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        rows={4}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Nhập văn bản bạn muốn chuyển thành giọng nói..."
                    />
                </div>

                {error && <p className="text-red-500 mb-4">{error}</p>}

                <button
                    type="submit"
                    disabled={isLoading}
                    className={`w-full bg-blue-500 text-white font-bold py-2 px-4 rounded hover:bg-blue-700 transition duration-200 
                        ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                    {isLoading ? 'Đang tạo giọng nói...' : 'Tạo giọng nói'}
                </button>

                <div className="mb-4">
                    <label htmlFor="playbackRate" className="block text-gray-700 font-bold mb-2">
                        Tốc độ nói: {playbackRate.toFixed(1)}x
                    </label>
                    <input
                        type="range"
                        id="playbackRate"
                        min="0.5"
                        max="2.0"
                        step="0.1"
                        value={playbackRate}
                        onChange={handlePlaybackRateChange} 
                        className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500 mt-1">
                        <span>Chậm</span>
                        <span>Bình thường</span>
                        <span>Nhanh</span>
                    </div>
                </div>                
            </form>

            {audioUrl && (
                <div className="mt-6">
                    <h2 className="text-xl font-semibold mb-3">Kết quả:</h2>
                    <audio
                        ref={audioRef}
                        src={audioUrl}
                        controls
                        className="w-full"
                    />
                    <div className="mt-4">
                        <a
                            href={audioUrl}
                            download="speech.mp3"
                            className="inline-flex items-center px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                            </svg>
                            Tải xuống
                        </a>
                    </div>
                </div>
            )}
        </div>
    );
}