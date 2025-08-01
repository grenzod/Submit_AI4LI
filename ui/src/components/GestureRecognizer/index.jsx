import React from 'react';
import useGestureRecognition from './socketGesture';

export default function GestureRecognizer() {
    const {
        active,
        gesture,
        confidence,
        fps,
        error,
        videoRef,
        canvasRef,
        startCamera,
        stopCamera
    } = useGestureRecognition();

    return (
        <div className="min-h-screen bg-gray-900 text-white p-8">
            <h1 className="text-3xl font-bold mb-6">Hand Gesture Recognition</h1>
            
            <div className="flex flex-col lg:flex-row gap-6">
                <div className="flex-1 bg-gray-800 rounded-xl overflow-hidden shadow-xl">
                    <div className="relative">
                        <video 
                            ref={videoRef} 
                            autoPlay 
                            playsInline 
                            className="w-full h-auto"
                        />
                        <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none"/>
                        <div className="absolute top-4 left-4 bg-black/70 px-3 py-1 rounded-lg">
                            FPS: {fps}
                        </div>
                    </div>
                    <div className="p-4">
                        <button
                            onClick={active ? stopCamera : startCamera}
                            className={`px-4 py-2 rounded-lg font-bold 
                                ${active ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'} transition-colors`
                            }
                        >
                            {active ? 'Stop' : 'Start'} Recognition
                        </button>
                    </div>
                </div>

                <div className="lg:w-80 bg-gray-800 rounded-xl p-6 shadow-xl">
                    <h2 className="text-2xl font-bold mb-4">Detection Result</h2>
                    <div className="mb-6">
                        <div className="text-3xl font-bold mb-2">{gesture != "Null" ? gesture : 'No gesture'}</div>
                        <div className="text-xl text-gray-400">Confidence: {gesture != "Null" ? `${confidence} %` : ' '}</div>
                    </div>
                    
                    {error && (
                        <div className="bg-red-500/20 border border-red-500 rounded-lg p-4 text-red-300">
                            {error}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}