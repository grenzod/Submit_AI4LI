import React from 'react'
import { MicrophoneIcon, StopIcon } from '@heroicons/react/24/solid'

export default function VoiceButton({ listening, onStart, onStop }) {
  return (
    <button
      onClick = {listening ? onStop : onStart}
      className = {`
        relative flex items-center justify-center
        w-32 h-32 rounded-full bg-red text-indigo-900 
        border-4 border-black
        shadow-lg transition-transform duration-200
        focus:outline-none focus:border-cyan-400
        ${listening ? 'scale-110' : 'hover:scale-105'}
      `}
    >
      {listening ? (
        <StopIcon className="w-16 h-16" />
      ) : (
        <MicrophoneIcon className="w-16 h-16" />
      )}
    </button>
  )
}

