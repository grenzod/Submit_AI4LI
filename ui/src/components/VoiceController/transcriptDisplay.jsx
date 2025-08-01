import React, { useEffect, useState } from 'react';

export default function TranscriptDisplay({ transcript, error, listening }) {
  const [gradientPos, setGradientPos] = useState(0);

  useEffect(() => {
    let animationFrame;

    if (listening) {
      const animate = () => {
        setGradientPos(prev => (prev + 0.5) % 100);
        animationFrame = requestAnimationFrame(animate);
      };
      animationFrame = requestAnimationFrame(animate);
    }

    return () => {
      if (animationFrame) cancelAnimationFrame(animationFrame);
    };
  }, [listening]);

  const baseClasses = `
    w-full bg-white bg-opacity-80 p-4 rounded-lg shadow-md max-h-40 overflow-auto
    ${!listening ? 'border-2 border-black' : 'border-0'}
  `;

  const gradientStyle = listening
    ? {
      backgroundImage: `
          linear-gradient(
            to right,
            #ffffff 0%, #ffffff 10%, #ffffff 90%, #ffffff 100%
          ),
          linear-gradient(
            to right,
            #00dbde ${gradientPos}%,
            #fc00ff ${gradientPos + 10}%,
            #00dbde ${gradientPos + 20}%
          )
        `,
      backgroundClip: 'padding-box, border-box',
      backgroundOrigin: 'border-box',
      border: '2px solid transparent',
    }
    : {};

  return (
    <div className={baseClasses} style={gradientStyle}>
      {error && <div className="text-red-600 font-medium">{error}</div>}
      {transcript
        ?
        (
          <div className="max-h-32">
            <p className="text-indigo-900 whitespace-pre-wrap">{transcript}</p>
          </div>
        )
        : <p className="text-indigo-400 italic">Nhấn vào mic để bắt đầu nói...</p>
      }
    </div>
  );
}
