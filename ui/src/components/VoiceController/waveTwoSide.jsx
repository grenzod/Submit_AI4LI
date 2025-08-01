import { useState, useEffect, useRef } from 'react';

export default function WaveTwoSide({ active = false, volume = 0.5 }) {
  const [animationPhase, setAnimationPhase] = useState(0);
  const frameIdRef = useRef(null);

  useEffect(() => {
    if (active) {
      const tick = () => {
        setAnimationPhase(prev => (prev + 2) % 360); 
        frameIdRef.current = requestAnimationFrame(tick);
      };
      frameIdRef.current = requestAnimationFrame(tick);
    } else {
      setAnimationPhase(0);
    }

    return () => {
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current);
      }
      frameIdRef.current = null;
      setAnimationPhase(0);
    };
  }, [active]);

  return (
    <div className="flex items-center justify-center space-x-8 h-full relative">
      <div className="flex items-center space-x-1 z-10">
        {[...Array(18)].map((_, i) => {
          const base = 15; // Chiều cao cơ bản
          const maxH = 100; // Chiều cao tối đa
          const phase = (animationPhase + i * 30) % 360; // Góc pha cho mỗi cột
          const waveOffset = Math.sin(phase * Math.PI / 180) * 0.8; // Dao động sin

          let h;
          if (active) {
            const volumeMultiplier = volume * 1.5 + waveOffset; // Tăng ảnh hưởng của volume
            const heightMultiplier = Math.sin((i + 1) * Math.PI / 16); // Hệ số phân bố
            h = base + volumeMultiplier * (maxH - base) * heightMultiplier;
          } else {
            h = base;
          }

          const delay = active ? i * 30 : 0; // Độ trễ khi active
          const intensity = (h - base) / (maxH - base); // Độ mạnh của hiệu ứng
          const hue = 180 + (i * 15); // Màu sắc thay đổi theo cột

          return (
            <div key={i} className="relative" style={{ animationDelay: `${delay}ms` }}>
              <span
                className="block w-3 rounded-full relative overflow-hidden shadow-lg"
                style = {{
                  height: `${Math.max(h, 8)}px`,
                  background: active
                    ? `linear-gradient(to top, hsl(${hue - 30}, 70%, 60%), hsl(${hue}, 80%, 65%), hsl(${hue + 30}, 90%, 70%))`
                    : 'linear-gradient(to top, #374151, #6b7280)', // Gradient khi không active
                  transition: 'height 0.1s ease-out, transform 0.1s ease-out', // Transition mượt mà
                  transform: active ? 'scaleY(1)' : 'scaleY(0.8)', // Hiệu ứng scale nhẹ
                  boxShadow: active
                    ? `0 0 ${intensity * 20}px hsla(${hue}, 80%, 65%, 0.6)`
                    : 'none', 
                }}>
                <div
                  className="absolute inset-0 rounded-full"
                  style={{
                    background: active
                      ? `linear-gradient(to top, hsla(${hue}, 100%, 80%, 0.3), hsla(${hue + 30}, 100%, 90%, 0.6))`
                      : 'none',
                    opacity: intensity,
                  }}
                />
              </span>
              {active && intensity > 0.7 && (
                <div
                  className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-2 h-2 rounded-full animate-pulse"
                  style={{
                    background: `hsl(${hue + 60}, 100%, 90%)`,
                    boxShadow: `0 0 8px hsl(${hue + 60}, 100%, 90%)`,
                  }}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}