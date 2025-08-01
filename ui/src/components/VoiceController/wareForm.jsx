export default function Waveform({ active, intense }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center z-0">
      <div className="flex items-end space-x-1 h-16">
        {[...Array(6)].map((_, i) => {
          const anim = active
            ? intense ? 'animate-wave-fast' : 'animate-wave-slow'
            : '';

          const delayStyle = active
            ? { animationDelay: `${i * (intense ? 80 : 200)}ms` }
            : {};

          return (
            <span
              key={i}
              className={`block w-2 bg-gradient-to-t from-cyan-400 to-purple-500 rounded-full ${active ? 'h-full' : ''
                } ${anim}`}
              style={delayStyle}
            />
          );
        })}
      </div>
    </div>
  );
}
