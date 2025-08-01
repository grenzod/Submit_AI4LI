/** @type {import('tailwindcss').Config} */
// tailwind.config.js
export default {
  content: ["./index.html","./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      keyframes: {
        wave: {
          "0%, 100%": { 
            transform: "scaleY(0.4)", 
            opacity: "0.6" 
          },
          "50%": { 
            transform: "scaleY(1)", 
            opacity: "1" 
          },
        },
      },
      animation: {
        "wave-slow": "wave 4s ease-in-out infinite",
        "wave-fast": "wave 1s ease-in-out infinite",
      },
    },
  },
  plugins: [],
}
