/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // TikTok brand colors
        tiktok: {
          red: '#fe2c55',
          cyan: '#25f4ee',
          black: '#010101',
        },
        // Custom dark theme
        dark: {
          900: '#0a0a0a',
          800: '#1a1a2e',
          700: '#16213e',
          600: '#1f2937',
          500: '#374151',
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'tiktok-gradient': 'linear-gradient(135deg, #fe2c55 0%, #25f4ee 100%)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
        'spin-slow': 'spin 3s linear infinite',
        'gradient': 'gradient 8s ease infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        gradient: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      boxShadow: {
        'glow': '0 0 40px rgba(254, 44, 85, 0.3)',
        'glow-cyan': '0 0 40px rgba(37, 244, 238, 0.3)',
        'inner-glow': 'inset 0 0 20px rgba(254, 44, 85, 0.1)',
      },
    },
  },
  plugins: [],
}
