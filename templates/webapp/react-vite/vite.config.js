import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// CRITICAL ARCHITECTURE RULE: 
// Do NOT modify this file to set `root: "src"`.
// Do NOT move index.html. Vite requires index.html to be at the project root.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
})
