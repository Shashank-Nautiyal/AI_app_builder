import { Monitor } from 'lucide-react'

function App() {
  return (
    <main className="app-shell">
      <section className="workspace">
        <Monitor aria-hidden="true" />
        <div>
          <p>Desktop workspace</p>
          <h1>Electron React App</h1>
          <span>Ready for implementation</span>
        </div>
      </section>
    </main>
  )
}

export default App
