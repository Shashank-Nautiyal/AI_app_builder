import { AlertTriangle, GitBranch, Radio, RotateCcw, TerminalSquare } from 'lucide-react';

export function StatusBar({ statusText, currentProject, cursor, activeFile, isStreaming }) {
  return (
    <footer className="status-strip">
      <div className="status-left"></div>
      <div className="status-center">{statusText}</div>
      <div className="status-right">
        <span className="status-item">{cursor ? `Ln ${cursor.line}, Col ${cursor.ch}` : 'Ln 1, Col 1'}</span>
        <span className="status-item">Spaces: 2</span>
        <span className="status-item">{activeFile?.split('.').pop()?.toUpperCase() || 'TXT'}</span>
        <span className="status-item"><TerminalSquare size={15} /> {currentProject?.status || 'idle'}</span>
      </div>
    </footer>
  );
}
