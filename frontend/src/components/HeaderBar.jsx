import {
  ChevronDown,
  Search,
  Sparkles
} from 'lucide-react';
import { shortProjectTitle } from '../services/projectUtils.js';

export function HeaderBar({
  prompt,
  setPrompt,
  currentProject,
  isBusy,
  onSubmit,
  onToggleAssistant
}) {
  const label = currentProject ? shortProjectTitle(currentProject) : 'create a new project';

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  }

  return (
    <header className="top-shell">
      <div className="brand-block" aria-label="CodeMind">
        <div className="brand-text">Code<span>Mind</span></div>
      </div>

      <button className="project-pill" type="button" title={label}>
        <span>{label}</span>
        <ChevronDown size={16} />
      </button>

      <div className="command-bar">
        <Search size={19} />
        <input
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={currentProject ? 'Describe what you want to fix or update...' : 'Describe the project you want CodeMind to build...'}
          disabled={isBusy}
        />
        <kbd>Ctrl /</kbd>
      </div>

      <button className="primary-action" type="button" onClick={onSubmit} disabled={isBusy || prompt.trim().length < 5}>
        <span>{currentProject ? 'Update Project' : 'Generate Project'}</span>
        <Sparkles size={16} />
      </button>

      <div className="header-icons">
        <button className="chatbot-btn" type="button" onClick={onToggleAssistant} title="Open Chatbot">
          <span>Chatbot</span>
        </button>
      </div>
    </header>
  );
}
