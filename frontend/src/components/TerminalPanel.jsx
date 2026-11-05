import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { ChevronUp, PanelBottom, Plus, SplitSquareHorizontal, Trash2 } from 'lucide-react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';
import { WS_BASE_URL } from '../services/api.js';

function normalizeTerminalText(value) {
  return String(value || '').replace(/\n/g, '\r\n');
}

export const TerminalPanel = forwardRef(function TerminalPanel({ boot }, ref) {
  const hostRef = useRef(null);
  const terminalsRef = useRef({});
  const activeRef = useRef(null);
  const [tabs, setTabs] = useState([]);
  const [activeId, setActiveId] = useState(null);

  function syncTabs(nextActiveId = activeRef.current) {
    const nextTabs = Object.entries(terminalsRef.current).map(([id, terminal]) => ({
      id,
      name: terminal.name
    }));
    setTabs(nextTabs);
    setActiveId(nextActiveId);
  }

  function switchTerminal(termId) {
    const terminal = terminalsRef.current[termId];
    if (!terminal) return;

    Object.entries(terminalsRef.current).forEach(([id, entry]) => {
      entry.element.style.display = id === termId ? 'block' : 'none';
    });

    activeRef.current = termId;
    syncTabs(termId);
    requestAnimationFrame(() => terminal.fit.fit());
  }

  function closeTerminal(termId) {
    const entry = terminalsRef.current[termId];
    if (!entry) return;

    entry.ws?.close();
    entry.subscription?.dispose();
    entry.term.dispose();
    entry.element.remove();
    delete terminalsRef.current[termId];

    const remaining = Object.keys(terminalsRef.current);
    if (activeRef.current === termId) {
      activeRef.current = remaining[0] || null;
    }
    syncTabs(activeRef.current);
    if (activeRef.current) switchTerminal(activeRef.current);
  }

  function closeAll() {
    Object.keys(terminalsRef.current).forEach((termId) => closeTerminal(termId));
    terminalsRef.current = {};
    activeRef.current = null;
    syncTabs(null);
  }

  function createTerminal({ termId, name, threadId, command, initialText }) {
    if (!hostRef.current || !threadId) return null;
    if (terminalsRef.current[termId]) {
      switchTerminal(termId);
      return terminalsRef.current[termId];
    }

    const element = document.createElement('div');
    element.className = 'terminal-surface';
    element.style.display = 'none';
    hostRef.current.appendChild(element);

    const term = new Terminal({
      cursorBlink: true,
      convertEol: true,
      fontFamily: "'JetBrains Mono', Consolas, monospace",
      fontSize: 13,
      lineHeight: 1.45,
      theme: {
        background: '#050914',
        foreground: '#dbe7ff',
        cursor: '#8fb7ff',
        black: '#050914',
        blue: '#4f8cff',
        brightBlue: '#79a8ff',
        brightCyan: '#49dce4',
        brightGreen: '#63e38b',
        brightMagenta: '#9d6cff',
        cyan: '#20c4cf',
        green: '#37c96d',
        magenta: '#7b4dff',
        red: '#ff5f6d',
        yellow: '#ffd166'
      }
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(element);

    const ws = new WebSocket(`${WS_BASE_URL}/ws/terminal/${encodeURIComponent(threadId)}/${encodeURIComponent(termId)}`);
    const output = [];
    const write = (text) => {
      const normalized = normalizeTerminalText(text);
      output.push(normalized);
      if (output.length > 120) output.shift();
      term.write(normalized);
    };

    const subscription = term.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(data);
      }
    });

    ws.onopen = () => {
      write(`\x1b[32mConnected to Terminal: ${name}\x1b[0m\r\n\r\n`);
      if (initialText) write(initialText);
      if (command) {
        write(command);
        ws.send(command);
      }
    };
    ws.onmessage = (event) => write(event.data);
    ws.onclose = () => write('\r\n\x1b[31mTerminal connection closed\x1b[0m\r\n');

    terminalsRef.current[termId] = { term, fit, ws, element, name, output, subscription, write };
    switchTerminal(termId);
    requestAnimationFrame(() => fit.fit());
    return terminalsRef.current[termId];
  }

  function createAdHocTerminal(command = null, name = 'Terminal') {
    if (!boot?.projectId) return null;
    const termId = `term-${Math.random().toString(36).slice(2, 8)}`;
    return createTerminal({ termId, name, threadId: boot.projectId, command });
  }

  useImperativeHandle(ref, () => ({
    connectDefault(threadId, initialText = '') {
      closeAll();
      createTerminal({ termId: 'default', name: 'Terminal', threadId, initialText });
    },
    createCommandTerminal(command, threadId = boot?.projectId) {
      if (!threadId || !command) return;
      const termId = `run-${Math.random().toString(36).slice(2, 8)}`;
      createTerminal({
        termId,
        name: command.name || 'Run',
        threadId,
        command: command.cmd || ''
      });
    },
    write(text) {
      const active = terminalsRef.current[activeRef.current];
      if (active) active.write(text);
    },
    clear() {
      const active = terminalsRef.current[activeRef.current];
      if (active) active.term.clear();
    },
    getRecentOutput() {
      const active = terminalsRef.current[activeRef.current];
      return active ? active.output.join('') : '';
    }
  }));

  useEffect(() => {
    if (boot?.projectId && boot.key) {
      closeAll();
      createTerminal({
        termId: 'default',
        name: 'Terminal',
        threadId: boot.projectId,
        initialText: boot.text || ''
      });
    }
  }, [boot?.key]);

  useEffect(() => {
    function handleResize() {
      Object.values(terminalsRef.current).forEach((entry) => entry.fit.fit());
    }
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      closeAll();
    };
  }, []);

  return (
    <section className="terminal-pane">
      <div className="terminal-header">
        <div className="terminal-tabs">
          {tabs.length === 0 && <button className="terminal-tab active" type="button">Terminal</button>}
          {tabs.map((tab) => (
            <button
              className={`terminal-tab ${activeId === tab.id ? 'active' : ''}`}
              type="button"
              key={tab.id}
              onClick={() => switchTerminal(tab.id)}
            >
              <PanelBottom size={15} />
              <span>{tab.name}</span>
              {tab.id !== 'default' && (
                <span
                  className="tab-close"
                  role="button"
                  tabIndex={0}
                  onClick={(event) => {
                    event.stopPropagation();
                    closeTerminal(tab.id);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      event.stopPropagation();
                      closeTerminal(tab.id);
                    }
                  }}
                >
                  x
                </span>
              )}
            </button>
          ))}
        </div>
        <div className="terminal-tools">
          <button className="tool-button" type="button" onClick={() => createAdHocTerminal()} title="New terminal">
            <Plus size={17} />
          </button>
          <button className="tool-button" type="button" title="Split terminal">
            <SplitSquareHorizontal size={17} />
          </button>
          <button className="tool-button" type="button" onClick={() => activeRef.current && closeTerminal(activeRef.current)} title="Close terminal">
            <Trash2 size={17} />
          </button>
          <button className="tool-button" type="button" title="Collapse">
            <ChevronUp size={17} />
          </button>
        </div>
      </div>
      <div className="terminal-host" ref={hostRef}>
        {tabs.length === 0 && (
          <div className="terminal-empty">
            <span className="online-dot" />
            Connected to Terminal
            <pre>PS C:\Projects\codemind&gt;</pre>
          </div>
        )}
      </div>
    </section>
  );
});
