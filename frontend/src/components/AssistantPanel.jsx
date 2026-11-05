import { useRef } from 'react';
import {
  Bot,
  ChevronDown,
  Paperclip,
  Send,
  Sparkles,
  X
} from 'lucide-react';
import { simpleMarkdownToHtml } from '../services/projectUtils.js';

function ChatMessage({ message, onTriggerUpdate }) {
  const isUser = message.role === 'user';
  const isThought = message.role === 'thought';

  return (
    <div className={`assistant-message ${isUser ? 'user' : ''} ${isThought ? 'thought' : ''}`}>
      {!isUser && <div className="message-avatar"><Bot size={17} /></div>}
      <div className="message-bubble">
        <div dangerouslySetInnerHTML={{ __html: simpleMarkdownToHtml(message.content || '') }} />
        {message.actionPrompt && (
          <button className="inline-update" type="button" onClick={() => onTriggerUpdate(message.actionPrompt)}>
            <Sparkles size={15} />
            <span>Approve and Run Update</span>
          </button>
        )}
      </div>
    </div>
  );
}

export function AssistantPanel({
  open,
  currentProject,
  chatHistory,
  chatStatus,
  chatMode,
  setChatMode,
  chatInput,
  setChatInput,
  onSendChat,
  onAttachFiles,
  onTriggerUpdate,
  onClose
}) {
  const fileInputRef = useRef(null);

  function submit(event) {
    event.preventDefault();
    onSendChat(chatInput);
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function handleFileSelection(event) {
    const selectedFiles = event.target.files;
    if (selectedFiles?.length) {
      onAttachFiles?.(selectedFiles);
      event.target.value = '';
    }
  }

  return (
    <aside className={`assistant-panel ${open ? 'open' : ''}`}>
      <div className="assistant-header">
        <div className="assistant-title">
          <h2>AI Assistant</h2>
        </div>
        <div className="assistant-header-actions">
          <span className="online-pill">
            <span className="online-pill-dot" />
            <span className="online-pill-label">Online</span>
          </span>
          <button className="panel-close" type="button" onClick={onClose} title="Close assistant">
            <X size={18} />
          </button>
          <ChevronDown size={18} />
        </div>
      </div>

      <div className="assistant-intro">
        <div>
          <h3>Hello i am CodeMind</h3>
        </div>
      </div>

      <div className="assistant-chat-log">
        {chatHistory.length === 0 && (
          <div className="empty-chat">Project messages and update approvals appear here.</div>
        )}
        {chatHistory.map((message) => (
          <ChatMessage key={message.id} message={message} onTriggerUpdate={onTriggerUpdate} />
        ))}
      </div>

      <form className="assistant-input" onSubmit={submit}>
        <input
          ref={fileInputRef}
          className="assistant-file-input"
          type="file"
          multiple
          onChange={handleFileSelection}
        />
        <textarea
          value={chatInput}
          onChange={(event) => setChatInput(event.target.value)}
          placeholder="Ask anything about your code..."
          disabled={!currentProject?.id || chatStatus === 'Thinking...'}
          rows={1}
        />
        <div className="assistant-input-actions">
          <div className="input-left-actions">
            <button className="soft-chip" type="button" onClick={openFilePicker}>
              <Paperclip size={15} />
              <span>Attach</span>
            </button>
            <div className="mode-toggle">
              <button
                className={chatMode === 'chat' ? 'active' : ''}
                type="button"
                onClick={() => setChatMode('chat')}
              >
                Chat
              </button>
              <button
                className={chatMode === 'update' ? 'active' : ''}
                type="button"
                onClick={() => setChatMode('update')}
              >
                Update
              </button>
            </div>
            <span className="chat-status">{chatStatus}</span>
          </div>
          <button className="send-button" type="submit" disabled={!currentProject?.id || !chatInput.trim()}>
            <Send size={19} />
          </button>
        </div>
      </form>
    </aside>
  );
}
