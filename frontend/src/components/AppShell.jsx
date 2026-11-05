import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityRail } from './ActivityRail.jsx';
import { AssistantPanel } from './AssistantPanel.jsx';
import { EditorWorkspace } from './EditorWorkspace.jsx';
import { HeaderBar } from './HeaderBar.jsx';
import { ProjectSidebar } from './ProjectSidebar.jsx';
import { StatusBar } from './StatusBar.jsx';
import { TerminalPanel } from './TerminalPanel.jsx';
import { useProjects } from '../hooks/useProjects.js';
import { useProjectStream } from '../hooks/useProjectStream.js';
import {
  extractThreadId,
  generateProject,
  getAutoModelFallbackSetting,
  getDownloadUrl,
  getProject,
  runProject,
  saveProjectFile,
  setAutoModelFallbackSetting,
  streamProjectChat,
  updateProject,
  resumeProject as resumeProjectGeneration,
  uploadProjectFiles
} from '../services/api.js';
import {
  buildFallbackStructure,
  buildFileRunCommand,
  getCurrentRunPlan,
  shortProjectTitle
} from '../services/projectUtils.js';

function createMessage(role, content, extra = {}) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    content,
    ...extra
  };
}

function unwrapStatusMessage(message) {
  if (!message) return '';
  if (typeof message === 'string') return message;
  if (typeof message.data === 'string') return message.data;
  if (typeof message.message === 'string') return message.message;
  if (message.data && typeof message.data.message === 'string') return message.data.message;
  return '';
}

export function AppShell() {
  const terminalRef = useRef(null);
  const initialLoadRef = useRef(false);
  const { projects, loading: projectLoading, refreshProjects, removeProject } = useProjects();

  const [activePanel, setActivePanel] = useState('history');
  const [prompt, setPrompt] = useState('');
  const [currentProject, setCurrentProject] = useState(null);
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState('no_file.py');
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState('Ready');
  const [streamProjectId, setStreamProjectId] = useState(null);
  const [terminalBoot, setTerminalBoot] = useState(null);
  const [cursor, setCursor] = useState({ line: 1, ch: 1 });
  const [assistantOpen, setAssistantOpen] = useState(() => window.matchMedia('(min-width: 1381px)').matches);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatMode, setChatMode] = useState('chat');
  const [chatStatus, setChatStatus] = useState('Idle');
  const [autoFallback, setAutoFallback] = useState(getAutoModelFallbackSetting);

  const activeContent = useMemo(() => {
    return files.find((file) => file.filepath === activeFile)?.content || '';
  }, [files, activeFile]);

  const structure = useMemo(() => {
    if (currentProject?.structure?.length) return currentProject.structure;
    return buildFallbackStructure(files);
  }, [currentProject, files]);

  const applyAutoFallback = useCallback((enabled) => {
    setAutoFallback(enabled);
    setAutoModelFallbackSetting(enabled);
  }, []);

  const setLoadedProject = useCallback((data) => {
    const loadedFiles = Array.isArray(data.files) ? data.files : [];
    const runPlan = data.run_plan || data.runPlan || { commands: [], html_file: null, html_path: null, serve_port: null };
    const project = {
      id: data.id,
      prompt: data.prompt || 'Untitled project',
      status: data.status || 'idle',
      structure: Array.isArray(data.structure) ? data.structure : buildFallbackStructure(loadedFiles),
      runPlan
    };

    setCurrentProject(project);
    setFiles(loadedFiles);
    setActiveFile(loadedFiles.length > 0 ? loadedFiles[loadedFiles.length - 1].filepath : 'no_file.py');
    setDirty(false);
    setStatusText(`Status: ${project.status}`);
    setPrompt('');

    if (project.status === 'running' || project.status === 'pending') {
      setStreamProjectId(project.id);
    } else {
      setStreamProjectId(null);
      setTerminalBoot({
        key: Date.now(),
        projectId: project.id,
        text: data.terminal_output || ''
      });
    }

    window.history.replaceState({}, '', `?project=${project.id}`);
  }, []);

  const loadProject = useCallback(async (threadId, options = {}) => {
    if (!threadId) return;
    const maxRetries = options.waitForSettledStatus ? 6 : 0;
    const retryDelayMs = 450;
    setStatusText('Loading project...');

    try {
      let data = null;
      for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
        data = await getProject(threadId);
        const stillRunning = data.status === 'running' || data.status === 'pending';
        if (!options.waitForSettledStatus || !stillRunning || attempt === maxRetries) {
          break;
        }
        setStatusText('Finalizing project state...');
        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
      }

      setLoadedProject(data);
      await refreshProjects();
    } catch (error) {
      setStatusText(error.message || 'Failed to load project.');
    }
  }, [refreshProjects, setLoadedProject]);

  useEffect(() => {
    if (initialLoadRef.current) return;
    initialLoadRef.current = true;
    const projectFromUrl = new URLSearchParams(window.location.search).get('project');
    if (projectFromUrl) {
      loadProject(projectFromUrl);
    }
  }, [loadProject]);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(min-width: 1381px)');
    const syncAssistant = () => {
      if (mediaQuery.matches) setAssistantOpen(true);
    };
    syncAssistant();
    mediaQuery.addEventListener('change', syncAssistant);
    return () => mediaQuery.removeEventListener('change', syncAssistant);
  }, []);

  const startGenerationOrUpdate = useCallback(async (promptOverride = null) => {
    const nextPrompt = String(promptOverride ?? prompt).trim();
    if (nextPrompt.length < 5 || busy) return;

    const isUpdate = Boolean(currentProject?.id);
    setBusy(true);
    setStatusText(isUpdate ? 'Starting update...' : 'Planning project...');

    if (!isUpdate) {
      setFiles([]);
      setActiveFile('no_file.py');
      setCurrentProject(null);
      setTerminalBoot(null);
      setChatHistory([]);
      window.history.replaceState({}, '', window.location.pathname);
    }

    try {
      const response = isUpdate
        ? await updateProject(currentProject.id, nextPrompt, autoFallback)
        : await generateProject(nextPrompt, autoFallback);
      const threadId = extractThreadId(response);
      if (!threadId) throw new Error('The backend did not return a project id.');

      setCurrentProject((previous) => ({
        ...(previous || {}),
        id: threadId,
        prompt: isUpdate ? previous?.prompt || nextPrompt : nextPrompt,
        status: 'running',
        structure: previous?.structure || [],
        runPlan: previous?.runPlan || { commands: [] }
      }));
      setStatusText(isUpdate ? 'Updating project...' : 'Generating project...');
      setStreamProjectId(threadId);
      setPrompt('');
      window.history.replaceState({}, '', `?project=${threadId}`);
      await refreshProjects();
    } catch (error) {
      setBusy(false);
      setStatusText(error.message || 'Project request failed.');
    }
  }, [autoFallback, busy, currentProject, prompt, refreshProjects]);

  const handleStreamMessage = useCallback((message) => {
    switch (message.type) {
      case 'status':
        setStatusText(unwrapStatusMessage(message.data) || 'Working...');
        break;
      case 'node_finished':
        setStatusText(`Completed: ${message.data?.node || 'step'}`);
        break;
      case 'file_update':
        if (message.data?.filepath) {
          setFiles((previous) => {
            const next = [...previous];
            const index = next.findIndex((file) => file.filepath === message.data.filepath);
            if (index >= 0) {
              next[index] = { ...next[index], content: message.data.content || '' };
            } else {
              next.push({ filepath: message.data.filepath, content: message.data.content || '' });
            }
            setCurrentProject((project) => project ? { ...project, structure: buildFallbackStructure(next) } : project);
            return next;
          });
          setActiveFile(message.data.filepath);
          setDirty(false);
        }
        break;
      case 'thoughts':
        terminalRef.current?.write(`\x1b[36m[Thought]\x1b[0m ${String(message.data || '')}\r\n`);
        if (String(message.data || '').trim()) {
          setChatHistory((previous) => [...previous, createMessage('thought', String(message.data))]);
        }
        break;
      case 'tool_calls':
        terminalRef.current?.write(`\x1b[35m[Tool Call]\x1b[0m ${JSON.stringify(message.data)}\r\n`);
        break;
      case 'terminal_output':
        terminalRef.current?.write(`${String(message.data || '').replace(/\\n/g, '\r\n')}\r\n`);
        break;
      case 'done': {
        const doneStatus = message.data?.status || 'done';
        const doneMessage = unwrapStatusMessage(message.data);
        const runPlan = message.data?.run_plan || message.data?.runPlan;
        setBusy(false);
        setChatStatus('Idle');
        setPrompt('');
        setCurrentProject((project) => project ? {
          ...project,
          status: doneStatus,
          structure: Array.isArray(message.data?.structure) ? message.data.structure : project.structure,
          runPlan: runPlan || project.runPlan
        } : project);
        setStatusText(doneStatus === 'rolled_back'
          ? doneMessage || 'Project restored to the last stable snapshot.'
          : 'Task finished. Run controls are ready.');
        setStreamProjectId(null);
        setTimeout(() => loadProject(message.data?.thread_id || currentProject?.id || streamProjectId, {
          waitForSettledStatus: true
        }), 550);
        break;
      }
      case 'error':
        setBusy(false);
        setStatusText(unwrapStatusMessage(message.data) || 'Stream error.');
        break;
      default:
        break;
    }
  }, [currentProject?.id, loadProject, streamProjectId]);

  useProjectStream({
    threadId: streamProjectId,
    enabled: Boolean(streamProjectId),
    onOpen: () => setStatusText('Connected to project stream.'),
    onMessage: handleStreamMessage,
    onClose: () => {
      if (busy) setStatusText('Project stream closed.');
    }
  });

  const selectFile = useCallback((filepath) => {
    setActiveFile(filepath);
    setDirty(false);
  }, []);

  const editActiveContent = useCallback((content) => {
    setDirty(true);
    setFiles((previous) => {
      const index = previous.findIndex((file) => file.filepath === activeFile);
      if (index < 0) {
        return activeFile === 'no_file.py' ? previous : [...previous, { filepath: activeFile, content }];
      }
      const next = [...previous];
      next[index] = { ...next[index], content };
      return next;
    });
  }, [activeFile]);

  const saveActiveFile = useCallback(async () => {
    if (!currentProject?.id || !activeFile || activeFile === 'no_file.py') return;
    setSaving(true);
    setStatusText('Saving file...');
    try {
      await saveProjectFile(currentProject.id, activeFile, activeContent);
      setDirty(false);
      setStatusText('File saved.');
    } catch (error) {
      setStatusText(error.message || 'Failed to save file.');
    } finally {
      setSaving(false);
    }
  }, [activeContent, activeFile, currentProject?.id]);

  const runCommand = useCallback((command) => {
    if (!currentProject?.id || !command) return;
    terminalRef.current?.createCommandTerminal(command, currentProject.id);
    setStatusText(`Running: ${command.name || 'command'}`);
  }, [currentProject?.id]);

  const runAll = useCallback(async () => {
    if (!currentProject?.id) return;
    let commands = getCurrentRunPlan(currentProject).commands || [];
    if (commands.length === 0) {
      try {
        const plan = await runProject(currentProject.id);
        commands = plan.commands || [];
        setCurrentProject((project) => project ? { ...project, runPlan: plan } : project);
      } catch (error) {
        setStatusText(error.message || 'Unable to build run plan.');
        return;
      }
    }

    if (commands.length === 0) {
      setStatusText('No run commands were detected.');
      return;
    }

    commands.forEach((command) => runCommand(command));
  }, [currentProject, runCommand]);

  const runFile = useCallback((filepath) => {
    const command = buildFileRunCommand(currentProject, files, filepath);
    if (command) {
      runCommand(command);
    } else {
      setStatusText('No run command is available for this file.');
    }
  }, [currentProject, files, runCommand]);

  const downloadProject = useCallback(() => {
    if (!currentProject?.id) return;
    window.location.href = getDownloadUrl(currentProject.id);
  }, [currentProject?.id]);

  const uploadFiles = useCallback(async (fileList) => {
    const selectedFiles = Array.from(fileList || []);
    if (selectedFiles.length === 0) return;
    setBusy(true);
    setStatusText('Importing project...');
    try {
      const response = await uploadProjectFiles(selectedFiles);
      const threadId = extractThreadId(response);
      await refreshProjects();
      if (threadId) {
        await loadProject(threadId);
      }
      setStatusText('Project imported.');
    } catch (error) {
      setStatusText(error.message || 'Project import failed.');
    } finally {
      setBusy(false);
    }
  }, [loadProject, refreshProjects]);

  const deleteSelectedProject = useCallback(async (threadId) => {
    if (!threadId) return;
    const confirmed = window.confirm('Delete this project and its generated files?');
    if (!confirmed) return;
    try {
      await removeProject(threadId);
      if (currentProject?.id === threadId) {
        setCurrentProject(null);
        setFiles([]);
        setActiveFile('no_file.py');
        setTerminalBoot(null);
        setStatusText('Project deleted.');
        window.history.replaceState({}, '', window.location.pathname);
      }
    } catch (error) {
      setStatusText(error.message || 'Unable to delete project.');
    }
  }, [currentProject?.id, removeProject]);

  const sendChat = useCallback(async (text) => {
    const message = String(text || '').trim();
    if (!message || !currentProject?.id) return;
    setChatInput('');
    setAssistantOpen(true);

    if (chatMode === 'update') {
      setChatHistory((previous) => [
        ...previous,
        createMessage('user', message),
        createMessage('assistant', 'Update pipeline started. File changes will appear as they stream in.')
      ]);
      await startGenerationOrUpdate(message);
      return;
    }

    const assistantId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setChatStatus('Thinking...');
    setChatHistory((previous) => [
      ...previous,
      createMessage('user', message),
      { id: assistantId, role: 'assistant', content: '' }
    ]);

    try {
      await streamProjectChat(currentProject.id, message, {
        onEvent(event) {
          if (event.type === 'token') {
            setChatHistory((previous) => previous.map((entry) => (
              entry.id === assistantId ? { ...entry, content: `${entry.content}${event.content || ''}` } : entry
            )));
          } else if (event.type === 'status') {
            setChatStatus(event.content || 'Working...');
          } else if (event.type === 'tool_call') {
            setChatStatus(`Using ${event.name || 'tool'}...`);
          } else if (event.type === 'action' && event.action_type === 'trigger_update') {
            setChatHistory((previous) => previous.map((entry) => (
              entry.id === assistantId
                ? {
                    ...entry,
                    content: `${entry.content}\n\nI prepared an update for this project.`,
                    actionPrompt: event.update_prompt
                  }
                : entry
            )));
          } else if (event.type === 'error') {
            setChatHistory((previous) => previous.map((entry) => (
              entry.id === assistantId ? { ...entry, content: event.content || 'Chat failed.' } : entry
            )));
          }
        },
        onDone() {
          setChatStatus('Idle');
        }
      });
    } catch (error) {
      setChatHistory((previous) => previous.map((entry) => (
        entry.id === assistantId ? { ...entry, content: error.message || 'Chat failed.' } : entry
      )));
    } finally {
      setChatStatus('Idle');
    }
  }, [chatMode, currentProject?.id, startGenerationOrUpdate]);

  const newProject = useCallback(() => {
    setCurrentProject(null);
    setFiles([]);
    setActiveFile('no_file.py');
    setPrompt('');
    setChatHistory([]);
    setTerminalBoot(null);
    setStatusText('Ready for a new project.');
    setActivePanel('history');
    window.history.replaceState({}, '', window.location.pathname);
  }, []);

  const resumeSelectedProject = useCallback(async (threadId) => {
    if (!threadId || busy) return;
    
    if (currentProject?.id !== threadId) {
      await loadProject(threadId, { waitForSettledStatus: false });
    }
    
    setAssistantOpen(true);
    setChatMode('update');
    setChatHistory((previous) => [
      ...previous,
      createMessage('user', 'Please continue the project generation.'),
      createMessage('assistant', 'Update pipeline started. File changes will appear as they stream in.')
    ]);
    
    setBusy(true);
    setStatusText('Resuming project...');
    
    try {
      await resumeProjectGeneration(threadId);
      setCurrentProject((previous) => ({
        ...(previous || {}),
        id: threadId,
        status: 'running',
      }));
      setStreamProjectId(threadId);
      window.history.replaceState({}, '', `?project=${threadId}`);
      await refreshProjects();
    } catch (error) {
      setBusy(false);
      setStatusText(error.message || 'Project resume failed.');
    }
  }, [busy, currentProject?.id, loadProject, refreshProjects]);

  return (
    <div className="app-shell">
      <HeaderBar
        prompt={prompt}
        setPrompt={setPrompt}
        currentProject={currentProject}
        isBusy={busy}
        onSubmit={() => startGenerationOrUpdate()}
        onToggleAssistant={() => setAssistantOpen((open) => !open)}
      />

      <div className={`workbench ${assistantOpen ? 'assistant-open' : ''}`}>
        <ActivityRail activePanel={activePanel} onSelect={setActivePanel} />
        <ProjectSidebar
          activePanel={activePanel}
          projects={projects}
          currentProject={currentProject}
          projectLoading={projectLoading}
          structure={structure}
          activeFile={activeFile}
          files={files}
          autoFallback={autoFallback}
          setAutoFallback={applyAutoFallback}
          onNew={newProject}
          onLoadProject={loadProject}
          onDeleteProject={deleteSelectedProject}
          onResumeProject={resumeSelectedProject}
          onUploadFiles={uploadFiles}
          onOpenFolder={uploadFiles}
          onOpenFile={selectFile}
          onRunAll={runAll}
          onRunFile={runFile}
          onRunCommand={runCommand}
        />

        <main className="workspace-grid">
          <EditorWorkspace
            files={files}
            activeFile={activeFile}
            activeContent={activeContent || '# CodeMind is ready.'}
            currentProject={currentProject}
            dirty={dirty}
            saving={saving}
            onSelectFile={selectFile}
            onEditContent={editActiveContent}
            onSave={saveActiveFile}
            onDownload={downloadProject}
            onRunAll={runAll}
            onCursor={setCursor}
          />
          <TerminalPanel ref={terminalRef} boot={terminalBoot} />
        </main>

        <AssistantPanel
          open={assistantOpen}
          currentProject={currentProject}
          chatHistory={chatHistory}
          chatStatus={chatStatus}
          chatMode={chatMode}
          setChatMode={setChatMode}
          chatInput={chatInput}
          setChatInput={setChatInput}
          onSendChat={sendChat}
          onAttachFiles={uploadFiles}
          onTriggerUpdate={startGenerationOrUpdate}
          onClose={() => setAssistantOpen(false)}
        />
      </div>

      <StatusBar
        statusText={statusText}
        currentProject={currentProject}
        cursor={cursor}
        activeFile={activeFile}
        isStreaming={Boolean(streamProjectId)}
      />
      <div className="mobile-assistant-toggle">
        <button className="primary-action" type="button" onClick={() => setAssistantOpen(true)}>
          Open Assistant
        </button>
      </div>
      {currentProject && <span className="sr-only">{shortProjectTitle(currentProject)}</span>}
    </div>
  );
}
