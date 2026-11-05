import {
  CheckCircle2,
  ChevronRight,
  FileCode2,
  Filter,
  Folder,
  FolderOpen,
  Play,
  Plus,
  Sparkles,
  Trash2,
  Upload
} from 'lucide-react';
import {
  buildFileRunCommand,
  canRunProjectStatus,
  formatProjectTime,
  getCurrentRunPlan,
  shortProjectTitle
} from '../services/projectUtils.js';

function ProjectCard({ project, active, onLoad, onDelete, onResume }) {
  const isCompleted = project.status === 'done' || project.status === 'rolled_back';
  return (
    <button className={`project-card ${active ? 'active' : ''}`} type="button" onClick={() => onLoad(project.id)}>
      <div className="project-icon"><FileCode2 size={21} /></div>
      <div className="project-copy">
        <strong>{shortProjectTitle(project)}</strong>
        <span>{formatProjectTime(project.created_at)}</span>
      </div>
      <CheckCircle2 className={`project-state ${project.status || ''}`} size={18} />
      
      {!isCompleted && (
        <span
          role="button"
          tabIndex={0}
          className="continue-project-btn"
          title="Resume project"
          onClick={(event) => {
            event.stopPropagation();
            onResume(project.id);
          }}
        >
          <Play size={15} />
        </span>
      )}
      <span
        role="button"
        tabIndex={0}
        className="delete-inline"
        title="Delete project"
        onClick={(event) => {
          event.stopPropagation();
          onDelete(project.id);
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            event.stopPropagation();
            onDelete(project.id);
          }
        }}
      >
        <Trash2 size={15} />
      </span>
    </button>
  );
}

function HistoryPanel({
  projects,
  currentProject,
  loading,
  onNew,
  onLoad,
  onDelete,
  onResume,
  onUploadFiles,
  onOpenFolder
}) {
  const completed = projects.filter((project) => project.status === 'done' || project.status === 'rolled_back');
  const active = projects.filter((project) => project.status !== 'done' && project.status !== 'rolled_back');
  const visible = [...active, ...completed].slice(0, 12);

  return (
    <div className="sidebar-panel active">
      <div className="panel-title-row">
        <h2>Recent Projects</h2>
        <button className="ghost-square" type="button" title="Filter">
          <Filter size={18} />
        </button>
      </div>
      <div className="sidebar-action-row">
        <button className="new-project-button" type="button" onClick={onNew}>
          <Plus size={18} />
          <span>New Project</span>
        </button>
        <label className="mini-action" title="Upload files">
          <Upload size={17} />
          <input type="file" multiple hidden onChange={(event) => onUploadFiles(event.target.files)} />
        </label>
        <label className="mini-action" title="Open folder">
          <FolderOpen size={17} />
          <input
            type="file"
            multiple
            webkitdirectory=""
            directory=""
            hidden
            onChange={(event) => onOpenFolder(event.target.files)}
          />
        </label>
      </div>

      <div className="project-list">
        {loading && <div className="empty-state">Loading projects...</div>}
        {!loading && visible.length === 0 && <div className="empty-state">No projects yet.</div>}
        {!loading && visible.map((project) => (
          <ProjectCard
            key={project.id}
            project={project}
            active={currentProject?.id === project.id}
            onLoad={onLoad}
            onDelete={onDelete}
            onResume={onResume}
          />
        ))}
      </div>
    </div>
  );
}

function FileTreePanel({ structure, activeFile, onOpenFile, project, files, onRunFile }) {
  const entries = Array.isArray(structure) ? structure : [];
  const canRun = canRunProjectStatus(project?.status);

  return (
    <div className="sidebar-panel active">
      <div className="panel-title-row">
        <h2>Files</h2>
        <button
          className="compact-run"
          type="button"
          disabled={!canRun || !buildFileRunCommand(project, files, activeFile)}
          onClick={() => onRunFile(activeFile)}
        >
          <Play size={15} />
          <span>Run File</span>
        </button>
      </div>
      <div className="tree-list">
        {entries.length === 0 && <div className="empty-state">Load a project to inspect files.</div>}
        {entries.map((entry) => {
          const isFile = entry.type === 'file';
          return (
            <button
              key={`${entry.type}-${entry.path}`}
              className={`tree-row ${isFile ? 'file' : 'folder'} ${activeFile === entry.path ? 'active' : ''}`}
              style={{ paddingLeft: `${12 + (entry.depth || 0) * 14}px` }}
              type="button"
              disabled={!isFile}
              onClick={() => isFile && onOpenFile(entry.path)}
            >
              {isFile ? <FileCode2 size={16} /> : <Folder size={16} />}
              <span>{entry.name || entry.path}</span>
              {isFile && <ChevronRight size={15} />}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function RunPanel({ project, onRunAll, onRunCommand }) {
  const runPlan = getCurrentRunPlan(project);
  const commands = Array.isArray(runPlan.commands) ? runPlan.commands : [];
  const canRun = canRunProjectStatus(project?.status);

  return (
    <div className="sidebar-panel active">
      <div className="panel-title-row">
        <h2>Run</h2>
        <button className="compact-run" type="button" disabled={!project?.id || !canRun} onClick={onRunAll}>
          <Play size={15} />
          <span>Run All</span>
        </button>
      </div>
      <div className="run-list">
        {commands.length === 0 && <div className="empty-state">Run commands appear after project analysis.</div>}
        {commands.map((command, index) => (
          <button className="run-card" type="button" key={`${command.name}-${index}`} onClick={() => onRunCommand(command)}>
            <div>
              <strong>{command.name || `Command ${index + 1}`}</strong>
              <code>{command.cmd}</code>
            </div>
            <Play size={17} />
          </button>
        ))}
      </div>
    </div>
  );
}

function SettingsPanel({ autoFallback, setAutoFallback }) {
  return (
    <div className="sidebar-panel active">
      <div className="panel-title-row">
        <h2>Settings</h2>
      </div>
      <label className="setting-card">
        <div>
          <strong>Auto-switch on model limits</strong>
          <p>Retry with configured backup models when rate, quota, or context limits are hit.</p>
        </div>
        <input
          type="checkbox"
          checked={autoFallback}
          onChange={(event) => setAutoFallback(event.target.checked)}
        />
      </label>
      <div className="setting-card stacked">
        <strong>Fallback routing</strong>
        <p>Planner, architect, coder, and retriever roles keep their existing backend routing.</p>
      </div>
    </div>
  );
}

function SimplePanel({ title, children }) {
  return (
    <div className="sidebar-panel active">
      <div className="panel-title-row">
        <h2>{title}</h2>
      </div>
      <div className="empty-state panel-fill">{children}</div>
    </div>
  );
}

export function ProjectSidebar(props) {
  const {
    activePanel,
    projects,
    currentProject,
    projectLoading,
    structure,
    activeFile,
    files,
    autoFallback,
    setAutoFallback,
    onNew,
    onLoadProject,
    onDeleteProject,
    onResumeProject,
    onUploadFiles,
    onOpenFolder,
    onOpenFile,
    onRunAll,
    onRunFile,
    onRunCommand
  } = props;

  return (
    <aside className="project-sidebar">
      {activePanel === 'history' && (
        <HistoryPanel
          projects={projects}
          currentProject={currentProject}
          loading={projectLoading}
          onNew={onNew}
          onLoad={onLoadProject}
          onDelete={onDeleteProject}
          onResume={onResumeProject}
          onUploadFiles={onUploadFiles}
          onOpenFolder={onOpenFolder}
        />
      )}
      {activePanel === 'files' && (
        <FileTreePanel
          structure={structure}
          activeFile={activeFile}
          onOpenFile={onOpenFile}
          project={currentProject}
          files={files}
          onRunFile={onRunFile}
        />
      )}
      {activePanel === 'run' && (
        <RunPanel
          project={currentProject}
          onRunAll={onRunAll}
          onRunCommand={onRunCommand}
        />
      )}
      {activePanel === 'settings' && (
        <SettingsPanel autoFallback={autoFallback} setAutoFallback={setAutoFallback} />
      )}
      {activePanel === 'deploy' && (
        <SimplePanel title="Deploy">Deployment controls will stay attached to the same project runtime once enabled.</SimplePanel>
      )}
      {activePanel === 'help' && (
        <SimplePanel title="Help">Use the assistant panel for project-specific questions and updates.</SimplePanel>
      )}
    </aside>
  );
}
