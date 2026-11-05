import { useEffect, useRef } from 'react';
import {
  Check,
  ChevronDown,
  Code2,
  Download,
  FileCode2,
  MoreVertical,
  Play,
  Save,
  X
} from 'lucide-react';
import CodeMirror from 'codemirror';
import 'codemirror/lib/codemirror.css';
import 'codemirror/theme/material-darker.css';
import 'codemirror/mode/python/python.js';
import 'codemirror/mode/javascript/javascript.js';
import 'codemirror/mode/htmlmixed/htmlmixed.js';
import 'codemirror/mode/xml/xml.js';
import 'codemirror/mode/css/css.js';
import 'codemirror/mode/markdown/markdown.js';
import 'codemirror/mode/yaml/yaml.js';
import 'codemirror/mode/sql/sql.js';
import { getCodeMirrorMode } from '../services/projectUtils.js';

function CodeEditor({ filepath, value, onChange, onCursor }) {
  const containerRef = useRef(null);
  const editorRef = useRef(null);
  const onChangeRef = useRef(onChange);
  const onCursorRef = useRef(onCursor);

  useEffect(() => {
    onChangeRef.current = onChange;
    onCursorRef.current = onCursor;
  }, [onChange, onCursor]);

  useEffect(() => {
    if (!containerRef.current || editorRef.current) return undefined;

    const editor = CodeMirror(containerRef.current, {
      value: value || '',
      mode: getCodeMirrorMode(filepath),
      theme: 'material-darker',
      lineNumbers: true,
      lineWrapping: false,
      tabSize: 2,
      indentUnit: 2,
      smartIndent: true,
      viewportMargin: Infinity
    });

    editor.on('change', (instance) => {
      onChangeRef.current?.(instance.getValue());
    });
    editor.on('cursorActivity', (instance) => {
      const cursor = instance.getCursor();
      onCursorRef.current?.({ line: cursor.line + 1, ch: cursor.ch + 1 });
    });

    editorRef.current = editor;
    requestAnimationFrame(() => editor.refresh());

    return () => {
      editor.getWrapperElement()?.remove();
      editorRef.current = null;
    };
  }, []);

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    if (editor.getValue() !== (value || '')) {
      const cursor = editor.getCursor();
      editor.setValue(value || '');
      editor.setCursor(cursor);
    }
  }, [value]);

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    editor.setOption('mode', getCodeMirrorMode(filepath));
    requestAnimationFrame(() => editor.refresh());
  }, [filepath]);

  return <div className="code-editor-host" ref={containerRef} />;
}

export function EditorWorkspace({
  files,
  activeFile,
  activeContent,
  currentProject,
  dirty,
  saving,
  onSelectFile,
  onEditContent,
  onSave,
  onDownload,
  onRunAll,
  onCursor
}) {
  const fileOptions = files || [];
  const activeName = activeFile || 'no_file.py';
  const breadcrumbs = activeName.split('/').filter(Boolean);

  return (
    <section className="editor-pane">
      <div className="editor-tabs">
        <div className="tab active">
          <FileCode2 size={18} />
          <span>{activeName.split('/').pop()}</span>
          <X size={16} />
        </div>
        <button className="tab-add" type="button" title="New tab">+</button>
      </div>

      <div className="editor-toolbar">
        <div className="breadcrumbs">
          <span>src</span>
          {breadcrumbs.map((part, index) => (
            <span key={`${part}-${index}`}>
              <ChevronDown size={14} className="crumb-divider" />
              {part}
            </span>
          ))}
        </div>
        <div className="editor-actions">
          <span className={`save-state ${dirty ? 'dirty' : ''}`}>
            <Check size={16} />
            {dirty ? 'Unsaved changes' : 'All changes saved'}
          </span>
          <button className="run-button" type="button" disabled={!currentProject?.id} onClick={onRunAll}>
            <Play size={17} />
            <span>Run</span>
          </button>
          <button className="tool-button" type="button" disabled={!currentProject?.id || saving} onClick={onSave} title="Save file">
            <Save size={17} />
          </button>
          <button className="tool-button" type="button" disabled={!currentProject?.id} onClick={onDownload} title="Download project">
            <Download size={17} />
          </button>
          <button className="tool-button" type="button" title="More">
            <MoreVertical size={17} />
          </button>
        </div>
      </div>

      <div className="file-select-row">
        <Code2 size={16} />
        <select value={activeName} onChange={(event) => onSelectFile(event.target.value)}>
          {fileOptions.length === 0 && <option value="no_file.py">no_file.py</option>}
          {fileOptions.map((file) => (
            <option key={file.filepath} value={file.filepath}>{file.filepath}</option>
          ))}
        </select>
      </div>

      <div className="editor-body">
        <CodeEditor
          filepath={activeName}
          value={activeContent}
          onChange={onEditContent}
          onCursor={onCursor}
        />
      </div>
    </section>
  );
}
