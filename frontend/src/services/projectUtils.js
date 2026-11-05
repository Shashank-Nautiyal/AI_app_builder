const FRONTEND_DIR_HINTS = new Set(['client', 'frontend', 'public', 'ui', 'web']);
const FRONTEND_DEP_HINTS = new Set(['@angular/core', 'next', 'react', 'react-dom', 'react-scripts', 'svelte', 'vite', 'vue']);
const FRONTEND_SCRIPT_HINTS = ['next dev', 'react-scripts', 'vite', 'vue-cli-service serve', 'webpack serve'];

export function normalizeProjectPath(filepath) {
  return String(filepath || '').replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');
}

export function isTerminalReadyStatus(status) {
  return status === 'done' || status === 'rolled_back';
}

export function canRunProjectStatus(status) {
  return isTerminalReadyStatus(status) || status === 'failed' || status === 'running';
}

export function buildFallbackStructure(files) {
  const directoryMap = new Map();
  const fileEntries = [];

  (files || []).forEach((file) => {
    const filepath = normalizeProjectPath(file.filepath);
    if (!filepath) return;
    const parts = filepath.split('/');
    parts.pop();
    let current = '';
    parts.forEach((part, index) => {
      current = current ? `${current}/${part}` : part;
      if (!directoryMap.has(current)) {
        directoryMap.set(current, {
          type: 'directory',
          path: current,
          name: part,
          depth: index
        });
      }
    });
    fileEntries.push({
      type: 'file',
      path: filepath,
      name: filepath.split('/').pop(),
      depth: filepath.split('/').length - 1
    });
  });

  return [...directoryMap.values(), ...fileEntries].sort((a, b) => a.path.localeCompare(b.path));
}

export function getCodeMirrorMode(filepath) {
  const lower = normalizeProjectPath(filepath).toLowerCase();
  if (lower.endsWith('.py')) return 'python';
  if (lower.endsWith('.html') || lower.endsWith('.htm')) return 'htmlmixed';
  if (lower.endsWith('.css') || lower.endsWith('.scss')) return 'css';
  if (lower.endsWith('.json')) return { name: 'javascript', json: true };
  if (lower.endsWith('.md')) return 'markdown';
  if (lower.endsWith('.yaml') || lower.endsWith('.yml')) return 'yaml';
  if (lower.endsWith('.xml') || lower.endsWith('.svg')) return 'xml';
  if (lower.endsWith('.sql')) return 'sql';
  if (lower.endsWith('.ts') || lower.endsWith('.tsx')) return 'javascript';
  if (lower.endsWith('.js') || lower.endsWith('.jsx') || lower.endsWith('.mjs') || lower.endsWith('.cjs')) return 'javascript';
  return 'text/plain';
}

export function shortProjectTitle(project) {
  const prompt = String(project?.prompt || 'Untitled project').split('\n')[0].trim();
  return prompt.length > 34 ? `${prompt.slice(0, 31)}...` : prompt;
}

export function formatProjectTime(value) {
  if (!value) return 'No date';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
}

export function getCurrentRunPlan(project) {
  return project?.runPlan || project?.run_plan || { commands: [], html_file: null, html_path: null, serve_port: null };
}

function getProjectFileRecord(files, filepath) {
  const normalizedPath = normalizeProjectPath(filepath);
  return (files || []).find((entry) => entry.filepath === normalizedPath) || null;
}

function getPackageJsonPath(relDir) {
  return !relDir || relDir === '.' ? 'package.json' : `${relDir}/package.json`;
}

function getPackageData(files, relDir) {
  const packageFile = getProjectFileRecord(files, getPackageJsonPath(relDir));
  if (!packageFile) return null;
  try {
    return JSON.parse(packageFile.content);
  } catch {
    return null;
  }
}

function getNearestPackageRoot(files, filepath) {
  const normalizedPath = normalizeProjectPath(filepath);
  if (!normalizedPath) return null;

  const parts = normalizedPath.split('/').filter(Boolean);
  if (parts.length === 0) return null;
  if ((parts[parts.length - 1] || '').includes('.')) parts.pop();

  while (parts.length >= 0) {
    const relDir = parts.length > 0 ? parts.join('/') : '.';
    if (getProjectFileRecord(files, getPackageJsonPath(relDir))) {
      return relDir;
    }
    if (parts.length === 0) break;
    parts.pop();
  }

  return null;
}

function isFrontendPackageData(relDir, packageData) {
  if (!packageData || typeof packageData !== 'object') return false;
  const dirParts = new Set(String(relDir || '').split('/').filter(Boolean).map((part) => part.toLowerCase()));
  if ([...dirParts].some((part) => FRONTEND_DIR_HINTS.has(part))) return true;

  const deps = new Set([
    ...Object.keys(packageData.dependencies || {}),
    ...Object.keys(packageData.devDependencies || {})
  ]);
  if ([...deps].some((dep) => FRONTEND_DEP_HINTS.has(dep))) return true;

  const scripts = Object.values(packageData.scripts || {}).join(' ').toLowerCase();
  return FRONTEND_SCRIPT_HINTS.some((hint) => scripts.includes(hint));
}

function getRunPlanCommandForDirectory(project, relDir) {
  const commands = getCurrentRunPlan(project).commands || [];
  const prefix = relDir && relDir !== '.' ? `cd /d "${relDir.replace(/"/g, '""')}" && ` : '';

  return commands.find((command) => {
    const cmdText = String(command.cmd || '');
    if (!cmdText.includes('npm ')) return false;
    if (prefix) return cmdText.startsWith(prefix);
    return !cmdText.startsWith('cd /d "');
  }) || null;
}

function buildPackageRunCommand(project, relDir, packageData, label = 'Package') {
  const fromRunPlan = getRunPlanCommandForDirectory(project, relDir);
  if (fromRunPlan) return fromRunPlan;

  const scripts = packageData?.scripts || {};
  const prefix = relDir && relDir !== '.' ? `cd /d "${relDir.replace(/"/g, '""')}" && ` : '';

  for (const scriptName of ['start', 'dev', 'serve', 'preview']) {
    const scriptValue = scripts[scriptName];
    if (typeof scriptValue !== 'string' || !scriptValue.trim()) continue;
    const npmCommand = scriptName === 'start' ? 'npm start' : `npm run ${scriptName}`;
    return {
      name: relDir && relDir !== '.' ? `${label} (${relDir})` : label,
      cmd: `${prefix}${npmCommand}\r\n`
    };
  }

  return null;
}

export function buildFileRunCommand(project, files, filepath) {
  const normalizedPath = normalizeProjectPath(filepath);
  if (!normalizedPath) return null;

  const fileParts = normalizedPath.split('/');
  const filename = fileParts.pop();
  const dir = fileParts.length > 0 ? fileParts.join('/') : '.';
  const packageRoot = getNearestPackageRoot(files, normalizedPath);
  const packageData = packageRoot ? getPackageData(files, packageRoot) : null;
  const isFrontendPackage = packageData ? isFrontendPackageData(packageRoot, packageData) : false;
  let baseCmd = '';

  if (packageData && filename === 'package.json') {
    return buildPackageRunCommand(project, packageRoot, packageData, isFrontendPackage ? 'Frontend' : 'Package');
  }

  if (packageData && isFrontendPackage && (
    filename?.endsWith('.js')
    || filename?.endsWith('.jsx')
    || filename?.endsWith('.ts')
    || filename?.endsWith('.tsx')
    || filename?.endsWith('.html')
  )) {
    return buildPackageRunCommand(project, packageRoot, packageData, 'Frontend');
  }

  if (filename === 'package.json') {
    baseCmd = 'npm install && npm start';
  } else if (filename === 'requirements.txt') {
    baseCmd = 'pip install -r requirements.txt';
  } else if (filename?.endsWith('.py')) {
    baseCmd = `python ${filename}`;
  } else if (filename?.endsWith('.js')) {
    baseCmd = `node ${filename}`;
  } else if (filename?.endsWith('.html')) {
    baseCmd = 'python -m http.server 3000';
  } else {
    return null;
  }

  return {
    name: filename === 'package.json'
      ? 'Frontend / Package'
      : filename === 'requirements.txt'
        ? 'Python Dependencies'
        : `Run ${filename}`,
    cmd: dir !== '.'
      ? `cd /d "${dir.replace(/"/g, '""')}" && ${baseCmd}\r\n`
      : `${baseCmd}\r\n`
  };
}

export function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export function simpleMarkdownToHtml(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br />');
}
