export const API_BASE_URL = window.location.protocol === 'file:' ? 'http://localhost:8000' : '';
export const WS_BASE_URL = window.location.protocol === 'file:'
  ? 'ws://localhost:8000'
  : window.location.protocol === 'https:'
    ? `wss://${window.location.host}`
    : `ws://${window.location.host}`;

export const AUTO_MODEL_FALLBACK_STORAGE_KEY = 'ai_studio_auto_model_fallback_on_limit';

async function readJson(response) {
  const contentType = response.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    return {};
  }
  return response.json();
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: options.body instanceof FormData
      ? options.headers
      : { 'Content-Type': 'application/json', ...(options.headers || {}) }
  });
  const data = await readJson(response);
  if (!response.ok || data?.ok === false) {
    const message = data?.error?.message || data?.detail || response.statusText || 'Request failed';
    throw new Error(message);
  }
  return data;
}

export function getAutoModelFallbackSetting() {
  try {
    const stored = window.localStorage.getItem(AUTO_MODEL_FALLBACK_STORAGE_KEY);
    return stored === null ? true : stored === 'true';
  } catch {
    return true;
  }
}

export function setAutoModelFallbackSetting(enabled) {
  try {
    window.localStorage.setItem(AUTO_MODEL_FALLBACK_STORAGE_KEY, String(Boolean(enabled)));
  } catch {
    // Storage can be disabled in private windows. The in-memory state still works.
  }
}

export function extractThreadId(data) {
  return data?.thread_id || data?.data?.thread_id || null;
}

export function listProjects() {
  return request('/api/projects');
}

export function getProject(threadId) {
  return request(`/api/projects/${encodeURIComponent(threadId)}?_=${Date.now()}`, {
    cache: 'no-store'
  });
}

export function deleteProject(threadId) {
  return request(`/api/projects/${encodeURIComponent(threadId)}`, {
    method: 'DELETE'
  });
}

export function generateProject(prompt, autoModelFallbackOnLimit = true) {
  return request('/api/generate', {
    method: 'POST',
    body: JSON.stringify({
      prompt,
      auto_model_fallback_on_limit: autoModelFallbackOnLimit
    })
  });
}

export function updateProject(threadId, prompt, autoModelFallbackOnLimit = true) {
  return request(`/api/projects/${encodeURIComponent(threadId)}/update`, {
    method: 'POST',
    body: JSON.stringify({
      prompt,
      auto_model_fallback_on_limit: autoModelFallbackOnLimit
    })
  });
}

export function resumeProject(threadId) {
  return request(`/api/projects/${encodeURIComponent(threadId)}/resume`, {
    method: 'POST'
  });
}

export function saveProjectFile(threadId, filepath, content) {
  return request(`/api/projects/${encodeURIComponent(threadId)}/file`, {
    method: 'POST',
    body: JSON.stringify({ filepath, content })
  });
}

export function runProject(threadId) {
  return request(`/api/projects/${encodeURIComponent(threadId)}/run`, {
    method: 'POST'
  });
}

export function getDownloadUrl(threadId) {
  return `${API_BASE_URL}/api/projects/${encodeURIComponent(threadId)}/download`;
}

export function uploadProjectFiles(files, projectName = '') {
  const fileList = Array.from(files || []);
  const formData = new FormData();

  if (fileList.length === 1 && fileList[0].name.toLowerCase().endsWith('.zip') && !fileList[0].webkitRelativePath) {
    formData.append('archive', fileList[0]);
  } else {
    fileList.forEach((file) => {
      formData.append('files', file, file.name);
      formData.append('relative_paths', file.webkitRelativePath || file.name);
    });
  }

  if (projectName.trim()) {
    formData.append('project_name', projectName.trim());
  }

  return request('/api/projects/upload', {
    method: 'POST',
    body: formData
  });
}

export async function streamProjectChat(threadId, message, handlers = {}) {
  const response = await fetch(`${API_BASE_URL}/api/projects/${encodeURIComponent(threadId)}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });

  if (!response.ok || !response.body) {
    const data = await readJson(response);
    throw new Error(data?.error?.message || data?.detail || 'Chat request failed');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split('\n\n');
    buffer = chunks.pop() || '';

    for (const chunk of chunks) {
      const line = chunk.split('\n').find((entry) => entry.startsWith('data:'));
      if (!line) continue;
      try {
        const event = JSON.parse(line.slice(5).trim());
        handlers.onEvent?.(event);
        if (event.type === 'done') {
          handlers.onDone?.();
        }
      } catch (error) {
        handlers.onEvent?.({ type: 'error', content: error.message });
      }
    }
  }
}
