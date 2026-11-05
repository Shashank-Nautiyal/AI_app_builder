import { useCallback, useEffect, useState } from 'react';
import { deleteProject, listProjects } from '../services/api.js';

export function useProjects() {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const refreshProjects = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await listProjects();
      setProjects(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err.message || 'Unable to load projects');
    } finally {
      setLoading(false);
    }
  }, []);

  const removeProject = useCallback(async (threadId) => {
    await deleteProject(threadId);
    await refreshProjects();
  }, [refreshProjects]);

  useEffect(() => {
    refreshProjects();
  }, [refreshProjects]);

  return {
    projects,
    loading,
    error,
    refreshProjects,
    removeProject
  };
}
