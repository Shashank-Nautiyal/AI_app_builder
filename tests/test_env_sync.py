from pathlib import Path

from env_sync import auto_fill_project_env


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_auto_fill_project_env_writes_frontend_and_backend_envs(tmp_path: Path) -> None:
    project = tmp_path / "generated"
    _write(
        project / "package.json",
        '{"dependencies":{"@supabase/supabase-js":"^2","@clerk/clerk-react":"^5","vite":"^5"}}',
    )
    _write(
        project / "src" / "lib" / "supabase.js",
        "createClient(import.meta.env.VITE_SUPABASE_URL, import.meta.env.VITE_SUPABASE_ANON_KEY)\n"
        "const clerkKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;\n",
    )
    _write(project / "backend" / "package.json", '{"dependencies":{"express":"^4"}}')
    _write(
        project / "backend" / "server.js",
        "process.env.SUPABASE_URL; process.env.SUPABASE_ANON_KEY; process.env.CLERK_SECRET_KEY;\n",
    )
    source_env = tmp_path / ".env"
    _write(
        source_env,
        "\n".join(
            [
                "SUPABASE_URL=https://demo.supabase.co",
                "SUPABASE_ANON_KEY=anon_live_123",
                "VITE_CLERK_PUBLISHABLE_KEY=pk_live_123",
                "CLERK_SECRET_KEY=sk_live_123",
            ]
        ),
    )

    result = auto_fill_project_env(project, source_env_path=source_env)

    assert ".env.local" in result.written_files
    assert "backend/.env" in result.written_files
    assert "backend/.env.local" in result.written_files

    frontend_env = (project / ".env.local").read_text(encoding="utf-8")
    assert "VITE_SUPABASE_URL=https://demo.supabase.co" in frontend_env
    assert "VITE_SUPABASE_ANON_KEY=anon_live_123" in frontend_env
    assert "VITE_CLERK_PUBLISHABLE_KEY=pk_live_123" in frontend_env
    assert "CLERK_SECRET_KEY" not in frontend_env

    backend_env = (project / "backend" / ".env").read_text(encoding="utf-8")
    assert "SUPABASE_URL=https://demo.supabase.co" in backend_env
    assert "SUPABASE_ANON_KEY=anon_live_123" in backend_env
    assert "CLERK_SECRET_KEY=sk_live_123" in backend_env
    assert ".env" in (project / ".gitignore").read_text(encoding="utf-8")


def test_auto_fill_project_env_preserves_existing_real_values(tmp_path: Path) -> None:
    project = tmp_path / "generated"
    _write(project / "backend" / "package.json", '{"dependencies":{"express":"^4"}}')
    _write(project / "backend" / "server.js", "process.env.SUPABASE_URL;\n")
    _write(project / "backend" / ".env", "SUPABASE_URL=https://existing.supabase.co\n")
    source_env = tmp_path / ".env"
    _write(source_env, "SUPABASE_URL=https://source.supabase.co\n")

    auto_fill_project_env(project, source_env_path=source_env)

    backend_env = (project / "backend" / ".env").read_text(encoding="utf-8")
    assert "SUPABASE_URL=https://existing.supabase.co" in backend_env


def test_auto_fill_project_env_skips_placeholder_source_values(tmp_path: Path) -> None:
    project = tmp_path / "generated"
    _write(project / "package.json", '{"dependencies":{"@supabase/supabase-js":"^2","vite":"^5"}}')
    _write(
        project / "src" / "supabase.js",
        "import.meta.env.VITE_SUPABASE_URL; import.meta.env.VITE_SUPABASE_ANON_KEY;\n",
    )
    source_env = tmp_path / ".env"
    _write(
        source_env,
        "SUPABASE_URL=https://your-project.supabase.co\nSUPABASE_ANON_KEY=your-anon-key\n",
    )

    result = auto_fill_project_env(project, source_env_path=source_env)

    assert result.written_files == []
    assert not (project / ".env.local").exists()
    assert "VITE_SUPABASE_URL" in result.missing_keys
    assert "VITE_SUPABASE_ANON_KEY" in result.missing_keys
