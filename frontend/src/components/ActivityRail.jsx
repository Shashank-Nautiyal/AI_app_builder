import {
  CircleHelp,
  Clock3,
  Folder,
  Play,
  Rocket,
  Settings
} from 'lucide-react';

const items = [
  { id: 'history', label: 'History', icon: Clock3 },
  { id: 'files', label: 'Files', icon: Folder },
  { id: 'run', label: 'Run', icon: Play },
  { id: 'deploy', label: 'Deploy', icon: Rocket },
  { id: 'settings', label: 'Settings', icon: Settings },
  { id: 'help', label: 'Help', icon: CircleHelp }
];

export function ActivityRail({ activePanel, onSelect }) {
  return (
    <nav className="activity-rail" aria-label="Workspace navigation">
      <div className="activity-stack">
        {items.slice(0, 4).map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={`activity-button ${activePanel === item.id ? 'active' : ''}`}
              type="button"
              onClick={() => onSelect(item.id)}
              title={item.label}
            >
              <Icon size={22} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
      <div className="activity-stack">
        {items.slice(4).map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={`activity-button ${activePanel === item.id ? 'active' : ''}`}
              type="button"
              onClick={() => onSelect(item.id)}
              title={item.label}
            >
              <Icon size={22} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
