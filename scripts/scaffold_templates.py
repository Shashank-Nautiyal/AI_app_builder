import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codemind.paths import TEMPLATES_DIR


def main():
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    os.chdir(TEMPLATES_DIR)
    
    print("Scaffolding react-vite...")
    subprocess.run("npx -y create-vite@latest react-vite --template react", shell=True)
    
    print("Scaffolding nextjs-app...")
    subprocess.run("npx -y create-next-app@latest nextjs-app --javascript --tailwind --eslint --app --src-dir --import-alias \"@/*\" --use-npm", shell=True)
    
    print("Scaffolding fastapi...")
    os.makedirs("fastapi", exist_ok=True)
    with open("fastapi/requirements.txt", "w") as f:
        f.write("fastapi>=0.110.0\nuvicorn>=0.28.0\npydantic>=2.6.0\n")
    with open("fastapi/main.py", "w") as f:
        f.write("from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\ndef read_root():\n    return {'Hello': 'World'}\n")
    
    print("Templates created.")

if __name__ == "__main__":
    main()
