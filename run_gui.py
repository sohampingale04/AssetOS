import subprocess
import webbrowser
import time
import os
import sys

def main():
    print("🚀 Initializing AssetOS Elite System...")

    # 1. Start Backend (Flask)
    print("   [1/3] Starting Core Engine (Backend)...")
    backend_process = subprocess.Popen(
        [sys.executable, "backend/app.py"],
        stdout=subprocess.PIPE,
        stderr=None, # Allow stderr to show in terminal
        text=True
    )

    # 2. Start Frontend (React)
    print("   [2/3] launching User Interface...")
    # We assume 'npm run dev' is already built or running, or we run it?
    # Actually, better to just run 'npm run dev' in a separate terminal or here.
    # For a Python script execution, we can try to run it.
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("   [3/3] Waiting for services...")
    time.sleep(5) # Give it a moment to spin up

    # 3. Open Browser
    url = "http://localhost:5173" # Vite default
    print(f"   ✅ System Online. Opening {url}")
    webbrowser.open(url)

    try:
        # Keep alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down AssetOS...")
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()
