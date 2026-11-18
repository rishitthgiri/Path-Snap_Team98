"""Serve the evaluator UI (runs on Laptop 2).

Usage: python ui/serve_ui.py --port 8501
Then open http://localhost:8501/?api=http://<LAPTOP1_IP>:8000
"""
import argparse, http.server, functools, os
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8501)
a = ap.parse_args()
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).parent))
print(f"PathSnap UI on http://0.0.0.0:{a.port}/  (add ?api=http://<LAPTOP1_IP>:8000)")
http.server.ThreadingHTTPServer(("0.0.0.0", a.port), handler).serve_forever()
