def main():
    import argparse
    import os
    import sys
    import threading
    import time
    import webbrowser

    import uvicorn

    parser = argparse.ArgumentParser(description="preCICE AI local assistant")
    parser.add_argument("--host", default=os.environ.get("PRECICE_AI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PRECICE_AI_PORT", "7860")))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set.")
        print("Get a free key at https://openrouter.ai/ then:")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    if args.model:
        os.environ["PRECICE_AI_MODEL"] = args.model

    if not args.no_browser:
        def _open():
            time.sleep(2.5)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"preCICE AI starting at http://{args.host}:{args.port}")
    uvicorn.run("precice_ai.server:app", host=args.host, port=args.port, reload=False)
