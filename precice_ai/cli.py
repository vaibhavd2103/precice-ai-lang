def main():
    import argparse
    import os
    import sys
    import threading
    import time
    import webbrowser

    import uvicorn
    from dotenv import load_dotenv
    from precice_ai import config

    parser = argparse.ArgumentParser(description="preCICE AI local assistant")
    parser.add_argument("--host", default=os.environ.get("PRECICE_AI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PRECICE_AI_PORT", "7860")))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)

    if args.provider:
        os.environ["PRECICE_AI_PROVIDER"] = args.provider
    if args.api_key:
        os.environ["PRECICE_AI_API_KEY"] = args.api_key

    if args.model:
        os.environ["PRECICE_AI_MODEL"] = args.model
    if args.base_url:
        os.environ["PRECICE_AI_BASE_URL"] = args.base_url

    api_key = (
        os.environ.get("PRECICE_AI_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not api_key:
        print("ERROR: no LLM API key found.")
        print("Set one in `.env` or pass it with `--api-key`.")
        print("Accepted variables: PRECICE_AI_API_KEY, LLM_API_KEY, OPENROUTER_API_KEY")
        sys.exit(1)

    if not args.no_browser:
        def _open():
            time.sleep(2.5)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"preCICE AI starting at http://{args.host}:{args.port}")
    print(f"Provider: {os.environ.get('PRECICE_AI_PROVIDER', os.environ.get('LLM_PROVIDER', 'openrouter'))}")
    print(f"Model: {os.environ.get('PRECICE_AI_MODEL', os.environ.get('LLM_MODEL', config.DEFAULT_MODEL))}")
    print("Choose a working directory inside the UI for each chat session.")
    uvicorn.run("precice_ai.server:app", host=args.host, port=args.port, reload=False)
