# Setup guide

How to run AI Notebook on Windows, macOS and Linux, with or without a virtual environment.

You need **Python 3.10 or newer** and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
python3 --version      # Windows: py --version
```

If Python is missing, install it from [python.org/downloads](https://www.python.org/downloads/).
On Windows, tick **Add python.exe to PATH** during installation.

---

## Option A — With a virtual environment (recommended)

A virtual environment (`venv`) keeps this project's packages separate from the rest of your machine.

### 1. Create it (once)

| OS | Command |
|---|---|
| Windows | `py -m venv venv` |
| macOS / Linux | `python3 -m venv venv` |

### 2. Activate it (every new terminal)

| OS / shell | Command |
|---|---|
| Windows PowerShell | `.\venv\Scripts\Activate.ps1` |
| Windows CMD | `venv\Scripts\activate.bat` |
| macOS / Linux (bash, zsh) | `source venv/bin/activate` |
| Linux / macOS (fish) | `source venv/bin/activate.fish` |

You should see `(venv)` at the start of your prompt.

> **PowerShell blocks the script?** Run this once, then activate again:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3. Install the packages (once)

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add your API key (once)

| OS | Command |
|---|---|
| Windows (PowerShell / CMD) | `copy .env.example .env` |
| macOS / Linux | `cp .env.example .env` |

Open `.env` and replace `your_key_here` with your key:

```env
GEMINI_API_KEY=your_real_key
```

### 5. Run the app

```bash
streamlit run app.py
```

Streamlit opens http://localhost:8501. Stop it with `Ctrl+C`. Leave the venv with `deactivate`.

### Skip activation

You can call the venv's programs directly, in any shell:

| OS | Command |
|---|---|
| Windows | `venv\Scripts\streamlit run app.py` |
| macOS / Linux | `./venv/bin/streamlit run app.py` |

---

## Option B — Without a virtual environment

Packages are installed into your global Python. This is quick, but it can clash with other projects.

```bash
pip install -r requirements.txt
```

Then add your API key (step 4 above) and run:

```bash
streamlit run app.py
```

If `streamlit` is not found, use `python -m streamlit run app.py` (Windows: `py -m streamlit run app.py`).

### "externally-managed-environment" error (Arch, Debian/Ubuntu, recent macOS Homebrew)

Your system Python refuses global installs. Pick one:

- Use Option A (a venv), which is the simplest fix.
- Use [uv](https://docs.astral.sh/uv/), which builds a temporary environment for you:
  ```bash
  uv run --with-requirements requirements.txt streamlit run app.py
  ```
- Force a user install (can conflict with system packages):
  ```bash
  pip install --user --break-system-packages -r requirements.txt
  ```

---

## Try it

1. Upload the four files of `data/sample_docs/` (a linear-algebra course in French).
2. Ask: *"Quelles sont les valeurs propres de Q = [[2, 1], [1, 2]] ?"*

## Run the tests (no API key needed)

```bash
pytest
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is missing or still set to 'your_key_here'` | Edit `.env` and restart the app. |
| `streamlit: command not found` | Activate the venv, or use `python -m streamlit run app.py`. |
| `ModuleNotFoundError` | The packages are installed in another environment. Activate the venv and run `pip install -r requirements.txt` again. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Answers ignore a changed document | Remove the document and add it again, or delete the `data/chroma/` folder. |
| `429` / quota errors | The free tier has rate limits. Wait a minute and retry. Agentic chunking uses the most calls. |
