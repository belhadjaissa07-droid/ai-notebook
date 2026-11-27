"""STEP 7 - THE TUTOR CONVERSATION: follow-ups, image questions, quizzes.

  /image <path> <question>   the student attaches a picture: VLM describes it -> joins the search query
  /quiz [topic]              the tutor asks ONE question and then grades the answer (a prompt rule, no extra code)
  /clear  /quit
Per turn: describe image -> rewrite 'And for wind?' into a full query -> retrieve -> stream the answer.
Code: shared/tutor.py (prepare, stream). The Streamlit page calls the same two functions.

Run: python src/rag/multimodal/7_mm_history_aware_generation.py [--clip]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # ponytail: see 1_parse_pdf.py
from src.rag.multimodal.shared.tutor import prepare, stream

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 7: Study Tutor chat")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--clip", action="store_true")
    args = ap.parse_args()
    history = []
    print("Study Tutor.  /image <path> <question> | /quiz [topic] | /clear | /quit\n")
    while (raw := input("You> ").strip()) != "/quit":
        if raw == "/clear":
            history.clear()
        if not raw or raw == "/clear":
            continue
        image, q = None, raw
        if raw.startswith("/image "):
            path, _, q = raw[7:].partition(" ")
            p = Path(path).expanduser()
            if not p.exists():
                print(f"(image not found: {p})\n")
                continue
            image, q = (p.read_bytes(), MIME.get(p.suffix.lower(), "image/png")), q.strip() or "What does this image show, and how does it relate to the document?"
        elif raw.startswith("/quiz"):
            q = f"Quiz me on {raw[5:].strip() or 'the key facts in this document'}. Ask one question."
        turn = prepare(history, q, image, args.k, args.clip)
        sources = ", ".join("[%s] %s p%s" % (s["label"], s["type"], s["page"]) for s in turn["sources"])
        if turn["image_desc"]:
            print(f"  (image understood as: {turn['image_desc']})")
        print(f"  (search: {turn['query']})\n  (sources: {sources})\n\nTutor> ", end="")
        answer = ""
        for piece in stream(history, q, turn, image):
            answer += piece
            print(piece, end="", flush=True)
        print("\n")
        history += [{"role": "user", "text": q}, {"role": "assistant", "text": answer}]


if __name__ == "__main__":
    main()
