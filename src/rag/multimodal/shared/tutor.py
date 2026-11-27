"""Study Tutor: the persona, query rewriting, image questions, one conversational turn."""
from google.genai import types

from .config import MODEL, ask, client
from .rag import parts, retrieve

TUTOR = """You are Study Tutor, a patient tutor who teaches from ONE document.
You receive CONTEXT (text, tables, images) labelled [S1], [S2]...
1. Teach, do not just answer: give the answer in one sentence, then explain how to get it in 2-4 simple steps.
2. Point at the evidence: name the figure or table and what to look at (e.g. the orange bars in [S1]). Cite [S#] after every fact.
3. Use ONLY the context. If it is not there, say 'I could not find this in the document' and suggest where to look.
4. End with ONE short check question for the student, never more.
5. If the student answers a question you asked, say if it is right, correct it kindly, then ask the next one.
   If the student asks to be quizzed, ask ONE question about the document and wait.
6. If the student sends an image (USER IMAGE), relate it to the document.
7. Charts: say 'approximately' when you estimate. Be warm and brief."""


def describe(image: tuple[bytes, str]) -> str:
    """User picture -> short text, because the retriever only understands text."""
    return ask([types.Part.from_bytes(data=image[0], mime_type=image[1]),
                "Describe this image in at most 60 words as a search query: kind of visual, topic, visible labels and numbers."],
               temperature=0)


def rewrite(history: list[dict], question: str, image_desc: str = "") -> str:
    """'And for wind?' is not searchable: turn it into a standalone query using the history."""
    if not history and not image_desc:
        return question  # nothing to resolve: skip the API call
    chat = "\n".join(f"{m['role'].upper()}: {m['text']}" for m in history[-6:]) or "(empty)"
    return ask("Rewrite the student's last message as ONE standalone search query for a document search engine. "
               "Resolve pronouns using the conversation; include the content of the attached image if any. "
               f"Output only the query.\n\nConversation:\n{chat}\n\nAttached image: {image_desc or 'none'}\n\n"
               f"Last message: {question}\n\nQuery:", temperature=0) or question


def prepare(history, question, image=None, k=4, use_clip=False) -> dict:
    """Everything BEFORE the first token: understand the image, rewrite, retrieve, build the parts."""
    desc = describe(image) if image else ""
    query = rewrite(history, question, desc)
    ctx, sources = parts(retrieve(query, k, use_clip=use_clip))
    return {"query": query, "image_desc": desc, "parts": ctx, "sources": sources}


def stream(history, question, turn, image=None):
    """Yield the tutor's answer. Old turns go back as TEXT only: resending old images would multiply the cost."""
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part.from_text(text=m["text"])]) for m in history[-6:]]
    last = list(turn["parts"])
    if image:
        last += [types.Part.from_text(text="USER IMAGE:"), types.Part.from_bytes(data=image[0], mime_type=image[1])]
    contents.append(types.Content(role="user", parts=last + [types.Part.from_text(text=f"QUESTION: {question}")]))
    for c in client().models.generate_content_stream(
            model=MODEL, contents=contents, config=types.GenerateContentConfig(system_instruction=TUTOR, temperature=0.2)):
        yield c.text or ""
