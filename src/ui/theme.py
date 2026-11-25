"""Dark theme for the Streamlit UI: chalk-grid blackboard, serif answers, highlighter sources."""

from __future__ import annotations

import streamlit as st

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
  --board: #0E1624;
  --panel: #152033;
  --card: #1A2740;
  --ink: #E6EBF2;
  --ink-soft: #B4C0D2;
  --rule: #26344D;
  --grid: #162238;
  --action: #7FB0FF;
  --highlighter: #FFD84D;
  --on-highlighter: #0E1624;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], label, button {
  font-family: 'Bricolage Grotesque', system-ui, sans-serif;
}

[data-testid="stAppViewContainer"] {
  background-color: var(--board);
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 28px 28px;
  background-position: -1px -1px;
}

[data-testid="stAppViewContainer"] > .main::before,
[data-testid="stMain"]::before {
  content: "";
  position: fixed;
  inset: 0;
  background: linear-gradient(180deg, rgba(14,22,36,0.80), rgba(14,22,36,0.92));
  pointer-events: none;
  z-index: 0;
}

[data-testid="stMainBlockContainer"] {
  position: relative;
  z-index: 1;
  max-width: 80rem;
  padding-top: 3rem;
}

h1, h2, h3, [data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
  font-family: 'Bricolage Grotesque', system-ui, sans-serif !important;
  letter-spacing: -0.02em;
  color: var(--ink);
}
h1 { font-weight: 700; font-size: 2.6rem; line-height: 1.05; }
h2, h3 { font-weight: 600; }

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
  color: var(--ink-soft) !important;
  opacity: 1 !important;
  font-size: 0.95rem;
}

[data-testid="stMarkdownContainer"] code,
[data-testid="stCaptionContainer"] code {
  background: var(--card) !important;
  color: #BFD6FF !important;
  border-radius: 3px;
}

[data-testid="stSidebar"] {
  background: var(--panel);
  border-right: 1px solid var(--rule);
}

[data-testid="stFileUploader"] section {
  background: var(--panel);
  border: 1.5px dashed var(--action);
  border-radius: 6px;
}

[data-testid="stChatMessage"] {
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 4px solid var(--rule);
  border-radius: 4px;
  padding: 1rem 1.25rem;
  box-shadow: none;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  border-left-color: var(--action);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] li {
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 1.05rem;
  line-height: 1.65;
  max-width: 68ch;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p {
  font-family: 'Source Serif 4', Georgia, serif !important;
}

[data-testid="stExpander"] {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 4px;
}
[class*="st-key-sources_"] button,
.st-key-starters button {
  border-radius: 3px;
  font-weight: 600;
}
[class*="st-key-sources_"] .stButton > button {
  background: var(--highlighter);
  color: var(--on-highlighter);
  border-color: var(--highlighter);
}
[class*="st-key-sources_"] .stButton > button:hover {
  background: #FFE98A;
  color: var(--on-highlighter);
  border-color: #FFE98A;
}
.st-key-chat_solo {
  max-width: 50rem;
  width: 100%;
}
.st-key-preview_pane {
  position: sticky;
  top: 4rem;
}

.stButton > button {
  border-radius: 4px;
  border: 1px solid var(--rule);
  background: var(--panel);
  color: var(--ink);
  font-weight: 600;
}
.stButton > button:hover {
  border-color: var(--action);
  color: var(--action);
}

[data-testid="stChatInput"] textarea:focus-visible {
  outline: none;
}

:focus-visible {
  outline: 3px solid var(--action);
  outline-offset: 2px;
}

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
