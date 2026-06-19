import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.graph import app as app_graph

# 1. Get the raw Mermaid string
mermaid_code = app_graph.get_graph().draw_mermaid()

# 2. Inject custom CSS styling — colors match the step badges in streamlit_app.py
styled_mermaid = mermaid_code.replace(
    "graph TD",
    "graph TD\n"
    "classDef default fill:#f0f2f6,stroke:#4b4b4b,stroke-width:2px,rx:10,ry:10;\n"
    "classDef startend fill:#ff4b4b,stroke:#ff4b4b,color:white,font-weight:bold,rx:20,ry:20;\n"
    "classDef tool fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,rx:10,ry:10;\n"
    "classDef generate fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10;\n"
    "classDef judge fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,rx:10,ry:10;\n"
    "classDef rewrite fill:#fff8e1,stroke:#e65100,stroke-width:2px,rx:10,ry:10;\n"
    "classDef hitl fill:#ffebee,stroke:#c62828,stroke-width:2px,rx:10,ry:10;\n"
    "class __start__,__end__ startend;\n"
    "class vector_search,web_search,github_read tool;\n"
    "class generator generate;\n"
    "class judge judge;\n"
    "class rewrite rewrite;\n"
    "class hitl hitl;\n"
)

# 3. Save styled Mermaid text (render in Streamlit or mermaid.live)
with open("graph_styled.txt", "w") as f:
    f.write(styled_mermaid)
print("Styled Mermaid code saved to graph_styled.txt")
print()
print("Paste the contents into https://mermaid.live to preview.")
print()
print(styled_mermaid)

# 4. Try to generate PNG via the remote mermaid.ink API.
# Note: LangChain's local-render fallback (MermaidDrawMethod.PYPPETEER) depends on the
# unmaintained `pyppeteer` package, whose pinned deps (old websockets/urllib3) conflict
# with langgraph-sdk and streamlit in this project — not worth the instability.
# If this fails (e.g. corporate TLS proxy blocking mermaid.ink), just use graph_styled.txt
# with https://mermaid.live instead.
try:
    png_data = app_graph.get_graph().draw_mermaid_png()
    with open("langgraph_flow.png", "wb") as f:
        f.write(png_data)
    print("PNG saved to langgraph_flow.png")
except Exception as e:
    print(f"PNG generation skipped: {e}")
    print("Use graph_styled.txt with https://mermaid.live instead.")
