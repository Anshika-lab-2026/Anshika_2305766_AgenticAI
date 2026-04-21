# ⚛️ Study Buddy — Physics
### Agentic AI Capstone Project

| | |
|---|---|
| **Student** | Anshika Rai |
| **Roll No** | 2305766 |
| **Batch** | 2027_Agentic AI |

---

## 📌 Problem Statement
B.Tech students need concept help at odd hours. This AI-powered Physics tutor answers questions from the course syllabus faithfully — without hallucinating formulas or values.

## 🏗️ Architecture
```
User Question
     ↓
[memory_node]   → sliding window, extract user name
     ↓
[router_node]   → LLM decides: retrieve / tool / memory_only
     ↓
[retrieval_node / tool_node / skip_node]
     ↓
[answer_node]   → grounded answer from context
     ↓
[eval_node]     → faithfulness score → retry if < 0.7
     ↓
[save_node]     → append to history → END
```

## 🧠 Tech Stack
- **LLM:** Llama 3.3 70B via Groq
- **RAG:** ChromaDB + SentenceTransformer (all-MiniLM-L6-v2)
- **Orchestration:** LangGraph StateGraph
- **Memory:** MemorySaver with thread_id
- **UI:** Streamlit
- **Evaluation:** RAGAS (faithfulness, answer_relevancy, context_precision)

## 📚 Knowledge Base Topics (12 documents)
1. Newton's First Law  2. Newton's Second Law  3. Newton's Third Law
4. Gravitation  5. Work, Energy & Power  6. Waves & Sound
7. Light — Reflection & Refraction  8. Electricity & Ohm's Law
9. Thermodynamics  10. Atomic & Nuclear Physics
11. Kinematics  12. Magnetism & Electromagnetism

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key in .env
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Run Streamlit app
streamlit run capstone_streamlit.py

# 4. Or run the notebook
jupyter notebook day13_capstone.ipynb
```

## 📁 File Structure
```
Anshika_2305766_AgenticAI/
├── agent.py                 ← Core agent (KB, nodes, graph)
├── capstone_streamlit.py    ← Streamlit UI
├── day13_capstone.ipynb     ← Full notebook with all 8 parts
├── requirements.txt         ← Dependencies
├── .env                     ← API key (add your own)
└── README.md                ← This file
```
