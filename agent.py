"""
=============================================================
  Study Buddy — Physics
  Agentic AI Capstone Project
  Student  : Anshika Rai
  Roll No  : 2305766
  Batch    : 2027_Agentic AI
=============================================================
"""

import os
import math
import datetime
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
MODEL_NAME          = "llama-3.3-70b-versatile"
EMBED_MODEL         = "all-MiniLM-L6-v2"
FAITHFULNESS_THRESH = 0.7
MAX_EVAL_RETRIES    = 2
TOP_K               = 3

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.2)

# ── Embedder ─────────────────────────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)

# ── Knowledge Base ───────────────────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Newton's First Law of Motion",
        "text": (
            "Newton's First Law states that an object at rest stays at rest and an object "
            "in motion stays in motion with the same speed and in the same direction unless "
            "acted upon by an unbalanced force. This is also called the Law of Inertia. "
            "Inertia is the tendency of an object to resist changes in its state of motion. "
            "Heavier objects have more inertia. Example: a book lying on a table stays at "
            "rest; a rolling ball continues rolling until friction or another force stops it."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Newton's Second Law of Motion",
        "text": (
            "Newton's Second Law states that the acceleration of an object is directly "
            "proportional to the net force acting on it and inversely proportional to its "
            "mass. Formula: F = ma, where F is force (Newtons), m is mass (kg), and a is "
            "acceleration (m/s²). If the mass is constant, increasing force increases "
            "acceleration. Doubling the mass halves the acceleration for the same force. "
            "Example: pushing a shopping cart — more force means faster acceleration."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Newton's Third Law of Motion",
        "text": (
            "Newton's Third Law states that for every action there is an equal and opposite "
            "reaction. Forces always occur in pairs. When object A exerts a force on object "
            "B, object B exerts an equal and opposite force on A. Example: when a rocket "
            "expels gas downward, the gas pushes the rocket upward. When you walk, your "
            "foot pushes the ground backward and the ground pushes you forward. "
            "Action-reaction pairs act on different objects."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Gravitation and Law of Universal Gravitation",
        "text": (
            "Newton's Law of Universal Gravitation states every mass attracts every other "
            "mass. Formula: F = G(m1*m2)/r², where G = 6.674×10⁻¹¹ N·m²/kg², m1 and m2 "
            "are masses, r is distance between centres. Gravity is always attractive. "
            "Acceleration due to gravity on Earth's surface g = 9.8 m/s². Weight = mg. "
            "The Moon orbits Earth due to gravitational pull. Gravitational force decreases "
            "as distance increases — doubling distance reduces force by factor of 4."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Work, Energy and Power",
        "text": (
            "Work is done when a force causes displacement. W = F × d × cos(θ), where θ "
            "is angle between force and displacement. Unit: Joule (J). Kinetic Energy: "
            "KE = ½mv². Potential Energy (gravitational): PE = mgh. The Law of Conservation "
            "of Energy states energy cannot be created or destroyed, only transformed. "
            "Power is the rate of doing work: P = W/t, unit is Watt (W). "
            "Efficiency = (useful output energy / total input energy) × 100%."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Waves and Sound",
        "text": (
            "A wave transfers energy without transferring matter. Transverse waves: "
            "displacement perpendicular to direction of travel (e.g., light, water waves). "
            "Longitudinal waves: displacement parallel to direction of travel (e.g., sound). "
            "Key quantities: wavelength (λ), frequency (f), amplitude, wave speed v = fλ. "
            "Sound needs a medium — it cannot travel through vacuum. Speed of sound in air "
            "≈ 343 m/s at 20°C. Pitch depends on frequency; loudness depends on amplitude. "
            "Echoes are reflections of sound."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Light — Reflection and Refraction",
        "text": (
            "Reflection: angle of incidence equals angle of reflection. Laws of reflection "
            "hold for plane, concave, and convex mirrors. Refraction: bending of light as "
            "it passes from one medium to another due to change in speed. Snell's Law: "
            "n1 sin(θ1) = n2 sin(θ2). Refractive index n = c/v. Total internal reflection "
            "occurs when angle exceeds critical angle — used in optical fibres. Lenses: "
            "convex lens converges light (used in cameras, eyes); concave lens diverges "
            "light (used in spectacles for myopia). Lens formula: 1/f = 1/v − 1/u."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Electricity — Ohm's Law and Circuits",
        "text": (
            "Electric current I = Q/t (Amperes). Ohm's Law: V = IR, where V is voltage "
            "(Volts), I is current (Amperes), R is resistance (Ohms). Resistance depends "
            "on material, length, and cross-sectional area: R = ρL/A. Series circuit: "
            "total R = R1 + R2 + ... Current same throughout. Parallel circuit: "
            "1/R = 1/R1 + 1/R2 + ... Voltage same across branches. Power: P = VI = I²R = V²/R. "
            "Electrical energy E = Pt. Kilowatt-hour (kWh) is the commercial unit."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Thermodynamics and Heat Transfer",
        "text": (
            "Heat is energy transferred due to temperature difference. Three modes of heat "
            "transfer: Conduction (through solids — particles vibrate and pass energy), "
            "Convection (through fluids — hot fluid rises, cool fluid sinks), Radiation "
            "(electromagnetic waves — no medium needed, e.g., Sun's heat reaching Earth). "
            "Specific heat capacity c: Q = mcΔT. First Law of Thermodynamics: ΔU = Q − W "
            "(internal energy change equals heat added minus work done by system). "
            "Second Law: heat flows spontaneously from hot to cold body."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Atomic Structure and Nuclear Physics",
        "text": (
            "Atom consists of nucleus (protons + neutrons) and electrons in shells. "
            "Atomic number Z = number of protons. Mass number A = protons + neutrons. "
            "Isotopes: same Z, different A. Radioactivity: unstable nuclei emit radiation. "
            "Alpha (α) — helium nucleus, low penetration. Beta (β) — electron/positron, "
            "medium penetration. Gamma (γ) — EM radiation, high penetration. Half-life: "
            "time for half the nuclei to decay. Nuclear fission: heavy nucleus splits, "
            "releasing energy (nuclear reactors). Nuclear fusion: light nuclei combine — "
            "powers the Sun. E = mc² (Einstein's mass-energy equivalence)."
        ),
    },
    {
        "id": "doc_011",
        "topic": "Kinematics — Equations of Motion",
        "text": (
            "Kinematics studies motion without considering its cause. Key quantities: "
            "displacement (s), initial velocity (u), final velocity (v), acceleration (a), "
            "time (t). Equations of motion for uniform acceleration: "
            "v = u + at, s = ut + ½at², v² = u² + 2as. "
            "Projectile motion: horizontal velocity constant, vertical motion under gravity. "
            "Range R = u²sin(2θ)/g. Maximum height H = u²sin²(θ)/2g. "
            "Relative velocity: velocity of A relative to B = velocity of A − velocity of B."
        ),
    },
    {
        "id": "doc_012",
        "topic": "Magnetism and Electromagnetism",
        "text": (
            "Magnetic field exists around magnets and current-carrying conductors. "
            "Right-hand thumb rule: thumb points in current direction, fingers curl in "
            "direction of magnetic field. Magnetic force on a current-carrying wire: F = BIL. "
            "Force on a moving charge: F = qvB sin(θ). Electromagnetic induction: changing "
            "magnetic flux induces EMF (Faraday's Law). Lenz's Law: induced current opposes "
            "the change causing it. Transformers use mutual induction: Vs/Vp = Ns/Np. "
            "Step-up transformer increases voltage; step-down decreases it."
        ),
    },
]

# ── Build ChromaDB ────────────────────────────────────────────────────────────
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("physics_kb")

collection.add(
    documents=[d["text"] for d in DOCUMENTS],
    embeddings=[embedder.encode(d["text"]).tolist() for d in DOCUMENTS],
    ids=[d["id"] for d in DOCUMENTS],
    metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
)

# ── State ─────────────────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str


# ── Tools ─────────────────────────────────────────────────────────────────────
def physics_calculator(expression: str) -> str:
    """Safe evaluator for basic physics calculations."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def get_current_datetime() -> str:
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%A, %d %B %Y, %I:%M %p')}"


# ── Nodes ─────────────────────────────────────────────────────────────────────
def memory_node(state: CapstoneState) -> CapstoneState:
    question = state["question"]
    messages = state.get("messages", [])
    messages = messages[-6:]  # sliding window
    messages.append({"role": "user", "content": question})

    user_name = state.get("user_name", "")
    q_lower = question.lower()
    if "my name is" in q_lower:
        name = question.lower().split("my name is")[-1].strip().split()[0].capitalize()
        user_name = name

    return {**state, "messages": messages, "user_name": user_name}


def router_node(state: CapstoneState) -> CapstoneState:
    question = state["question"]
    history = state.get("messages", [])

    prompt = f"""You are a router for a Physics Study Buddy assistant.
Given the user's question, decide the best route. Reply with EXACTLY one word only.

Routes:
- retrieve   → question is about a physics concept, law, formula, or topic
- tool       → question requires calculation/arithmetic OR asks for current date/time
- memory_only → casual greeting, thanks, or question answerable from chat history alone

Conversation history (last 3 turns): {history[-3:]}
Question: {question}

Reply with ONE word: retrieve, tool, or memory_only"""

    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower().split()[0]
    if route not in ("retrieve", "tool", "memory_only"):
        route = "retrieve"
    return {**state, "route": route}


def retrieval_node(state: CapstoneState) -> CapstoneState:
    query_embedding = embedder.encode(state["question"]).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context = "\n\n".join(
        [f"[{m['topic']}]\n{d}" for d, m in zip(docs, metas)]
    )
    sources = [m["topic"] for m in metas]
    return {**state, "retrieved": context, "sources": sources}


def skip_node(state: CapstoneState) -> CapstoneState:
    return {**state, "retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> CapstoneState:
    question = state["question"].lower()
    if any(w in question for w in ["date", "time", "today", "day"]):
        result = get_current_datetime()
    else:
        # extract expression — look for digits/operators
        import re
        expr = re.sub(r"[^0-9+\-*/().^ a-z]", "", state["question"])
        expr = expr.strip()
        result = physics_calculator(expr) if expr else "Could not parse a calculation from the question."
    return {**state, "tool_result": result}


def answer_node(state: CapstoneState) -> CapstoneState:
    user_name = state.get("user_name", "")
    greeting = f"Hi {user_name}! " if user_name else ""
    retries = state.get("eval_retries", 0)

    context_section = ""
    if state.get("retrieved"):
        context_section = f"\n\nKNOWLEDGE BASE CONTEXT:\n{state['retrieved']}"
    if state.get("tool_result"):
        context_section += f"\n\nTOOL RESULT:\n{state['tool_result']}"

    strictness = ""
    if retries > 0:
        strictness = " Be very precise and strictly use ONLY the provided context."

    system = f"""You are Study Buddy, a friendly Physics tutor for B.Tech students.
{greeting}
Rules:
1. Answer ONLY from the provided context or tool result.
2. If the answer is not in the context, say: "I don't have information on that topic yet. Please ask your teacher or refer to your textbook."
3. Never fabricate formulas, values, or facts.
4. Use simple language. Provide step-by-step explanations where helpful.
5. For calculations, show each step clearly.{strictness}
{context_section}

Conversation history: {state.get('messages', [])}"""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["question"]),
    ])
    return {**state, "answer": response.content.strip()}


def eval_node(state: CapstoneState) -> CapstoneState:
    if not state.get("retrieved"):
        return {**state, "faithfulness": 1.0}

    prompt = f"""Rate how faithfully this answer is grounded in the provided context.
Score 0.0 (completely unfaithful) to 1.0 (perfectly faithful).
Reply with a number only.

Context: {state['retrieved'][:1000]}
Answer: {state['answer']}

Score:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        score = float(response.content.strip().split()[0])
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 1.0

    retries = state.get("eval_retries", 0) + 1
    return {**state, "faithfulness": score, "eval_retries": retries}


def save_node(state: CapstoneState) -> CapstoneState:
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": state["answer"]})
    return {**state, "messages": messages}


# ── Conditional edges ─────────────────────────────────────────────────────────
def route_decision(state: CapstoneState) -> str:
    r = state.get("route", "retrieve")
    if r == "tool":
        return "tool"
    if r == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    if (state.get("faithfulness", 1.0) < FAITHFULNESS_THRESH
            and state.get("eval_retries", 0) < MAX_EVAL_RETRIES):
        return "answer"
    return "save"


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(CapstoneState)

    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges("router", route_decision, {
        "retrieve": "retrieve",
        "skip": "skip",
        "tool": "tool",
    })
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges("eval", eval_decision, {
        "answer": "answer",
        "save": "save",
    })
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    print("✅ Graph compiled successfully")
    return app


app = build_graph()


# ── Helper ────────────────────────────────────────────────────────────────────
def ask(question: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: CapstoneState = {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "faithfulness": 1.0,
        "eval_retries": 0,
        "user_name": "",
    }
    result = app.invoke(initial_state, config=config)
    return result
