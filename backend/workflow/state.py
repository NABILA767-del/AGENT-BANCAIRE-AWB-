"""
state.py — État LangGraph AWB
Carnet de bord partagé entre tous les nodes du workflow.
"""

from typing import TypedDict, List, Dict, Optional
from datetime import datetime


class AgentState(TypedDict):
    # ── Identification session ────────────────────────────────────────────
    session_id:   str
    user_id:      str
    timestamp:    datetime

    # ── Message ───────────────────────────────────────────────────────────
    message:               str    # message courant (peut être modifié)
    original_message:      str    # message d'origine non modifié
    pseudonymized_message: str    # version masquant les PII

    # ── Modération ────────────────────────────────────────────────────────
    toxicity_score: float         # 0.0 → 1.0
    is_toxic:       bool

    # ── Routing ───────────────────────────────────────────────────────────
    intent:       str             # credit | info | fraud | complaint | ...
    target_agent: str             # AG04, AG06, AG09, ...
    confidence:   float           # confiance du routeur (0.0 → 1.0)

    # ── RAG ───────────────────────────────────────────────────────────────
    rag_results:       List[Dict] # documents ChromaDB récupérés
    compressed_context: str       # contexte compressé injecté dans le LLM

    # ── Réponse agent ─────────────────────────────────────────────────────
    agent_response:  str
    confidence_score: float       # confiance de la réponse (0.0 → 1.0)

    # ── HITL ──────────────────────────────────────────────────────────────
    requires_hitl: bool
    hitl_reason:   str            # explication lisible (ex: "Montant > 100k€")

    # ── Sécurité & audit ──────────────────────────────────────────────────
    signatures:         List[str] # signatures Ed25519 des checkpoints
    worm_logs:          List[Dict] # entrées WORM de la session
    rgpd_check_passed:  bool
    pii_detected:       List[str] # liste des PII masqués

    # ── Métadonnées de traçabilité ────────────────────────────────────────
    processing_time: float        # durée totale en secondes
    nodes_visited:   List[str]    # liste ordonnée des nodes traversés
    user_context: Dict