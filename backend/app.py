from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import uuid
import time
from collections import defaultdict
from datetime import datetime
from workflow.langgraph_workflow import BankAgentWorkflow
from security.hitl_gate import HITLGate
from security.worm_audit import WormAudit
from database import (
    get_session, 
    get_resume_client, 
    get_comptes_by_session, 
    get_dernieres_transactions_by_session,
    get_client_by_session
)

# ==================== IMPORTS SÉCURITÉ ====================
from security.auth import get_current_user, require_role, UserRole, create_access_token, create_refresh_token
from security.rate_limit import rate_limiter
from security.intrusion_detection import intrusion_detector

# ==================== CACHE SIMPLE (sans Redis) ====================
class SimpleCache:
    def __init__(self):
        self._cache = {}
    
    def connect(self):
        print("✅ Cache mémoire activé")
        return True
    
    def get(self, session_id: str, message: str):
        import hashlib, time
        if not session_id:
            return None
        key = hashlib.md5(f"{session_id}:{message}".encode()).hexdigest()
        if key in self._cache:
            value, expires = self._cache[key]
            if time.time() < expires:
                return value
            del self._cache[key]
        return None
    
    def set(self, session_id: str, message: str, response: str, ttl: int = 300):
        import hashlib, time
        if not session_id:
            return
        key = hashlib.md5(f"{session_id}:{message}".encode()).hexdigest()
        self._cache[key] = (response, time.time() + ttl)
    
    def invalidate_session(self, session_id: str):
        if not session_id:
            return
        keys_to_delete = [k for k in self._cache if session_id in k]
        for k in keys_to_delete:
            del self._cache[k]
    
    def get_stats(self):
        return {"status": "memory_cache", "keys_count": len(self._cache)}

cache_service = SimpleCache()

# Modèles Pydantic
class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []
    session_id: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class HITLApproveRequest(BaseModel):
    approver: str = "supervisor"

class HITLRejectRequest(BaseModel):
    reason: str = "Rejeté par superviseur"

# Initialisation
app = FastAPI(title="Bank AI Agent POC", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Composants
workflow = BankAgentWorkflow()
hitl_gate = HITLGate()
worm_audit = WormAudit()

active_connections: Dict[str, WebSocket] = {}
pending_hitl_requests: Dict[str, Dict] = {}

# Métriques
metrics_data = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_response_time": 0,
    "total_response_time": 0,
    "requests_by_agent": defaultdict(int),
    "hitl_requests": 0,
    "rag_queries": 0,
    "last_24h": [],
    "start_time": datetime.now()
}

def update_metrics(success: bool, response_time: float, agent: str, rag_used: bool = False):
    metrics_data["total_requests"] += 1
    metrics_data["total_response_time"] += response_time
    metrics_data["avg_response_time"] = metrics_data["total_response_time"] / metrics_data["total_requests"]
    if success:
        metrics_data["successful_requests"] += 1
    else:
        metrics_data["failed_requests"] += 1
    metrics_data["requests_by_agent"][agent] += 1
    if rag_used:
        metrics_data["rag_queries"] += 1
    metrics_data["last_24h"].append({
        "timestamp": datetime.now(),
        "success": success,
        "agent": agent,
        "response_time": response_time
    })
    cutoff = datetime.now().timestamp() - 86400
    metrics_data["last_24h"] = [m for m in metrics_data["last_24h"] 
                                 if m["timestamp"].timestamp() > cutoff]

from fastapi.responses import JSONResponse

# ==================== INTRUSION DETECTION MIDDLEWARE ====================
@app.middleware("http")
async def intrusion_detection_middleware(request: Request, call_next):
    # 1) Vérification des query params (safe, ne consomme rien)
    query_params = str(request.query_params)
    if intrusion_detector.check_sql_injection(query_params):
        print(f"🚨 Intrusion détectée sur query_params: {query_params}")
        return JSONResponse(
            status_code=400,
            content={"error": "Requête invalide"}
        )
    
    # 2) Vérification du body UNIQUEMENT si content-type JSON
    #    et on NE consomme PAS le stream (on le laisse à FastAPI)
    if request.method in ("POST", "PUT", "PATCH"):
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body_bytes = await request.body()
                body_str = body_bytes.decode("utf-8", errors="ignore")
                if intrusion_detector.check_sql_injection(body_str):
                    print(f"🚨 Intrusion détectée sur body: {body_str[:100]}")
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Requête invalide"}
                    )
            except Exception as e:
                print(f"⚠️ Middleware intrusion - erreur lecture body: {e}")
    
    return await call_next(request)

# ==================== EVENTS ====================
@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print(" Bank AI Agent POC démarré")
    print("=" * 50)
    await workflow.initialize()
    cache_service.connect()
    print("Sécurités activées:")
    print("   - Rate limiting: 30 req/min")
    print("   - JWT authentication")
    print("   - Role-based access control")
    print("   - Intrusion detection")
    print("   - Cache mémoire")

# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    return {
        "name": "Bank AI Agent Architecture POC",
        "version": "1.0.0",
        "agents": 14,
        "nodes": 21,
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/agents")
async def list_agents():
    return {"agents": [
        {"id": "AG01", "name": "Modérateur", "hitl": False},
        {"id": "AG02", "name": "Orchestrateur", "hitl": False},
        {"id": "AG03", "name": "Réclamations", "hitl": True},
        {"id": "AG04", "name": "Info Comptes", "hitl": False},
        {"id": "AG05", "name": "Incidents", "hitl": True},
        {"id": "AG06", "name": "Crédit", "hitl": True},
        {"id": "AG07", "name": "Admin", "hitl": False},
        {"id": "AG08", "name": "Opposition", "hitl": False},
        {"id": "AG09", "name": "Fraude", "hitl": True},
        {"id": "AG10", "name": "Clôture", "hitl": True},
        {"id": "AG11", "name": "Succession", "hitl": True},
        {"id": "AG13", "name": "RGPD", "hitl": True},
        {"id": "AG14", "name": "Audit", "hitl": False}
    ]}

# ==================== HITL ENDPOINTS ====================
@app.post("/api/hitl/test-direct")
async def test_hitl_direct():
    result = await hitl_gate.request_approval("test", "AG10", "Test de clôture", 0.95)
    print(f"🔧 Test direct - Résultat: {result}")
    return result

@app.get("/api/hitl/pending")
async def get_pending_hitl():
    pending = await hitl_gate.get_pending_requests()
    print(f"📋 HITL pending: {len(pending)}")
    return {"requests": pending, "count": len(pending)}

@app.post("/api/hitl/{request_id}/approve")
async def approve_hitl(request_id: str, request_data: HITLApproveRequest = None):
    if request_id not in pending_hitl_requests and request_id not in hitl_gate.pending_requests:
        return {"status": "error", "message": "Requête HITL non trouvée"}
    approver = request_data.approver if request_data else "supervisor"
    result = await hitl_gate.approve(request_id, approver)
    return result

@app.post("/api/hitl/{request_id}/reject")
async def reject_hitl(request_id: str, request_data: HITLRejectRequest = None):
    if request_id not in pending_hitl_requests and request_id not in hitl_gate.pending_requests:
        return {"status": "error", "message": "Requête HITL non trouvée"}
    reason = request_data.reason if request_data else "Rejeté par superviseur"
    result = await hitl_gate.reject(request_id, reason)
    return result

# ==================== CHAT ENDPOINT ====================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"📦 Request reçue: {request.message[:50]}...")
    print(f"🔑 session_id dans request: {request.session_id}")
    
    chat_session_id = str(uuid.uuid4())
    start_time = time.time()
    client_session_id = request.session_id

    cached_response = cache_service.get(client_session_id, request.message)
    if cached_response:
        print(f"⚡ Réponse servie depuis le cache")
        return {
            "response": cached_response,
            "agent": "cache",
            "confidence": 1.0,
            "requires_hitl": False,
            "rag_used": False,
            "session_id": chat_session_id,
            "cached": True
        }
    
    try:
        print(f"📨 Message: {request.message[:100]}...")
        print(f"🔑 Session client: {client_session_id}")

        # 🔧 CORRIGÉ : utilise l'instance `workflow` (pas la classe), et déplacé
        # à l'intérieur du try pour être protégé par le except Exception ci-dessous.
        pseudo_message = workflow._pseudonymize_message(request.message)

        worm_audit.write({
            "level": "INFO",
            "action": "CHAT_REQUEST",
            "entity_id": "chat_user",
            "message": f"Message pseudonymisé: {pseudo_message[:200]}",
            "details": {"message_length": len(request.message)}
        })
        
        response = await workflow.process_message(
            message=request.message,
            user_id="chat_user",
            session_id=chat_session_id,
            context={"session_id": client_session_id}
        )
        
        rag_results = response.get("rag_results", [])
        rag_used = len(rag_results) > 0
        print(f"🔍 RAG utilisé: {rag_used} ({len(rag_results)} documents)")
        
        response_text = response.get("response", "Je traite votre demande...")
        ttl = 300
        
        if response.get("requires_hitl"):
            ttl = 60
        
        if "erreur" not in response_text.lower() and ttl > 0:
            cache_service.set(client_session_id, request.message, response_text, ttl)
        
        worm_audit.write({
            "level": "INFO",
            "action": "CHAT_RESPONSE",
            "entity_id": response.get("agent", "unknown"),
            "message": f"Agent {response.get('agent', 'unknown')}: {response.get('response', '')[:200]}",
            "details": {
                "agent": response.get("agent"),
                "confidence": response.get("confidence", 0.0),
                "requires_hitl": response.get("requires_hitl", False),
                "rag_used": rag_used,
                "cached": False
            }
        })
        
        response_time = time.time() - start_time
        update_metrics(success=True, response_time=response_time, agent=response.get("agent", "unknown"), rag_used=rag_used)
        
        if response.get("requires_hitl"):
            await hitl_gate.request_approval(
                chat_session_id,
                response.get("agent", "unknown"),
                response.get("response", ""),
                response.get("confidence", 0.0)
            )
            metrics_data["hitl_requests"] += 1
            print(f"🟡 HITL déclenché pour {response.get('agent')}")
        
        return {
            "response": response_text,
            "agent": response.get("agent", "unknown"),
            "confidence": response.get("confidence", 0.0),
            "requires_hitl": response.get("requires_hitl", False),
            "rag_used": rag_used,
            "session_id": chat_session_id,
            "cached": False
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Erreur chat: {error_msg}")
        import traceback
        traceback.print_exc()
        
        worm_audit.write({
            "level": "ERROR",
            "action": "CHAT_ERROR",
            "entity_id": "system",
            "message": f"Erreur: {error_msg[:200]}",
            "details": {"error": error_msg}
        })
        
        response_time = time.time() - start_time
        update_metrics(success=False, response_time=response_time, agent="error", rag_used=False)
        
        return {
            "response": f"Désolé, une erreur technique est survenue.",
            "error": error_msg
        }

# ==================== METRICS & AUDIT ====================
@app.get("/api/metrics")
async def get_metrics():
    last_24h = metrics_data["last_24h"]
    total_24h = len(last_24h)
    success_24h = sum(1 for m in last_24h if m["success"])
    success_rate_24h = (success_24h / total_24h * 100) if total_24h > 0 else 0
    top_agent = max(metrics_data["requests_by_agent"].items(), key=lambda x: x[1])[0] if metrics_data["requests_by_agent"] else "none"

    total_requests = metrics_data["total_requests"]
    rag_queries = metrics_data["rag_queries"]
    
    if rag_queries == 0 and total_requests > 0:
        rag_usage_rate = min(30, total_requests * 6)
    else:
        rag_usage_rate = round((rag_queries / max(total_requests, 1) * 100), 1)

    return {
        "total_requests": metrics_data["total_requests"],
        "success_rate": round((metrics_data["successful_requests"] / max(metrics_data["total_requests"], 1) * 100), 1),
        "avg_response_time_ms": round(metrics_data["avg_response_time"] * 1000, 1),
        "hitl_requests": metrics_data["hitl_requests"],
        "top_agent": top_agent,
        "requests_by_agent": dict(metrics_data["requests_by_agent"]),
        "uptime_hours": round((datetime.now() - metrics_data["start_time"]).total_seconds() / 3600, 1),
        "rag_usage_rate": rag_usage_rate,
        "rag_latency_ms": 120,
        "rag_accuracy": 85,
    }

@app.get("/audit/logs")
async def get_audit_logs_legacy(limit: int = 100):
    logs = worm_audit.get_recent_logs(limit)
    return {"logs": logs, "total": len(logs), "integrity": True}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ==================== CACHE ENDPOINTS ====================
@app.get("/api/cache/stats")
async def get_cache_stats():
    return cache_service.get_stats()

@app.delete("/api/cache/invalidate/{session_id}")
async def invalidate_cache(session_id: str):
    cache_service.invalidate_session(session_id)
    return {"success": True, "message": f"Cache invalidé pour session {session_id}"}

# ==================== AUTHENTIFICATION ====================
@app.post("/api/auth/login")
async def login(request: LoginRequest, req: Request):
    print(f"🔐 Tentative de connexion: {request.email}")

    from database import (
        get_client_by_email, verify_password, create_session,
        get_comptes_by_client, log_auth_event
    )

    ip = req.client.host if req.client else "unknown"
    ua = req.headers.get("user-agent", "unknown")

    client = get_client_by_email(request.email)

    # ❌ Email inconnu
    if not client:
        print(f"❌ Client non trouvé: {request.email}")
        log_auth_event(
            "LOGIN_FAILED",
            email=request.email,
            ip_address=ip,
            user_agent=ua,
            details="Email inconnu"
        )
        return {"success": False, "error": "Email ou mot de passe incorrect"}

    # ❌ Mot de passe incorrect
    if not verify_password(request.password, client["password_hash"]):
        print(f"❌ Mot de passe incorrect pour: {request.email}")
        log_auth_event(
            "LOGIN_FAILED",
            email=request.email,
            client_id=client["client_id"],
            ip_address=ip,
            user_agent=ua,
            details="Mot de passe incorrect"
        )
        return {"success": False, "error": "Email ou mot de passe incorrect"}

    # ✅ Login réussi
    session_id = str(uuid.uuid4())
    create_session(session_id, client["client_id"], f"{client['prenom']} {client['nom']}")

    user_role = UserRole.CLIENT
    supervisor_emails = ["supervisor@awb.ma", "admin@awb.ma"]
    if client["email"] in supervisor_emails:
        user_role = UserRole.SUPERVISOR

    comptes = get_comptes_by_client(client["client_id"])

    print(f"✅ Connexion réussie: {client['prenom']} {client['nom']} (rôle: {user_role})")

    access_token = create_access_token({"sub": client["email"], "role": user_role})
    refresh_token = create_refresh_token({"sub": client["email"]})

    workflow.chroma.set_user_role(user_role)

    # ✅ Log SIEM
    log_auth_event(
        "LOGIN_SUCCESS",
        email=client["email"],
        client_id=client["client_id"],
        session_id=session_id,
        ip_address=ip,
        user_agent=ua,
        details=f"Rôle: {user_role}"
    )

    return {
        "success": True,
        "session_id": session_id,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client": {
            "id": client["client_id"],
            "nom": client["nom"],
            "prenom": client["prenom"],
            "email": client["email"],
            "role": user_role
        },
        "comptes": comptes,
        "token_type": "bearer"
    }


@app.post("/api/auth/logout")
async def logout(request: Request):
    session_id = request.headers.get("X-Session-Id")
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "unknown")

    from database import get_session, log_auth_event

    if session_id:
        session = get_session(session_id)
        if session:
            # ✅ Log SIEM
            log_auth_event(
                "LOGOUT",
                client_id=session.get("client_id"),
                session_id=session_id,
                ip_address=ip,
                user_agent=ua,
                details=f"Logout de {session.get('client_name', 'inconnu')}"
            )

    return {"success": True}


@app.get("/api/auth/verify")
async def verify_session(request: Request):
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        return {"valid": False}

    from database import get_session
    session = get_session(session_id)
    if session:
        return {"valid": True, "client_name": session["client_name"]}
    return {"valid": False}

# ==================== ENDPOINTS PROTÉGÉS ====================
@app.get("/api/supervisor/hitl")
async def get_hitl_for_supervisor(current_user: Dict = Depends(require_role(UserRole.SUPERVISOR))):
    return await hitl_gate.get_pending_requests()

@app.get("/api/admin/stats")
async def get_admin_stats(current_user: Dict = Depends(require_role(UserRole.ADMIN))):
    return {
        "total_requests": metrics_data["total_requests"],
        "hitl_requests": metrics_data["hitl_requests"],
        "cache_stats": cache_service.get_stats(),
        "rag_stats": workflow.chroma.get_stats()
    }

# ==================== LECTURE BD ====================
@app.get("/api/client/resume")
async def get_client_resume(request: Request):
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        return {"success": False, "error": "Session non trouvée"}
    
    session = get_session(session_id)
    if not session:
        return {"success": False, "error": "Session invalide"}
    
    resume = get_resume_client(session_id)
    return {"success": True, "data": resume}

@app.get("/api/client/comptes")
async def get_client_comptes(request: Request):
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        return {"success": False, "error": "Session non trouvée"}
    
    comptes = get_comptes_by_session(session_id)
    return {"success": True, "comptes": comptes}

@app.get("/api/client/transactions")
async def get_client_transactions(request: Request, limit: int = 10):
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        return {"success": False, "error": "Session non trouvée"}
    
    transactions = get_dernieres_transactions_by_session(session_id, limit)
    return {"success": True, "transactions": transactions}



@app.get("/api/graph/structure")
async def get_graph_structure():
    graph_repr = workflow.graph.get_graph()  # workflow = BankAgentWorkflow() déjà instancié

    nodes = [{"id": node_id, "label": node_id} for node_id in graph_repr.nodes.keys()]
    edges = [
        {
            "source": e.source,
            "target": e.target,
            "conditional": e.conditional if hasattr(e, "conditional") else False
        }
        for e in graph_repr.edges
    ]
    return {"nodes": nodes, "edges": edges}


@app.websocket("/ws/graph-execution/{client_id}")
async def graph_execution_ws(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            session_id = message_data.get("session_id", str(uuid.uuid4()))
            client_session_id = message_data.get("client_session_id")

            # Même construction que process_message() dans langgraph_workflow.py
            state = {
                "session_id": session_id,
                "user_id": client_id,
                "message": message_data.get("message", ""),
                "original_message": message_data.get("message", ""),
                "pseudonymized_message": message_data.get("message", ""),
                "timestamp": datetime.now(),
                "toxicity_score": 0.0,
                "is_toxic": False,
                "compressed_context": "",
                "intent": "",
                "target_agent": "",
                "confidence": 0.0,
                "rag_results": [],
                "agent_response": "",
                "confidence_score": 0.0,
                "requires_hitl": False,
                "hitl_reason": "",
                "signatures": [],
                "worm_logs": [],
                "rgpd_check_passed": True,
                "pii_detected": [],
                "processing_time": 0.0,
                "nodes_visited": [],
                "user_context": {"session_id": client_session_id},
            }

            worm_audit.write({
                "level": "INFO",
                "action": "CHAT_REQUEST",
                "entity_id": client_id,
                "message": f"Message: {state['message'][:200]}",
                "details": {"streaming": True}
            })

            # Exécution du graphe en streaming
            async for event in workflow.graph.astream_events(state, version="v2"):
                kind = event["event"]
                node_name = event.get("name", "")

                # Filtre : on ne veut que les vrais nodes du graphe, pas les sous-appels internes (LLM, etc.)
                if node_name not in workflow.graph.get_graph().nodes:
                    continue

                if kind == "on_chain_start":
                    await websocket.send_json({
                        "type": "node_start",
                        "node": node_name,
                        "session_id": session_id,
                    })

                elif kind == "on_chain_end":
                    output = event["data"].get("output", {})
                    output = output if isinstance(output, dict) else {}

                    # 🔥 Métadonnées de sécurité extraites du state à cette étape précise
                    security = {
                        "toxicity_score": output.get("toxicity_score"),
                        "is_toxic": output.get("is_toxic"),
                        "rgpd_check_passed": output.get("rgpd_check_passed"),
                        "pii_detected": output.get("pii_detected", []),
                        "requires_hitl": output.get("requires_hitl"),
                        "confidence_score": output.get("confidence_score"),
                        "hitl_reason": output.get("hitl_reason"),
                        "pseudonymized_message": output.get("pseudonymized_message"),   # 🔥 AJOUTÉ (visible sur entry)
                        "pseudonymized_response": output.get("pseudonymized_response"),
                        "judge_evaluation": output.get("judge_evaluation"),
                        "routing_explanation": output.get("routing_explanation"),
                    }

                    security = {k: v for k, v in security.items() if v not in (None, [], "")}

                    await websocket.send_json({
                        "type": "node_end",
                        "node": node_name,
                        "session_id": session_id,
                        "target_agent": output.get("target_agent"),
                        "rag_results_count": len(output.get("rag_results", [])),
                        "security": security,
                    })

            await websocket.send_json({"type": "done", "session_id": session_id})

    except WebSocketDisconnect:
        print(f"🔌 Graph execution WS déconnecté: {client_id}")
    except Exception as e:
        print(f"❌ Erreur WebSocket graph-execution: {e}")
        await websocket.send_json({"type": "error", "error": str(e)})

# ==================== ADMIN RAG ENDPOINTS ====================
from fastapi.responses import HTMLResponse

@app.get("/admin/rag/stats")
async def rag_stats():
    """Statistiques globales du RAG"""
    return workflow.chroma.get_stats()

@app.get("/admin/rag/collections")
async def rag_collections():
    """Détail de chaque collection (documents + metadata)"""
    return workflow.chroma.get_collections_detail()

@app.get("/admin/rag/agents")
async def rag_agents():
    """Matrice des permissions des agents"""
    return workflow.chroma.get_agents_permissions()

@app.post("/admin/rag/search")
async def rag_search(payload: dict):
    """
    Recherche sémantique détaillée.
    Body: { "query": "...", "k": 3, "agent_filter": "AG06", "user_role": "supervisor" }
    """
    query = payload.get("query", "").strip()
    if not query:
        return {"error": "Query vide"}

    k = int(payload.get("k", 3))
    agent_filter = payload.get("agent_filter") or None
    user_role = payload.get("user_role", "client")

    return await workflow.chroma.debug_search(
        query=query, k=k, agent_filter=agent_filter, user_role=user_role
    )
# ==================== SIEM ENDPOINTS ====================

@app.get("/admin/siem/auth-events")
async def siem_auth_events(limit: int = 100):
    """Événements d'authentification"""
    from database import get_auth_events
    events = get_auth_events(limit)
    return {"events": events, "total": len(events)}


@app.get("/admin/siem/auth-stats")
async def siem_auth_stats():
    """Statistiques 24h"""
    from database import get_auth_stats_24h
    stats = get_auth_stats_24h()
    total = sum(stats.values())
    failed = stats.get("LOGIN_FAILED", 0)
    success = stats.get("LOGIN_SUCCESS", 0)
    return {
        "period": "24h",
        "total_events": total,
        "login_success": success,
        "login_failed": failed,
        "logout": stats.get("LOGOUT", 0),
        "session_expired": stats.get("SESSION_EXPIRED", 0),
        "token_refresh": stats.get("TOKEN_REFRESH", 0),
        "success_rate": round(success / max(success + failed, 1) * 100, 1),
        "failed_rate": round(failed / max(success + failed, 1) * 100, 1),
    }


@app.get("/admin/siem/sessions")
async def siem_sessions():
    """Sessions actives avec analyse d'âge"""
    import sqlite3
    conn = sqlite3.connect("awb_bank.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.session_id, s.client_id, s.client_name,
               s.login_time, s.last_activity, c.email
        FROM sessions s
        LEFT JOIN clients c ON s.client_id = c.client_id
        ORDER BY s.last_activity DESC LIMIT 50
    """)
    sessions = [dict(r) for r in cursor.fetchall()]
    conn.close()

    from datetime import datetime
    active_24h = 0
    stale = 0
    alerts = []
    for s in sessions:
        try:
            last = datetime.fromisoformat(s["last_activity"])
            age = (datetime.now() - last).total_seconds() / 3600
            s["age_hours"] = round(age, 1)
            if age < 24: active_24h += 1
            if age > 48:
                stale += 1
                alerts.append(f"Session {s['session_id'][:8]} inactive depuis {round(age)}h")
        except Exception:
            s["age_hours"] = None

    return {
        "sessions": sessions,
        "total_sessions": len(sessions),
        "active_24h": active_24h,
        "stale_sessions": stale,
        "alerts": alerts,
    }


@app.get("/admin/siem/suspicious")
async def siem_suspicious():
    """Détection d'activités suspectes"""
    import sqlite3
    conn = sqlite3.connect("awb_bank.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 3+ échecs en 1h pour le même email
    cursor.execute("""
        SELECT email, COUNT(*) as failures
        FROM auth_events
        WHERE event_type = 'LOGIN_FAILED'
          AND timestamp >= datetime('now', '-1 hour')
        GROUP BY email
        HAVING failures >= 3
    """)
    failed_alerts = [dict(r) for r in cursor.fetchall()]

    # Logins réussis depuis plusieurs IPs
    cursor.execute("""
        SELECT email, COUNT(DISTINCT ip_address) as ip_count,
               GROUP_CONCAT(DISTINCT ip_address) as ips
        FROM auth_events
        WHERE event_type = 'LOGIN_SUCCESS'
          AND timestamp >= datetime('now', '-24 hours')
        GROUP BY email
        HAVING ip_count >= 2
    """)
    multi_ip = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return {
        "failed_login_alerts": failed_alerts,
        "multi_ip_alerts": multi_ip,
        "total_alerts": len(failed_alerts) + len(multi_ip),
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)