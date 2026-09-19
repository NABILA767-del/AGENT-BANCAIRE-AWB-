# security/hitl_gate.py
from typing import Dict, List
from datetime import datetime
import uuid

class HITLGate:
    def __init__(self):
        self.pending_requests = {}
        print("✅ HITL Gate initialisé")
    
    async def request_approval(self, session_id: str, agent: str, context: str, confidence: float) -> Dict:
        request_id = str(uuid.uuid4())[:8]
        self.pending_requests[request_id] = {
            "request_id": request_id,
            "session_id": session_id,
            "agent": agent,
            "agent_name": self._get_agent_name(agent),
            "context": context[:200],
            "confidence": confidence,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
        print(f"🟡 HITL Request created: {request_id} for agent {agent}")
        return {"status": "pending", "request_id": request_id}
    
    async def get_pending_requests(self) -> List[Dict]:
        return list(self.pending_requests.values())
    
    async def approve(self, request_id: str, approver: str) -> Dict:
        if request_id in self.pending_requests:
            self.pending_requests[request_id]["status"] = "approved"
        return {"status": "approved", "request_id": request_id}
    
    async def reject(self, request_id: str, reason: str) -> Dict:
        if request_id in self.pending_requests:
            self.pending_requests[request_id]["status"] = "rejected"
        return {"status": "rejected", "request_id": request_id}
    
    def _get_agent_name(self, agent: str) -> str:
        names = {
            "AG03": "Réclamation",
            "AG05": "Incident",
            "AG06": "Crédit",
            "AG07": "Admin",
            "AG08": "Opposition",
            "AG09": "Anti-Fraude",
            "AG10": "Clôture",
            "AG11": "Succession",
            "AG13": "RGPD"
        }
        return names.get(agent, agent)