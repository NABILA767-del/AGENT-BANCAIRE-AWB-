from agents.base_agent import BaseAgent
from typing import Dict
import hashlib
import json
from datetime import datetime

class AuditAgent(BaseAgent):
    def __init__(self):
        super().__init__("AG14", "qwen:7b")
        self.worm_logs = []
    
    async def log_action(self, action: str, entity_id: str, details: Dict) -> Dict:
        log_entry = {"timestamp": datetime.now().isoformat(), "action": action, "entity_id": entity_id, "details": details, "agent_id": "AG14"}
        self.worm_logs.append(log_entry)
        return log_entry
    
    async def process(self, message: str, context: Dict = None) -> Dict:
        return {"message": "Action auditée", "confidence": 1.0}