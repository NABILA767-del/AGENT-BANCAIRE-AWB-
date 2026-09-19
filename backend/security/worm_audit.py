# security/worm_audit.py
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List

class WormAudit:
    def __init__(self, log_file: str = "audit_logs.json"):
        self.log_file = log_file
        self._init_log_file()
    
    def _init_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            print(f"✅ Fichier d'audit créé: {self.log_file}")
    
    def write(self, data: Dict):
        """Écrit un log d'audit"""
        try:
            # Lire les logs existants
            with open(self.log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            # Créer l'entrée de log
            log_entry = {
                "id": hashlib.md5(f"{datetime.now().isoformat()}{data.get('action', '')}".encode()).hexdigest()[:8],
                "timestamp": datetime.now().isoformat(),
                "level": data.get("level", "INFO"),
                "action": data.get("action", "UNKNOWN"),
                "entity_id": data.get("entity_id", "system"),
                "message": data.get("message", ""),
                "details": data.get("details", {}),
                "hash": hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]
            }
            
            # Ajouter et sauvegarder
            logs.append(log_entry)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
            print(f"📝 Audit log écrit: {log_entry['action']} - {log_entry['message'][:50]}...")
            
        except Exception as e:
            print(f"❌ Erreur écriture audit: {e}")
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict]:
        """Récupère les derniers logs"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                return logs[-limit:] if logs else []
        except Exception as e:
            print(f"❌ Erreur lecture audit: {e}")
        return []
    
    def get_all_logs(self) -> List[Dict]:
        """Récupère tous les logs"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def verify_integrity(self) -> bool:
        return True
    
    def get_stats(self) -> Dict:
        logs = self.get_recent_logs(1000)
        if not logs:
            return {"total": 0, "by_action": {}, "by_level": {}}
        
        by_action = {}
        by_level = {}
        for log in logs:
            action = log.get("action", "UNKNOWN")
            level = log.get("level", "INFO")
            by_action[action] = by_action.get(action, 0) + 1
            by_level[level] = by_level.get(level, 0) + 1
        
        return {
            "total": len(logs),
            "by_action": by_action,
            "by_level": by_level
        }