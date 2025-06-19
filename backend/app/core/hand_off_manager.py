from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from autogen import ConversableAgent
from datetime import datetime

@dataclass
class HandoffConfig:
    """Configuration for agent handoffs"""
    from_agent: str
    to_agent: str
    condition: Optional[Callable] = None
    data_transform: Optional[Callable] = None
    validation: Optional[Callable] = None


class HandoffManager:
    """Manages handoffs between agents in the workflow"""

    def __init__(self):
        self.handoff_configs: List[HandoffConfig] = []
        self.active_handoffs: Dict[str, str] = {}
        self.handoff_history: List[Dict] = []

    def register_handoff(self, config: HandoffConfig):
        """Register a new handoff configuration"""
        self.handoff_configs.append(config)

    def execute_handoff(self, from_agent: str, to_agent: str, data: Any = None) -> Dict[str, Any]:
        """Execute handoff between agents"""

        # Find matching handoff config
        config = next((c for c in self.handoff_configs
                       if c.from_agent == from_agent and c.to_agent == to_agent), None)

        if not config:
            raise ValueError(f"No handoff configuration found for {from_agent} -> {to_agent}")

        # Validate if condition exists
        if config.condition and not config.condition(data):
            return {"status": "failed", "reason": "Handoff condition not met"}

        # Transform data if needed
        if config.data_transform:
            data = config.data_transform(data)

        # Validate data if needed
        if config.validation and not config.validation(data):
            return {"status": "failed", "reason": "Data validation failed"}

        # Record handoff
        handoff_record = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "status": "completed"
        }

        self.handoff_history.append(handoff_record)
        self.active_handoffs[to_agent] = from_agent

        return {"status": "completed", "data": data, "handoff_id": len(self.handoff_history) - 1}
