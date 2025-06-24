import json
from memory_log_model import MemoryLog

class SharedMemoryManager:
    def __init__(self, db_session):
        self.db = db_session

    def log_interaction(self, session_id, agent_name, step, input_text, output_text, context=None):
        try:
            log = MemoryLog(
                session_id=session_id,
                agent_name=agent_name,
                step=step,
                input_text=input_text,
                output_text=output_text,
                context=json.dumps(context) if context else None
            )
            self.db.add(log)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise e

    def get_logs(self, session_id):
        logs = self.db.query(MemoryLog).filter_by(session_id=session_id).order_by(MemoryLog.step).all()
        return [{
            "step": l.step,
            "agent_name": l.agent_name,
            "input": l.input_text,
            "output": l.output_text,
            "context": json.loads(l.context) if l.context else {}
        } for l in logs]
