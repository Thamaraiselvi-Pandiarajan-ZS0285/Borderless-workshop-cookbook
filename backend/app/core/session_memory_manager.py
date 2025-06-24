import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import sessionmaker

from backend.app.core.orchestrator_agent import Orchestrator
from backend.models.all_db_models import ChatMessage, Base
import logging


class AutogenMemoryManager:
    """Memory manager that works with Autogen's built-in chat_messages"""

    def __init__(self, db_engine: Engine, db_session: sessionmaker):
        self.db_engine = db_engine
        self.db_session = db_session
        Base.metadata.create_all(bind=self.db_engine)
        self.logger = logging.getLogger(__name__)

    def save_conversation(self, conversation_id: str, messages: List[Dict[str, Any]],
                          metadata: Optional[Dict] = None):
        """Save conversation messages to database"""
        try:
            with self.db_session() as session:
                # Clear existing messages for this conversation
                session.query(ChatMessage).filter_by(conversation_id=conversation_id).delete()

                # Save new messages
                for idx, msg in enumerate(messages):
                    db_msg = ChatMessage(
                        conversation_id=conversation_id,
                        role=msg.get('role', 'unknown'),
                        content=msg.get('content', ''),
                        sender=msg.get('name', 'unknown')   ,
                        timestamp=datetime.now(),
                        message_index=idx,
                        metadata=json.dumps(metadata) if metadata else None
                    )
                    session.add(db_msg)

                session.commit()
                self.logger.info(f"Saved {len(messages)} messages for conversation {conversation_id}")

        except Exception as e:
            self.logger.error(f"Error saving conversation: {str(e)}")
            raise

    def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load conversation messages from database"""
        try:
            with self.db_session() as session:
                rows = (session.query(ChatMessage)
                        .filter_by(conversation_id=conversation_id)
                        .order_by(ChatMessage.message_index)
                        .all())

                messages = []
                for row in rows:
                    msg = {
                        'role': row.role,
                        'content': row.content,
                        'name': row.sender
                    }
                    if hasattr(row, 'metadata_in') and row.metadata_in:
                        try:
                            metadata = json.loads(row.metadata_in)
                            if isinstance(metadata, dict):
                                msg.update(metadata)
                        except json.JSONDecodeError:
                            pass
                    messages.append(msg)

                self.logger.info(f"Loaded {len(messages)} messages for conversation {conversation_id}")
                return messages

        except Exception as e:
            self.logger.error(f"Error loading conversation: {str(e)}")
            return []

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation summary with metadata"""
        try:
            with self.db_session() as session:
                messages = (session.query(ChatMessage)
                            .filter_by(conversation_id=conversation_id)
                            .order_by(ChatMessage.created_on)
                            .all())

                if not messages:
                    return {}

                return {
                    'conversation_id': conversation_id,
                    'message_count': len(messages),
                    'start_time': messages[0].timestamp.isoformat(),
                    'last_activity': messages[-1].timestamp.isoformat(),
                    'participants': list(set(msg.sender for msg in messages))
                }

        except Exception as e:
            self.logger.error(f"Error getting conversation summary: {str(e)}")
            return {}

    def delete_conversation(self, conversation_id: str):
        """Delete conversation from database"""
        try:
            with self.db_session() as session:
                deleted_count = (session.query(ChatMessage)
                                 .filter_by(conversation_id=conversation_id)
                                 .delete())
                session.commit()
                self.logger.info(f"Deleted {deleted_count} messages for conversation {conversation_id}")

        except Exception as e:
            self.logger.error(f"Error deleting conversation: {str(e)}")
            raise


class AutogenSessionManager:
    """Session manager for Autogen workflows with persistence"""

    def __init__(self, db_engine: Engine, db_session: sessionmaker):
        self.memory_manager = AutogenMemoryManager(db_engine, db_session)
        self.active_sessions: Dict[str, 'Orchestrator'] = {}
        self.logger = logging.getLogger(__name__)

    def create_session(self, conversation_id: str = None) -> str:
        """Create new session"""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        # Import here to avoid circular imports
        from backend.app.core.orchestrator_agent import Orchestrator

        orchestrator = Orchestrator(conversation_id=conversation_id)
        self.active_sessions[conversation_id] = orchestrator

        self.logger.info(f"Created new session: {conversation_id}")
        return conversation_id

    def get_session(self, conversation_id: str) -> Optional['Orchestrator']:
        """Get existing session"""
        return self.active_sessions.get(conversation_id)

    def load_session(self, conversation_id: str) -> Optional['Orchestrator']:
        """Load session from database"""
        try:
            # Load conversation history
            messages = self.memory_manager.load_conversation(conversation_id)

            # Inside load_session before setting chat_messages
            for msg in messages:
                if not isinstance(msg, dict):
                    raise ValueError(f"Invalid message format: {msg}")
                if 'name' not in msg or 'content' not in msg:
                    raise ValueError(f"Missing required keys in message: {msg}")
            if not messages:
                self.logger.warning(f"No messages found for conversation {conversation_id}")
                return None

            # Create new orchestrator instance
            from backend.app.core.orchestrator_agent import Orchestrator
            orchestrator = Orchestrator(conversation_id=conversation_id)

            # Restore chat history to all agents
            for agent in orchestrator.agents:
                if hasattr(agent, 'chat_messages'):
                    agent.chat_messages = messages.copy()

            # Restore group chat messages
            orchestrator.group_chat.messages = messages.copy()

            # Add to active sessions
            self.active_sessions[conversation_id] = orchestrator
            self.logger.info(f"Restoring messages: {messages}")
            self.logger.info(f"Loaded session {conversation_id} with {len(messages)} messages")
            return orchestrator

        except Exception as e:
            self.logger.error(f"Error loading session: {str(e)}")
            return None

    def save_session(self, conversation_id: str, metadata: Optional[Dict] = None):
        """Save session to database"""
        orchestrator = self.active_sessions.get(conversation_id)
        if not orchestrator:
            self.logger.warning(f"Session {conversation_id} not found in active sessions")
            return

        try:
            # Get messages from group chat (Autogen's built-in message history)
            messages = orchestrator.group_chat.messages

            # Add metadata
            if metadata is None:
                metadata = {}
            metadata.update({
                'current_stage': orchestrator.current_stage.value,
                'agent_count': len(orchestrator.agents),
                'saved_at': datetime.utcnow().isoformat()
            })

            # Save to database
            self.memory_manager.save_conversation(conversation_id, messages, metadata)
            self.logger.info(f"Saved session {conversation_id}")

        except Exception as e:
            self.logger.error(f"Error saving session: {str(e)}")
            raise

    def close_session(self, conversation_id: str, save_before_close: bool = True):
        """Close and optionally save session"""
        if save_before_close:
            self.save_session(conversation_id)

        if conversation_id in self.active_sessions:
            del self.active_sessions[conversation_id]
            self.logger.info(f"Closed session {conversation_id}")

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations with summaries"""
        try:
            with self.memory_manager.db_session() as session:
                conversations = (session.query(ChatMessage.conversation_id)
                                 .distinct()
                                 .all())

                summaries = []
                for (conv_id,) in conversations:
                    summary = self.memory_manager.get_conversation_summary(conv_id)
                    if summary:
                        summaries.append(summary)

                return summaries

        except Exception as e:
            self.logger.error(f"Error listing conversations: {str(e)}")
            return []

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.active_sessions.keys())

