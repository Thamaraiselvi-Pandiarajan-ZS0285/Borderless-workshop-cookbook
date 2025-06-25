import json
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import logging
from autogen_core import CancellationToken

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import sessionmaker
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_agentchat.messages import TextMessage, HandoffMessage

from backend.models.all_db_models import ChatMessage, Base


class DatabaseMemory(ListMemory):
    """Custom memory that persists to database while maintaining Autogen compatibility"""

    def __init__(self, conversation_id: str, db_engine: Engine, db_session: sessionmaker):
        super().__init__()
        self.conversation_id = conversation_id
        self.db_engine = db_engine
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)

        # Load existing messages from database
        self._load_from_database()

    async def _load_from_database(self):
        """Load existing messages from database into memory"""
        try:
            with self.db_session() as session:
                rows = (session.query(ChatMessage)
                        .filter_by(conversation_id=self.conversation_id)
                        .order_by(ChatMessage.message_index)
                        .all())

                for row in rows:
                    # Convert database row to MemoryContent
                    memory_content = MemoryContent(
                        content=row.content,
                        mime_type=MemoryMimeType.TEXT
                    )

                    # Add metadata if available
                    if row.metadata_in:
                        try:
                            metadata = json.loads(row.metadata_in)
                            memory_content.metadata = metadata
                        except json.JSONDecodeError:
                            pass

                    await super().add(memory_content)

                self.logger.info(f"Loaded {len(rows)} messages from database for conversation {self.conversation_id}")

        except Exception as e:
            self.logger.error(f"Error loading messages from database: {str(e)}")

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add memory content and persist to database if item is of type MemoryContent"""
        # Only process items of expected type
        if isinstance(content, MemoryContent):
            await super().add(content, cancellation_token)
            self._save_to_database(content)
        else:
            # Optionally log or raise a warning for unexpected types
            self.logger.warning(f"Unexpected item type added to memory: {type(content)}")

    def _save_to_database(self, memory_content: MemoryContent, sender: str = "system"):
        """Save memory content to database"""
        try:
            with self.db_session() as session:
                # Get the next message index
                max_index = (session.query(ChatMessage.message_index)
                             .filter_by(conversation_id=self.conversation_id)
                             .order_by(ChatMessage.message_index.desc())
                             .first())

                next_index = (max_index[0] + 1) if max_index and max_index[0] is not None else 0

                # Determine role and sender from metadata or defaults
                role = "assistant"
                if hasattr(memory_content, 'metadata') and memory_content.metadata:
                    role = memory_content.metadata.get('role', 'assistant')
                    sender = memory_content.metadata.get('sender', sender)

                db_msg = ChatMessage(
                    conversation_id=self.conversation_id,
                    role=role,
                    content=str(memory_content.content),
                    sender=sender,
                    message_index=next_index,
                    metadata_in=json.dumps(memory_content.metadata) if hasattr(memory_content,
                                                                               'metadata') and memory_content.metadata else None,
                    created_on=datetime.now()
                )

                session.add(db_msg)
                session.commit()

        except Exception as e:
            self.logger.error(f"Error saving to database: {str(e)}")


class SharedMemoryManager:
    """Manages shared memory across all agents in a conversation"""

    def __init__(self, conversation_id: str, db_engine: Engine, db_session: sessionmaker):
        self.conversation_id = conversation_id
        self.db_engine = db_engine
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)

        # Create shared memory instance
        self.shared_memory = DatabaseMemory(conversation_id, db_engine, db_session)

        # Track agent-specific memory contexts
        self.agent_memories: Dict[str, DatabaseMemory] = {}

    def get_shared_memory(self) -> DatabaseMemory:
        """Get the shared memory instance for all agents"""
        return self.shared_memory

    def get_agent_memory(self, agent_name: str) -> DatabaseMemory:
        """Get or create agent-specific memory"""
        if agent_name not in self.agent_memories:
            # Create agent-specific conversation ID
            agent_conversation_id = f"{self.conversation_id}_{agent_name}"
            self.agent_memories[agent_name] = DatabaseMemory(
                agent_conversation_id, self.db_engine, self.db_session
            )
        return self.agent_memories[agent_name]

    async def add_message_to_shared_memory(self, content: str, sender: str,
                                           role: str = "assistant",
                                           metadata: Dict = None,
                                           cancellation_token: Optional[CancellationToken] = None):
        """Add a message to shared memory with proper metadata"""
        memory_content = MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT
        )

        # Add metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'sender': sender,
            'role': role,
            'timestamp': datetime.now().isoformat()
        })
        memory_content.metadata = metadata

        await self.shared_memory.add(memory_content, cancellation_token)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history in a format suitable for agents"""
        try:
            with self.db_session() as session:
                rows = (session.query(ChatMessage)
                        .filter_by(conversation_id=self.conversation_id)
                        .order_by(ChatMessage.message_index)
                        .all())

                history = []
                for row in rows:
                    msg = {
                        'role': row.role,
                        'content': row.content,
                        'sender': row.sender,
                        'timestamp': row.created_on.isoformat() if row.created_on else None
                    }

                    if row.metadata_in:
                        try:
                            metadata = json.loads(row.metadata_in)
                            msg.update(metadata)
                        except json.JSONDecodeError:
                            pass

                    history.append(msg)

                return history

        except Exception as e:
            self.logger.error(f"Error getting conversation history: {str(e)}")
            return []

    def clear_conversation(self):
        """Clear all conversation data"""
        try:
            with self.db_session() as session:
                session.query(ChatMessage).filter_by(conversation_id=self.conversation_id).delete()

                # Also clear agent-specific conversations
                for agent_name in self.agent_memories.keys():
                    agent_conversation_id = f"{self.conversation_id}_{agent_name}"
                    session.query(ChatMessage).filter_by(conversation_id=agent_conversation_id).delete()

                session.commit()

        except Exception as e:
            self.logger.error(f"Error clearing conversation: {str(e)}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for monitoring"""
        try:
            with self.db_session() as session:
                total_messages = (session.query(ChatMessage)
                                  .filter_by(conversation_id=self.conversation_id)
                                  .count())

                agent_counts = {}
                for agent_name in self.agent_memories.keys():
                    agent_conversation_id = f"{self.conversation_id}_{agent_name}"
                    count = (session.query(ChatMessage)
                             .filter_by(conversation_id=agent_conversation_id)
                             .count())
                    agent_counts[agent_name] = count

                return {
                    'conversation_id': self.conversation_id,
                    'total_shared_messages': total_messages,
                    'agent_specific_messages': agent_counts,
                    'last_updated': datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Error getting memory stats: {str(e)}")
            return {}


class AutogenSessionManager:
    """Enhanced session manager with integrated memory management"""
    def __init__(self, db_engine: Engine, db_session: sessionmaker):
        from backend.app.core.orchestrator_agent import Orchestrator
        self.db_engine = db_engine
        self.db_session = db_session
        self.active_sessions: Dict[str, 'Orchestrator'] = {}
        self.memory_managers: Dict[str, SharedMemoryManager] = {}
        self.logger = logging.getLogger(__name__)

        # Ensure database tables exist
        Base.metadata.create_all(bind=self.db_engine)

    def create_session(self, conversation_id: str = None) -> str:
        from backend.app.core.orchestrator_agent import Orchestrator
        """Create new session with memory management"""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        # Create memory manager for this conversation
        memory_manager = SharedMemoryManager(conversation_id, self.db_engine, self.db_session)
        self.memory_managers[conversation_id] = memory_manager

        # Create orchestrator with memory
        orchestrator = Orchestrator(
            conversation_id=conversation_id,
            db_engine=self.db_engine,
            db_session=self.db_session,
            memory_manager=memory_manager
        )

        self.active_sessions[conversation_id] = orchestrator
        self.logger.info(f"Created new session with memory: {conversation_id}")
        return conversation_id

    def get_session(self, conversation_id: str) -> Optional['Orchestrator']:
        """Get existing session"""
        return self.active_sessions.get(conversation_id)

    def get_memory_manager(self, conversation_id: str) -> Optional[SharedMemoryManager]:
        """Get memory manager for a conversation"""
        return self.memory_managers.get(conversation_id)

    def load_session(self, conversation_id: str) -> Optional['Orchestrator']:
        """Load session from database with memory restoration"""
        try:
            # Create memory manager and load existing data
            memory_manager = SharedMemoryManager(conversation_id, self.db_engine, self.db_session)
            self.memory_managers[conversation_id] = memory_manager

            # Create orchestrator with restored memory
            from backend.app.core.orchestrator_agent import Orchestrator
            orchestrator = Orchestrator(
                conversation_id=conversation_id,
                db_engine=self.db_engine,
                db_session=self.db_session,
                memory_manager=memory_manager
            )

            self.active_sessions[conversation_id] = orchestrator

            # Get memory stats for logging
            stats = memory_manager.get_memory_stats()
            self.logger.info(f"Loaded session {conversation_id} with {stats.get('total_shared_messages', 0)} messages")

            return orchestrator

        except Exception as e:
            self.logger.error(f"Error loading session: {str(e)}")
            return None

    def save_session(self, conversation_id: str):
        """Save session - memory is auto-saved, but we can trigger manual sync"""
        try:
            memory_manager = self.memory_managers.get(conversation_id)
            if memory_manager:
                stats = memory_manager.get_memory_stats()
                self.logger.info(f"Session {conversation_id} memory stats: {stats}")

        except Exception as e:
            self.logger.error(f"Error saving session: {str(e)}")

    def close_session(self, conversation_id: str):
        """Close session and cleanup memory managers"""
        if conversation_id in self.active_sessions:
            del self.active_sessions[conversation_id]

        if conversation_id in self.memory_managers:
            del self.memory_managers[conversation_id]

        self.logger.info(f"Closed session {conversation_id}")

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations with memory statistics"""
        try:
            with self.db_session() as session:
                conversations = (session.query(ChatMessage.conversation_id)
                                 .distinct()
                                 .all())

                summaries = []
                for (conv_id,) in conversations:
                    # Skip agent-specific conversation IDs
                    if '_' in conv_id and len(conv_id.split('_')) > 5:  # UUID format check
                        continue

                    messages = (session.query(ChatMessage)
                                .filter_by(conversation_id=conv_id)
                                .order_by(ChatMessage.created_on)
                                .all())

                    if messages:
                        summary = {
                            'conversation_id': conv_id,
                            'message_count': len(messages),
                            'start_time': messages[0].created_on.isoformat(),
                            'last_activity': messages[-1].created_on.isoformat(),
                            'participants': list(set(msg.sender for msg in messages if msg.sender))
                        }
                        summaries.append(summary)

                return summaries

        except Exception as e:
            self.logger.error(f"Error listing conversations: {str(e)}")
            return []