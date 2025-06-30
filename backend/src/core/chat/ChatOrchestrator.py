from sqlalchemy.orm import Session
from autogen_agentchat.messages import TextMessage
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.db.models.chat_models import ChatMessage, ChatSession

class ChatOrchestrator:
    def __init__(self, db_engine: Engine, db_sessionmaker: sessionmaker):
        self.client = OpenAiClient().open_ai_chat_completion_client
        self.base_agent = BaseAgent(self.client)
        self.db_engine = db_engine
        self.db_sessionmaker = db_sessionmaker
        self.user_query_agent = self.base_agent.create_assistant_agent(
            name="ChatBotAgent",
            prompt="You are a helpful assistant specialized in market research."
        )

    def get_or_create_session(self, db: Session, session_name: str) -> ChatSession:
        session_obj = db.query(ChatSession).filter_by(session_name=session_name).first()
        if not session_obj:
            session_obj = ChatSession(session_name=session_name)
            db.add(session_obj)
            db.commit()
            db.refresh(session_obj)
        return session_obj

    def get_session_history(self, db: Session, session_obj: ChatSession):
        return [
            TextMessage(source=msg.role, content=msg.content)
            for msg in db.query(ChatMessage)
                        .filter_by(session_id=session_obj.id)
                        .order_by(ChatMessage.timestamp)
        ]

    def save_message(self, db: Session, session_obj: ChatSession, role: str, content: str):
        msg = ChatMessage(session_id=session_obj.id, role=role, content=content)
        db.add(msg)
        db.commit()

    async def process_user_query(self, session_name: str, user_input: str) -> str:
        """
        Handles a full user query round: history + prompt + saving to DB.
        """
        try:
            with self.db_sessionmaker() as db:
                chat_session = self.get_or_create_session(db, session_name)
                history = self.get_session_history(db, chat_session)

                assistant =self.user_query_agent
                history.append(TextMessage(source="user", content=user_input))

                reply =await assistant.run(task=history)
                assistant_msg = reply.messages[-1].content.strip()

                self.save_message(db, chat_session, "user", user_input)
                self.save_message(db, chat_session, "assistant", assistant_msg)

                return assistant_msg
        except Exception as e:
            raise