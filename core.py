# Optimized Gigi Personal Growth Coach - LangGraph Framework
# Now with multi-user, multi-session, and ChromaDB-powered long-term memory

import os
import json
import asyncio
import logging
import uuid
import time
from google.api_core import exceptions
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict

# Core dependencies
from dotenv import load_dotenv
load_dotenv()

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Database
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

# AI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ======================== 
# CONFIGURATION
# ======================== 

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./gigi_coach.db")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./gigi_memory")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ======================== 
# DATABASE MODELS
# ======================== 

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("Session", back_populates="user")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="sessions")
    conversations = relationship("Conversation", back_populates="session")
    goals = relationship("Goal", back_populates="session")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="conversations")

class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    goal_data = Column(Text, nullable=False)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="goals")

# Initialize database
engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_database():
    """Initialize database with correct schema"""
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization error: {e}")

init_database()

# ======================== 
# VECTOR DATABASE SERVICE (CHROMA)
# ======================== 

class LocalEmbeddingFunction(EmbeddingFunction):
    """
    An embedding function that runs a local sentence-transformer model.
    """
    def __init__(self):
        print("Initializing local embedding model. This may take a moment...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Local embedding model loaded.")

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embeds a list of documents using the local model.
        """
        return self.model.encode(input).tolist()

class ChromaDBService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.embedding_function = LocalEmbeddingFunction()
    
    def get_or_create_collection(self, user_id: str):
        """Get or create a ChromaDB collection for a user"""
        collection_name = f"user_{user_id.replace('-', '_')}_local"
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_conversation_to_memory(self, user_id: str, conversation: str):
        """Add a piece of conversation to the user's memory"""
        collection = self.get_or_create_collection(user_id)
        collection.add(
            documents=[conversation],
            ids=[str(uuid.uuid4())]
        )

    def search_memory(self, user_id: str, query: str, n_results: int = 3) -> List[str]:
        """Search for relevant memories for a user"""
        try:
            collection = self.get_or_create_collection(user_id)
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logging.error(f"ChromaDB search error with local model: {e}")
            return []

chroma_service = ChromaDBService()

# ======================== 
# LANGGRAPH STATE
# ======================== 

class AgentState(TypedDict):
    user_message: str
    user_id: str
    session_id: str
    current_step: str
    analysis_complete: bool
    goal_identified: bool
    plan_generated: bool
    user_profile: Optional[Dict[str, Any]]
    current_goal: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    retrieved_context: Optional[str]
    user_analysis: Optional[str]
    goal_assessment: Optional[Dict[str, Any]]
    comprehensive_plan: Optional[str]
    response_message: str
    created_at: str
    last_updated: str
    processing_errors: List[str]

# ========================
# AI SERVICE
# ========================

class AIService:
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    async def analyze_user_input(self, user_message: str, context: Dict = None) -> Dict[str, Any]:
        """Analyze user input with context and return a structured JSON analysis."""
        context_str = ""
        if context and context.get("retrieved_context"):
            context_str += f"\n\n--- Relevant Long-Term Memories ---\n{context['retrieved_context']}"

        history_str = ""
        if context and context.get("conversation_history"):
            recent_history = context['conversation_history'][:3]
            formatted_history = "\n".join([f"User: {h['message']}\nGigi: {h['response']}" for h in reversed(recent_history)])
            history_str = f"\n\n--- Recent Conversation History ---\n{formatted_history}"

        prompt = f"""You are Gigi, an expert personal growth coach. Your task is to provide a deep, actionable analysis of the user's message by synthesizing information from multiple sources.

**Core Directives:**
1.  **Recall and Connect:** Always connect the user's current message to their past goals and conversations.
2.  **Detect Repetition:** If the user is asking a question they've asked recently, acknowledge it.
3.  **Empathize and Refocus:** If the user expresses low motivation, validate their feelings and gently reconnect them to their primary stated goals.

**Context Analysis:**
Below is the user's current message, recent conversation history, and relevant long-term memories.

{history_str}
{context_str}

--- Current User Message ---
{user_message}

--- Analysis Task ---
Based on ALL the information above, provide a JSON object with the following structure. DO NOT add any text outside the JSON object.
{{
  "connection_to_past": "How does this message relate to the user's main goals and recent conversation?",
  "emotional_state": "What is the user's likely emotional state (e.g., motivated, overwhelmed)?",
  "implicit_need": "What is the user's unspoken need (e.g., reassurance, a specific plan, a reminder of their goal)?",
  "task_type": "Categorize the user's request. Choose one: 'NewGoal', 'RefineGoal', 'QuestionAnswering', 'MotivationSupport'.",
  "key_insight_for_response": "What is the single most important thing to address in the next response?"
}}
"""
        response_text = self.model.generate_content(prompt).text
        return self._parse_json_safe(response_text)

    async def assess_goals(self, user_message: str, user_analysis: str) -> Dict[str, Any]:
        """Extract and assess goals from user input"""
        prompt = f"""Extract goal information from user message and analysis.

Return ONLY valid JSON:
{{
    "primary_goal": "specific goal statement",
    "domains": ["nutrition", "fitness", "study", "lifestyle", "career"],
    "timeframe": "specific timeframe",
    "desired_outcomes": ["outcome 1", "outcome 2"],
    "difficulty_level": "beginner|intermediate|advanced",
    "motivation_score": 7
}}

User Message: {user_message}
Analysis: {user_analysis}
"""
        response = self.model.generate_content(prompt).text
        return self._parse_json_safe(response)

    async def generate_plan(self, goal_data: Dict, user_context: Dict = None) -> str:
        """Generate comprehensive action plan"""
        prompt = f"""Create a personalized wellness plan:

Goal: {goal_data}
Context: {user_context or "New user"}

Create structured plan with:
1. Week-by-week breakdown (4 weeks)
2. Daily routines (specific and realistic)
3. Progress milestones
4. Potential challenges and solutions
5. Success metrics

Use markdown formatting, under 800 words but comprehensive.
"""
        return self.model.generate_content(prompt).text

    async def generate_direct_answer(self, state: AgentState) -> str:
        """Generates a direct, contextual answer to a user's question."""
        
        history = state.get("conversation_history", [])
        history_str = "\n".join([f"User: {h['message']}\nGigi: {h['response']}" for h in reversed(history[:3])])

        analysis = state.get("user_analysis", {})
        analysis_str = json.dumps(analysis, indent=2)

        prompt = f"""You are Gigi, a personal growth coach. Your task is to provide a direct, helpful, and empathetic answer to the user's question based on the full context of the conversation.

**Context:**
- **User's Goal:** The user's primary goal is often mentioned in the history. Find it and use it.
- **Analysis of Current Message:** {analysis_str}
- **Recent Conversation History:**
{history_str}

**Task:**
Based on the analysis and history, generate a direct response to the user's message: "{state['user_message']}".

**Instructions:**
- Do not act as an analyzer; speak directly to the user as Gigi.
- If the analysis says to remind the user of their goal, do so clearly and concisely as you described in your ideal response.
- If the analysis says the user is unmotivated, be encouraging and connect your answer back to their goals.
- Keep the response natural and conversational.
"""
        return self.model.generate_content(prompt).text

    def _parse_json_safe(self, text: str) -> Dict[str, Any]:
        """Robust JSON parsing"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(text[start:end])
            except:
                pass
        return {
            "primary_goal": "Personal growth and wellness",
            "domains": ["lifestyle"],
            "timeframe": "4 weeks",
            "desired_outcomes": ["Improved well-being"],
            "difficulty_level": "beginner",
            "motivation_score": 5
        }

ai_service = AIService()

# ======================== 
# DATABASE SERVICE
# ======================== 

class DatabaseService:
    def __init__(self):
        self.SessionLocal = SessionLocal

    def get_or_create_user(self, username: str) -> User:
        """Get or create a user by username"""
        db = self.SessionLocal()
        try:
            user = db.query(User).filter_by(username=username).first()
            if user:
                return user
            
            new_user = User(username=username)
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            return new_user
        finally:
            db.close()

    def create_session(self, user_id: str) -> Session:
        """Create a new session for a user"""
        db = self.SessionLocal()
        try:
            new_session = Session(user_id=user_id)
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            return new_session
        finally:
            db.close()

    def save_conversation(self, session_id: str, message: str, response: str):
        """Save conversation to database"""
        db = self.SessionLocal()
        try:
            conv = Conversation(session_id=session_id, message=message, response=response)
            db.add(conv)
            db.commit()
        finally:
            db.close()

    def save_goal(self, session_id: str, goal_data: Dict):
        """Save goal to database"""
        db = self.SessionLocal()
        try:
            goal = Goal(session_id=session_id, goal_data=json.dumps(goal_data))
            db.add(goal)
            db.commit()
        finally:
            db.close()

    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user conversation history across all sessions. The user_id parameter is the username."""
        db = self.SessionLocal()
        try:
            # Find the user by their username (which is what's passed from the terminal)
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []  # If user doesn't exist, return no history.
            
            # Correctly query conversations by joining through Session and filtering on the user's actual ID (UUID)
            conversations = db.query(Conversation).join(Session).filter(Session.user_id == user.user_id).order_by(Conversation.created_at.desc()).limit(limit).all()
            return [
                {"message": conv.message, "response": conv.response, "timestamp": conv.created_at.isoformat()}
                for conv in conversations
            ]
        finally:
            db.close()

db_service = DatabaseService()

# ======================== 
# LANGGRAPH NODES
# ======================== 

async def start_session_node(state: AgentState) -> AgentState:
    """Initialize user and session, and load history."""
    now = datetime.utcnow()
    user = db_service.get_or_create_user(state["user_id"])
    session = db_service.create_session(user.user_id)
    history = db_service.get_user_history(state["user_id"], limit=10)
    
    state.update({
        "user_id": user.user_id,
        "session_id": session.session_id,
        "conversation_history": history,
        "current_step": "retrieving_context",
        "created_at": now.isoformat(),
        "last_updated": now.isoformat(),
        "processing_errors": []
    })
    return state

async def retrieve_context_node(state: AgentState) -> AgentState:
    """Retrieve relevant context from ChromaDB"""
    try:
        retrieved_docs = chroma_service.search_memory(state["user_id"], state["user_message"])
        state["retrieved_context"] = "\n".join(retrieved_docs)
        state["current_step"] = "analyzing_input"
    except Exception as e:
        state["processing_errors"].append(f"Context retrieval failed: {e}")
        state["current_step"] = "error_handling"
    return state

async def analyze_input_node(state: AgentState) -> AgentState:
    """Analyze user input with context"""
    try:
        # Pass both long-term (retrieved) and short-term (history) context
        context = {
            "retrieved_context": state.get("retrieved_context"),
            "conversation_history": state.get("conversation_history")
        }
        analysis = await ai_service.analyze_user_input(state["user_message"], context)
        
        state["user_analysis"] = analysis
        state["analysis_complete"] = True
        state["current_step"] = "identifying_goals"
    except Exception as e:
        state["processing_errors"].append(f"Analysis failed: {e}")
        state["current_step"] = "error_handling"
    return state

async def identify_goals_node(state: AgentState) -> AgentState:
    """Extract and assess goals"""
    try:
        goal_data = await ai_service.assess_goals(state["user_message"], state.get("user_analysis", ""))
        state["goal_assessment"] = goal_data
        state["current_goal"] = goal_data
        state["goal_identified"] = True
        state["current_step"] = "generating_plan"
    except Exception as e:
        state["processing_errors"].append(f"Goal identification failed: {e}")
        state["current_step"] = "error_handling"
    return state

async def generate_plan_node(state: AgentState) -> AgentState:
    """Generate comprehensive plan"""
    try:
        plan = await ai_service.generate_plan(state.get("goal_assessment", {}), state.get("user_profile"))
        state["comprehensive_plan"] = plan
        state["plan_generated"] = True
        state["current_step"] = "finalizing_response"
    except Exception as e:
        state["processing_errors"].append(f"Plan generation failed: {e}")
        state["current_step"] = "error_handling"
    return state

async def finalize_response_node(state: AgentState) -> AgentState:
    """Create final response for a plan-based workflow and save data."""
    try:
        # Construct response
        response_parts = []
        # The user analysis is an internal thought process and should not be shown to the user.
        # The plan is the primary output of this path.
        if state.get('comprehensive_plan'):
            response_parts.append(f"## Your Personalized Action Plan\n{state['comprehensive_plan']}")
        else:
            # Fallback if a plan was expected but not generated
            response_parts.append("Here is the information you requested.")

        if state.get("current_goal"):
            goal = state["current_goal"]
            response_parts.append(f"\n## Goal Summary\n**Primary Goal:** {goal.get('primary_goal', 'N/A')}\n**Timeframe:** {goal.get('timeframe', 'N/A')}\n")
        
        response_parts.append("\n---\n*I'm here to support you!*")
        state["response_message"] = "\n".join(response_parts)
        
        # Save to databases
        db_service.save_conversation(state["session_id"], state["user_message"], state["response_message"])
        if state.get("current_goal"):
            db_service.save_goal(state["session_id"], state["current_goal"])
        
        # Add to long-term memory
        memory_text = f"User said: '{state['user_message']}'. Goal was: {state.get('current_goal', {}).get('primary_goal', 'N/A')}. Plan created."
        chroma_service.add_conversation_to_memory(state["user_id"], memory_text)

        state["current_step"] = "complete"
    except Exception as e:
        state["processing_errors"].append(f"Response finalization failed: {e}")
        state["current_step"] = "error_handling"
    return state

async def answer_question_node(state: AgentState) -> AgentState:
    """Generates a direct answer to a user's question and ends the workflow."""
    try:
        answer = await ai_service.generate_direct_answer(state)
        state["response_message"] = answer
    except Exception as e:
        state["processing_errors"].append(f"Direct answer generation failed: {e}")
    # This will now be routed to the error handler node
    return state

async def error_handling_node(state: AgentState) -> AgentState:
    """Handle errors gracefully"""
    error_messages = state.get("processing_errors", ["An unknown error occurred."])
    full_error_message = "\n".join(error_messages)
    logging.error(f"Workflow failed with errors: {full_error_message}")
    
    user_facing_error = f"I apologize, but I encountered a technical difficulty. The specific error was: {full_error_message}"
    state["response_message"] = user_facing_error
    state["current_step"] = "complete"
    return state

# ======================== 
# LANGGRAPH WORKFLOW
# ======================== 

def create_workflow():
    """Create LangGraph workflow"""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_session", start_session_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("analyze_input", analyze_input_node)
    workflow.add_node("identify_goals", identify_goals_node)
    workflow.add_node("generate_plan", generate_plan_node)
    workflow.add_node("finalize_response", finalize_response_node)
    workflow.add_node("answer_question", answer_question_node)
    workflow.add_node("error_handling", error_handling_node)
    
    workflow.set_entry_point("start_session")
    workflow.add_edge("start_session", "retrieve_context")
    workflow.add_edge("retrieve_context", "analyze_input")

    def decide_next_step(state: AgentState):
        if state.get("processing_errors"):
            return "error_handling"
        
        analysis = state.get("user_analysis", {})
        task_type = analysis.get("task_type", "NewGoal")
        
        if task_type in ["QuestionAnswering", "MotivationSupport"]:
            return "answer_question"
        else: # NewGoal, RefineGoal
            return "identify_goals"

    workflow.add_conditional_edges(
        "analyze_input",
        decide_next_step,
        {
            "error_handling": "error_handling",
            "answer_question": "answer_question",
            "identify_goals": "identify_goals",
        }
    )

    workflow.add_edge("answer_question", END)
    workflow.add_conditional_edges("identify_goals", lambda s: "error_handling" if s.get("processing_errors") else "generate_plan")
    workflow.add_conditional_edges("generate_plan", lambda s: "error_handling" if s.get("processing_errors") else "finalize_response")
    workflow.add_edge("finalize_response", END)
    workflow.add_edge("error_handling", END)
    
    return workflow

# ======================== 
# MAIN SERVICE
# ======================== 

class GigiService:
    def __init__(self):
        self.workflow = create_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    async def process_message(self, user_message: str, user_id: str) -> Dict[str, Any]:
        """Process user message through LangGraph workflow"""
        config = {"configurable": {"thread_id": user_id}} # Use user_id as thread_id for state persistence
        
        current_input = {"user_message": user_message, "user_id": user_id}
        
        try:
            result = await self.app.ainvoke(current_input, config)
            return {
                "success": True,
                "response": result.get("response_message", "I'm here to help!"),
                "session_id": result.get("session_id"),
                "user_id": result.get("user_id")
            }
        except Exception as e:
            logging.error(f"Workflow execution failed: {e}")
            return {"success": False, "response": "An error occurred.", "error": str(e)}

    async def get_history(self, user_id: str) -> Dict[str, Any]:
        """Get user's conversation history"""
        return {"success": True, "conversation_history": db_service.get_user_history(user_id)}

# ======================== 
# API INTERFACE
# ======================== 

class GigiAPI:
    def __init__(self):
        self.service = GigiService()

    async def chat(self, message: str, user_id: str) -> Dict[str, Any]:
        """Main chat endpoint"""
        return await self.service.process_message(message, user_id)

    async def get_history(self, user_id: str) -> Dict[str, Any]:
        """Get conversation history"""
        return await self.service.get_history(user_id)

    async def health_check(self) -> Dict[str, Any]:
        """System health check"""
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ======================== 
# USAGE EXAMPLE
# ======================== 

async def main():
    """Example usage"""
    api = GigiAPI()
    print("=== Gigi Personal Growth Coach Demo ===\n")
    
    user_id = "demo_user_123"
    
    # First interaction
    print("\n=== New User Interaction ===")
    response1 = await api.chat(
        "I want to lose 5kg in 8 weeks while studying for my finals. I'm vegetarian.",
        user_id=user_id
    )
    print(f"Response:\n{response1['response']}")
    
    # Follow-up
    print("\n=== Follow-up (with memory) ===")
    response2 = await api.chat(
        "That sounds good. What are some quick meal prep ideas for a busy student?",
        user_id=user_id
    )
    print(f"Follow-up Response:\n{response2['response']}")

if __name__ == "__main__":
    asyncio.run(main())