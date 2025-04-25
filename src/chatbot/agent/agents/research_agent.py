"""
Research Agent Module

This module implements the Research Agent for the Personal Trainer AI system.
The Research Agent is responsible for conducting deep fitness research based on user queries.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .base_agent import BaseAgent
from ..llm_providers import LLMProvider


class ResearchAgent(BaseAgent):
    """
    Research Agent for conducting deep fitness research.
    
    This agent is responsible for:
    1. Planning research steps
    2. Generating RAG queries
    3. Executing RAG queries
    4. Synthesizing RAG results
    5. Reflecting on progress
    6. Finalizing research reports
    """
    
    def __init__(self, llm_provider: LLMProvider):
        """Initialize the Research Agent with an LLM provider."""
        super().__init__(llm_provider)
        self.name = "research_agent"
        self.description = "Conducts deep fitness research based on user queries"
    
    async def process(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Process the current state and conduct research.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state
        """
        # Determine which research step to execute based on state
        research_step = state.get("research_step", "plan_research")
        
        if research_step == "plan_research":
            return await self.plan_research_steps(state, config)
        elif research_step == "generate_rag_query":
            return await self.generate_rag_query(state, config)
        elif research_step == "execute_rag":
            return await self.execute_rag_direct(state, config)
        elif research_step == "synthesize_results":
            return await self.synthesize_rag_results(state, config)
        elif research_step == "reflect_on_progress":
            return await self.reflect_on_progress(state, config)
        elif research_step == "finalize_report":
            return await self.finalize_research_report(state, config)
        else:
            # Default to planning research
            return await self.plan_research_steps(state, config)
    
    async def plan_research_steps(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Plan the research steps for a given research topic.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state with research plan
        """
        research_topic = state.get("research_topic", "")
        user_profile = state.get("user_profile_str", "")
        
        system_prompt = f"""You are a fitness research expert planning a deep research investigation.
        
TASK: Break down the research topic into 3-5 specific sub-questions that will help thoroughly investigate the topic.

RESEARCH TOPIC: {research_topic}

USER PROFILE: {user_profile}

OUTPUT REQUIREMENTS:
1. Provide 3-5 specific sub-questions that will help thoroughly investigate the topic
2. For each sub-question, provide 1-2 initial search queries that would help answer it
3. Explain why each sub-question is important for the overall research

Your output should be structured and focused on creating an effective research plan.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please create a research plan for: {research_topic}")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # Extract sub-questions from response
        # This is a simplified implementation - in a real system, you'd want to parse the response more carefully
        sub_questions = []
        for line in response.split("\n"):
            if line.strip().startswith("Sub-question") or line.strip().startswith("- Sub-question"):
                question_text = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                sub_questions.append(question_text)
        
        # Update state
        updated_state = state.copy()
        updated_state["sub_questions"] = sub_questions
        updated_state["current_sub_question_idx"] = 0
        updated_state["iteration_count"] = 0
        updated_state["max_iterations"] = 5
        updated_state["max_queries_per_sub_question"] = 3
        updated_state["accumulated_findings"] = []
        updated_state["reflections"] = []
        updated_state["research_complete"] = False
        updated_state["research_step"] = "generate_rag_query"
        
        # Add AI message to messages
        if "messages" in updated_state:
            updated_state["messages"].append(AIMessage(content=f"I've planned our research approach for '{research_topic}'. We'll explore {len(sub_questions)} key areas to provide you with comprehensive information."))
        
        return updated_state
    
    async def generate_rag_query(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Generate a RAG query for the current sub-question.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state with RAG query
        """
        sub_questions = state.get("sub_questions", [])
        current_idx = state.get("current_sub_question_idx", 0)
        iteration_count = state.get("iteration_count", 0)
        accumulated_findings = state.get("accumulated_findings", [])
        
        if not sub_questions or current_idx >= len(sub_questions):
            # No sub-questions or all sub-questions completed
            updated_state = state.copy()
            updated_state["research_step"] = "finalize_report"
            return updated_state
        
        current_sub_question = sub_questions[current_idx]
        
        system_prompt = f"""You are a fitness research expert generating search queries.
        
TASK: Generate an effective search query for the current sub-question in our research.

CURRENT SUB-QUESTION: {current_sub_question}

ITERATION: {iteration_count + 1}

ACCUMULATED FINDINGS SO FAR:
{accumulated_findings}

OUTPUT REQUIREMENTS:
1. Generate ONE specific search query that will help answer the current sub-question
2. The query should be focused and use relevant fitness terminology
3. If this is a follow-up iteration, your query should build on previous findings and address gaps

Your output should be a single, focused search query without any additional explanation.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate a search query for: {current_sub_question}")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # Update state
        updated_state = state.copy()
        updated_state["current_rag_query"] = response.strip()
        updated_state["research_step"] = "execute_rag"
        
        return updated_state
    
    async def execute_rag_direct(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Execute a RAG query and retrieve results.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state with RAG results
        """
        query = state.get("current_rag_query", "")
        
        # In a real implementation, this would call a RAG system
        # For now, we'll simulate RAG results
        simulated_rag_results = [
            "Regular exercise has been shown to improve cardiovascular health by strengthening the heart muscle and improving blood flow.",
            "Studies indicate that a combination of strength training and cardio provides optimal fitness benefits for most individuals.",
            "Proper nutrition is essential for exercise recovery, with protein intake being particularly important for muscle repair and growth.",
            "Rest days are crucial for allowing muscles to recover and prevent overtraining syndrome, which can lead to decreased performance and increased injury risk."
        ]
        
        # Update state
        updated_state = state.copy()
        updated_state["rag_results"] = simulated_rag_results
        updated_state["research_step"] = "synthesize_results"
        
        return updated_state
    
    async def synthesize_rag_results(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Synthesize RAG results into coherent findings.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state with synthesized findings
        """
        sub_questions = state.get("sub_questions", [])
        current_idx = state.get("current_sub_question_idx", 0)
        current_sub_question = sub_questions[current_idx] if current_idx < len(sub_questions) else ""
        rag_results = state.get("rag_results", [])
        accumulated_findings = state.get("accumulated_findings", [])
        
        system_prompt = f"""You are a fitness research expert synthesizing search results.
        
TASK: Synthesize the search results into coherent findings that address the current sub-question.

CURRENT SUB-QUESTION: {current_sub_question}

SEARCH RESULTS:
{rag_results}

ACCUMULATED FINDINGS SO FAR:
{accumulated_findings}

OUTPUT REQUIREMENTS:
1. Synthesize the search results into 3-5 key findings
2. Ensure the findings directly address the current sub-question
3. Integrate new information with previous findings when relevant
4. Be specific and cite evidence from the search results

Your output should be a concise synthesis of the search results that advances our understanding of the sub-question.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Synthesize the search results for: {current_sub_question}")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # Update state
        updated_state = state.copy()
        updated_state["accumulated_findings"].append({
            "sub_question": current_sub_question,
            "findings": response
        })
        updated_state["research_step"] = "reflect_on_progress"
        
        return updated_state
    
    async def reflect_on_progress(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Reflect on research progress and determine next steps.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state with reflection and next steps
        """
        sub_questions = state.get("sub_questions", [])
        current_idx = state.get("current_sub_question_idx", 0)
        current_sub_question = sub_questions[current_idx] if current_idx < len(sub_questions) else ""
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 5)
        max_queries_per_sub_question = state.get("max_queries_per_sub_question", 3)
        accumulated_findings = state.get("accumulated_findings", [])
        
        system_prompt = f"""You are a fitness research expert reflecting on research progress.
        
TASK: Reflect on the current research progress and determine next steps.

CURRENT SUB-QUESTION: {current_sub_question}

ITERATION: {iteration_count + 1} of {max_iterations}

QUERIES PER SUB-QUESTION: {max_queries_per_sub_question}

ACCUMULATED FINDINGS:
{accumulated_findings}

OUTPUT REQUIREMENTS:
1. Assess how well the current sub-question has been addressed
2. Identify any gaps or areas that need further investigation
3. Recommend whether to:
   a. Continue researching the current sub-question (if gaps remain and iterations < max)
   b. Move to the next sub-question (if current is sufficiently addressed or iterations = max)
   c. Conclude research (if all sub-questions are addressed or iterations = max)

Your output should be a thoughtful reflection on the research progress and clear recommendation for next steps.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Reflect on our research progress for: {current_sub_question}")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # Update state
        updated_state = state.copy()
        updated_state["reflections"].append({
            "sub_question": current_sub_question,
            "reflection": response
        })
        
        # Determine next steps based on iteration count and completion
        queries_this_sub_question = state.get("queries_this_sub_question", 0) + 1
        updated_state["queries_this_sub_question"] = queries_this_sub_question
        updated_state["iteration_count"] = iteration_count + 1
        
        # Check if we should move to the next sub-question
        if "move to the next sub-question" in response.lower() or queries_this_sub_question >= max_queries_per_sub_question:
            updated_state["current_sub_question_idx"] = current_idx + 1
            updated_state["queries_this_sub_question"] = 0
            
            # Check if we've completed all sub-questions
            if current_idx + 1 >= len(sub_questions) or updated_state["iteration_count"] >= max_iterations:
                updated_state["research_step"] = "finalize_report"
            else:
                updated_state["research_step"] = "generate_rag_query"
        else:
            # Continue with current sub-question
            updated_state["research_step"] = "generate_rag_query"
        
        return updated_state
    
    async def finalize_research_report(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Finalize the research report.
        
        Args:
            state: The current state
            config: Optional runnable config
            
        Returns:
            Updated state with final research report
        """
        research_topic = state.get("research_topic", "")
        sub_questions = state.get("sub_questions", [])
        accumulated_findings = state.get("accumulated_findings", [])
        
        system_prompt = f"""You are a fitness research expert finalizing a comprehensive research report.
        
TASK: Create a comprehensive research report that synthesizes all findings from our investigation.

RESEARCH TOPIC: {research_topic}

SUB-QUESTIONS INVESTIGATED:
{sub_questions}

ACCUMULATED FINDINGS:
{accumulated_findings}

OUTPUT REQUIREMENTS:
1. Create a well-structured research report with clear sections
2. Begin with an executive summary of key findings
3. Address each sub-question with synthesized findings
4. Include practical recommendations based on the research
5. Use a professional, evidence-based tone
6. Format the report for readability with headings and bullet points where appropriate

Your output should be a complete, polished research report that provides valuable insights on the research topic.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create a final research report for: {research_topic}")
        ]
        
        response = await self.llm_provider.generate_response(messages)
        
        # Update state
        updated_state = state.copy()
        updated_state["final_report"] = response
        updated_state["research_complete"] = True
        updated_state["current_agent"] = "coordinator"
        
        # Add AI message to messages
        if "messages" in updated_state:
            updated_state["messages"].append(AIMessage(content=response))
        
        return updated_state