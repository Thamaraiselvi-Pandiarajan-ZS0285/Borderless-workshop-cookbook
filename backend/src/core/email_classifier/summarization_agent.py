import logging
from datetime import datetime
from typing import Dict

from backend.src.config.dev_config import SUMMARIZATION_AGENT_NAME
from backend.src.controller.request_handler.email_request import RFPContext, DimensionalSummary
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.prompts.summarization_prompt import SUMMARIZATION_PROMPT, PERSPECTIVE_PROMPT_VARIANTS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SummarizationAgent:
    """
    SummarizationAgent handles the task of summarizing input text using an AI assistant.

    Attributes:
        client: The OpenAI client for chat completion.
        base_agent: BaseAgent object to create assistant agents.
    """

    def __init__(self):
        """
        Initializes the summarization agent with the OpenAI client and base agent.
        """
        try:
            self.client = OpenAiClient()
            self.base_agent = BaseAgent(self.client.open_ai_chat_completion_client)
            self.dimension_prompts = self._initialize_dimension_prompts()

            logger.info("SummarizationAgent initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize SummarizationAgent.")
            raise RuntimeError(f"Initialization failed: {e}")

    def _initialize_dimension_prompts(self) -> Dict[str,str]:
        return PERSPECTIVE_PROMPT_VARIANTS

    async def summarize_text(self, text: str, task_prompt: str = "") -> str:
        """
        Summarizes the given input text using an LLM agent.

        Args:
            text (str): The input text to be summarized.
            task_prompt (str, optional): An optional task-specific instruction to customize the summary.

        Returns:
            str: The summarized version of the input text.

        Raises:
            RuntimeError: If the summarization process fails.
        """
        if not text or not isinstance(text, str):
            logger.error("Invalid text input for summarization.")
            raise ValueError("Text input must be a non-empty string.")

        try:
            prompt = f"{SUMMARIZATION_PROMPT}\n\n{task_prompt}" if task_prompt else SUMMARIZATION_PROMPT
            summarization_agent = self.base_agent.create_assistant_agent(
                name=SUMMARIZATION_AGENT_NAME,
                prompt=prompt
            )

            logger.info("Summarization task started.")
            response = await summarization_agent.run(task=text)

            # Handle both dict and non-dict response formats
            if isinstance(response, dict):
                result = response.get("content", "")
            elif hasattr(response, 'messages'):  # Handle OpenAI-style response
                result = response.messages[-1].content if response.messages else ""
            else:
                result = str(response).strip()

            logger.info("Summarization completed successfully.")
            return result

        except Exception as e:
            logger.exception("Summarization failed.")
            raise RuntimeError(f"Summarization failed: {e}")

    async def analyze_rfp_multidimensional(self, rfp_context: RFPContext) -> Dict[str, DimensionalSummary]:
        """
        Perform multidimensional analysis of RFP content

        Args:
            rfp_context: RFP context containing all relevant content

        Returns:
            Dictionary of dimensional summaries
        """
        results = {}

        # Combine all content sources
        full_content = f"""
           Email Content: {rfp_context.raw_content}

           Attachments Summary: {rfp_context.attachments_content}
           """

        for dimension, prompt in self.dimension_prompts.items():
            try:
                # Create dimension-specific agent
                dimension_agent = self.base_agent.create_assistant_agent(
                    name=f"MarketResearch_{dimension.upper()}_Agent",
                    prompt=f"""
                            You are a market research analysis specialist focusing on {dimension} analysis.

                            {prompt}

                            Response format: Provide a concise 2-3 sentence summary focusing on key insights.
                            Maximum length: 75 tokens.
                            """
                )

                # Generate dimensional summary
                response = await dimension_agent.run(task=full_content)

                # Extract content from response
                if isinstance(response, dict):
                    summary_text = response.get("content", "")
                elif hasattr(response, 'messages'):
                    summary_text = response.messages[-1].content if response.messages else ""
                else:
                    summary_text = str(response).strip()

                # Calculate confidence score (basic implementation)
                confidence_score = self._calculate_confidence_score(summary_text, full_content)

                results[dimension] = DimensionalSummary(
                    dimension=dimension,
                    summary=summary_text,
                    confidence_score=confidence_score
                )

                logger.info(f"Completed analysis for dimension: {dimension}")

            except Exception as e:
                logger.error(f"Error analyzing dimension {dimension}: {str(e)}")
                results[dimension] = DimensionalSummary(
                    dimension=dimension,
                    summary=f"Error analyzing {dimension}",
                    confidence_score=0.0
                )
        return results

    def _calculate_confidence_score(self, summary: str, full_content: str) -> float:
        """Calculate confidence score for the summary"""
        if not summary or len(summary.strip()) < 10:
            return 0.0

        # Simple confidence calculation based on content coverage
        summary_words = set(summary.lower().split())
        content_words = set(full_content.lower().split())

        if len(content_words) == 0:
            return 0.0

        overlap = len(summary_words.intersection(content_words))
        coverage = overlap / min(len(summary_words), 20)  # Normalize by reasonable summary length

        return min(coverage, 1.0)

    async def _create_master_aggregation_summary(self, dimensional_summaries: Dict[str, DimensionalSummary]) -> str:
        """
        Create a master summary that aggregates all dimensional insights into a cohesive narrative
        while ensuring no data leakage
        """
        try:
            # Combine all high-confidence dimensional summaries
            high_quality_summaries = []
            for dim, summary in dimensional_summaries.items():
                if summary.confidence_score > 0.3 and summary.summary and len(summary.summary.strip()) > 15:
                    high_quality_summaries.append(f"{dim.replace('_', ' ').title()}: {summary.summary}")

            if not high_quality_summaries:
                return "Insufficient quality data for comprehensive analysis."

            # Create aggregation prompt
            aggregation_prompt = """
            You are a senior market research analyst creating a comprehensive RFP summary.

            Your task: Synthesize the dimensional analyses below into ONE cohesive, executive-level summary 
            that captures the complete picture of this market research opportunity.

            CRITICAL REQUIREMENTS:
            1. Create a flowing narrative that connects all dimensions naturally
            2. Highlight the most significant aspects of this RFP opportunity
            3. Focus on what makes this project unique or notable
            4. Maximum length: 150 tokens
            5. Write in professional, executive summary style

            Structure your response as a single paragraph that tells the complete story of this RFP with all the details
            """

            # Create aggregation agent
            aggregation_agent = self.base_agent.create_assistant_agent(
                name="RFP_Master_Aggregation_Agent",
                prompt=aggregation_prompt
            )

            # Combine dimensional summaries for input
            combined_dimensional_input = "\n\n".join(high_quality_summaries)

            # Generate master summary
            response = await aggregation_agent.run(task=combined_dimensional_input)

            # Extract content from response
            if isinstance(response, dict):
                master_summary = response.get("content", "")
            elif hasattr(response, 'messages'):
                master_summary = response.messages[-1].content if response.messages else ""
            else:
                master_summary = str(response).strip()

            return master_summary if master_summary else "Unable to generate master summary"

        except Exception as e:
            logger.error(f"Error creating master aggregation summary: {str(e)}")
            return "Error generating master summary"