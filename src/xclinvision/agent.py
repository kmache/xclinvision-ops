"""LLM-based Clinical Decision Support Agent with RAG."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Qdrant
    from qdrant_client import QdrantClient
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@dataclass
class ClinicalContext:
    """Structured clinical context for LLM input."""
    prediction: str
    probabilities: List[float]
    confidence: float
    uncertainty_level: str
    highlighted_regions: List[str]
    # Optional — if supplied, used to label each probability; works for any number of classes
    class_names: Optional[List[str]] = None
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


class ClinicalDecisionSupportAgent:
    """RAG-enhanced LLM agent for clinical decision support."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.3,
        vector_db_path: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.vector_db_path = vector_db_path
        self.llm = None
        self.retriever = None
        self.vector_store = None

        if not LANGCHAIN_AVAILABLE:
            logger.warning(
                "LangChain / Qdrant packages not available. "
                "Using rule-based fallback responses."
            )
            return

        self._init_llm()
        self._init_rag()
        
    def _init_llm(self):
        """Initialize LLM client."""
        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Set it before instantiating ClinicalDecisionSupportAgent."
                )
            self.llm = ChatOpenAI(
                model_name=self.model,
                temperature=self.temperature,
                api_key=api_key,
            )
            logger.info("LLM initialised: provider=%s model=%s", self.llm_provider, self.model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                
    def _init_rag(self):
        """Initialize RAG components backed by Qdrant."""
        if not self.vector_db_path:
            logger.info("No vector_db_path provided — RAG disabled.")
            return

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # QdrantClient with a local path persists to disk.
        # Pass url="http://localhost:6333" for a remote Qdrant server.
        client = QdrantClient(path=self.vector_db_path)
        self.vector_store = Qdrant(
            client=client,
            collection_name="clinical_guidelines",
            embeddings=embeddings,
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        logger.info("Qdrant RAG initialised from '%s'.", self.vector_db_path)
            
    def generate_report(
        self,
        context: ClinicalContext,
        model_metadata: Optional[Dict] = None,
    ) -> Dict:
        """Generate structured clinical report."""
        
        system_prompt = """You are a clinical decision-support assistant for chest X-ray interpretation. 
You do not diagnose disease. You summarize AI findings, explain uncertainty, and encourage clinical correlation. 
Use neutral medical language. Always emphasize that this is AI-assisted analysis requiring clinician review.

Guidelines:
- Explain findings in neutral medical language
- List possible interpretations and emphasize uncertainty
- Recommend clinical correlation or further tests when appropriate
- Never provide definitive diagnoses
- Include appropriate caveats and limitations"""

        user_prompt = self._build_user_prompt(context, model_metadata)
        
        # Retrieve relevant clinical guidelines if RAG available
        retrieved_context = ""
        if self.retriever:
            query = f"{context.prediction} chest X-ray findings {', '.join(context.highlighted_regions)}"
            docs = self.retriever.invoke(query)
            retrieved_context = "\n\n".join([d.page_content for d in docs[:3]])
            
        # Generate response
        if LANGCHAIN_AVAILABLE and self.llm is not None:
            full_prompt = f"{system_prompt}\n\n{retrieved_context}\n\n{user_prompt}"
            
            try:
                response = self.llm.invoke(full_prompt).content
                structured_output = self._parse_response(response)
            except Exception as e:
                logger.exception("LLM call failed: %s", e)
                structured_output = self._generate_fallback_response(context, str(e))
        else:
            structured_output = self._generate_fallback_response(context)
            
        return structured_output
        
    def _build_user_prompt(
        self,
        context: ClinicalContext,
        model_metadata: Optional[Dict],
    ) -> str:
        """Build user prompt from clinical context."""
        
        prompt = f"""AI Analysis Results:
- Prediction: {context.prediction}
- Confidence: {context.confidence:.1%}
- Uncertainty Level: {context.uncertainty_level}
- Highlighted Regions: {', '.join(context.highlighted_regions) if context.highlighted_regions else 'None specifically'}
"""
        # Build class probabilities dynamically — works for any num_classes
        names = context.class_names or [f"Class {i}" for i in range(len(context.probabilities))]
        prob_str = ", ".join(f"{n}={p:.1%}" for n, p in zip(names, context.probabilities))
        prompt += f"- Class Probabilities: {prob_str}\n"
        
        if context.patient_age:
            prompt += f"- Patient Age: {context.patient_age}\n"
        if context.patient_sex:
            prompt += f"- Patient Sex: {context.patient_sex}\n"
            
        if model_metadata:
            prompt += f"\nModel Information:\n"
            prompt += f"- Version: {model_metadata.get('version', 'unknown')}\n"
            prompt += f"- ECE Score: {model_metadata.get('ece', 'N/A')}\n"
            
        prompt += """
Please generate a structured clinical summary with the following sections:
1. Findings - Describe what the AI detected
2. Impression - Summarize the findings with appropriate uncertainty
3. Uncertainty - Explain the uncertainty level and its implications
4. Recommendation - Suggest next steps for clinical correlation

Format the response as a JSON object with these keys."""

        return prompt
        
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured format."""
        
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
            
        # Fallback: return as text
        return {
            "findings": response,
            "impression": "See findings above",
            "uncertainty": "Please review AI limitations",
            "recommendation": "Clinical correlation required",
        }
        
    def _generate_fallback_response(
        self,
        context: ClinicalContext,
        error_msg: Optional[str] = None,
    ) -> Dict:
        """Generate fallback response when LLM is unavailable."""
        
        findings = f"AI analysis indicates {context.prediction} with {context.confidence:.1%} confidence."
        
        if context.highlighted_regions:
            findings += f" Key regions: {', '.join(context.highlighted_regions)}."
            
        impression = f"{context.prediction} suggested by AI"
        
        if context.uncertainty_level == "high":
            impression += " (high uncertainty - requires careful review)"
        elif context.uncertainty_level == "medium":
            impression += " (moderate confidence)"
        else:
            impression += " (AI model relatively confident)"
            
        uncertainty_note = f"Uncertainty level: {context.uncertainty_level}. "
        uncertainty_note += "AI models can make errors. Clinical judgment is essential."
        
        recommendation = "Clinical correlation with patient history, symptoms, and physical examination is essential. "
        
        if context.prediction in ["Pneumonia", "Cardiomegaly"]:
            recommendation += "Consider additional imaging (CT) and laboratory tests if clinically indicated."
        else:
            recommendation += "Follow standard clinical protocols for patient management."
            
        result: Dict = {
            "findings": findings,
            "impression": impression,
            "uncertainty": uncertainty_note,
            "recommendation": recommendation,
        }
        if error_msg is not None:
            result["error"] = error_msg
        return result


def create_agent(
    llm_provider: str = "openai",
    model: str = "gpt-4",
    temperature: float = 0.3,
    vector_db_path: Optional[str] = None,
) -> ClinicalDecisionSupportAgent:
    """Factory function to create a clinical decision support agent."""
    return ClinicalDecisionSupportAgent(
        llm_provider=llm_provider,
        model=model,
        temperature=temperature,
        vector_db_path=vector_db_path,
    )
