"""LLM-based Clinical Decision Support Agent with RAG."""

from typing import Dict, List, Optional
import os
from dataclasses import dataclass

try:
    from langchain import OpenAI, LLMChain, PromptTemplate
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
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
        
        if not LANGCHAIN_AVAILABLE:
            print("Warning: LangChain not available. Using mock responses.")
            
        self._init_llm()
        self._init_rag()
        
    def _init_llm(self):
        """Initialize LLM client."""
        if LANGCHAIN_AVAILABLE:
            if self.llm_provider == "openai":
                from langchain.chat_models import ChatOpenAI
                self.llm = ChatOpenAI(
                    model_name=self.model,
                    temperature=self.temperature,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                
    def _init_rag(self):
        """Initialize RAG components."""
        if LANGCHAIN_AVAILABLE and self.vector_db_path:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=embeddings,
            )
            
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            )
        else:
            self.vector_store = None
            self.retriever = None
            
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
            docs = self.retriever.get_relevant_documents(query)
            retrieved_context = "\n\n".join([d.page_content for d in docs[:3]])
            
        # Generate response
        if LANGCHAIN_AVAILABLE:
            full_prompt = f"{system_prompt}\n\n{retrieved_context}\n\n{user_prompt}"
            
            try:
                response = self.llm.predict(full_prompt)
                structured_output = self._parse_response(response)
            except Exception as e:
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
- Class Probabilities: Normal={context.probabilities[0]:.1%}, Pneumonia={context.probabilities[1]:.1%}, TB={context.probabilities[2]:.1%}
"""
        
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
        import json
        
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
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
        
        if context.prediction in ["Pneumonia", "Tuberculosis"]:
            recommendation += "Consider additional imaging (CT) and laboratory tests if clinically indicated."
        else:
            recommendation += "Follow standard clinical protocols for patient management."
            
        return {
            "findings": findings,
            "impression": impression,
            "uncertainty": uncertainty_note,
            "recommendation": recommendation,
            "error": error_msg,
        }


def create_agent(
    llm_provider: str = "openai",
    model: str = "gpt-4",
    vector_db_path: Optional[str] = None,
) -> ClinicalDecisionSupportAgent:
    """Factory function to create a clinical decision support agent."""
    return ClinicalDecisionSupportAgent(
        llm_provider=llm_provider,
        model=model,
        vector_db_path=vector_db_path,
    )
