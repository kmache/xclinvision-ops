"""Centralized API client for the XClinVision Streamlit frontend.

The Bridge: Handles all HTTP communication with the FastAPI backend.
Provides a single `requests.Session` with unified error handling,
timeouts, and logging — replacing scattered raw `requests` calls
across view files.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from config import (
    Endpoints,
    HEALTH_CHECK_TIMEOUT,
    INFERENCE_TIMEOUT,
    EXPLAIN_TIMEOUT,
    CHAT_TIMEOUT,
    FEEDBACK_TIMEOUT,
    REPORT_TIMEOUT,
    HISTORY_TIMEOUT,
    DRIFT_TIMEOUT,
    MODEL_CARD_TIMEOUT,
    FEEDBACK_STATS_TIMEOUT,
    EXPORT_REPORT_TIMEOUT,
    API_BASE_URL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("XClinVisionClient")


class XClinVisionClient:
    """Single HTTP client for all backend API calls.

    Reuses a `requests.Session` for connection pooling and provides
    consistent error handling across every endpoint.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        token = os.environ.get("XCLINVISION_API_TOKEN", "").strip()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
            logger.info("XClinVision API Client initialized (bearer token attached)")
        else:
            logger.warning(
                "XCLINVISION_API_TOKEN unset; protected v2 endpoints will return 401."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get(
        self,
        url: str,
        params: Optional[Dict] = None,
        timeout: float = 10.0,
    ) -> Optional[Any]:
        """Robust GET request with unified error handling."""
        try:
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.warning("Connection refused: %s", url)
            return None
        except requests.exceptions.Timeout:
            logger.warning("Request timed out: %s", url)
            return None
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP error (%s): %s", url, e)
            return None
        except Exception as e:
            logger.error("API error (%s): %s", url, e)
            return None

    def _post_json(
        self,
        url: str,
        json_data: Dict,
        timeout: float = 10.0,
    ) -> Optional[Any]:
        """Robust POST request (JSON body) with unified error handling."""
        try:
            response = self.session.post(url, json=json_data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.warning("Connection refused: %s", url)
            return None
        except requests.exceptions.Timeout:
            logger.warning("Request timed out: %s", url)
            return None
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP error (%s): %s", url, e)
            return None
        except Exception as e:
            logger.error("API error (%s): %s", url, e)
            return None

    def _post_multipart(
        self,
        url: str,
        files: Dict,
        data: Dict,
        timeout: float = 10.0,
    ) -> Optional[Any]:
        """Robust POST request (multipart form) with unified error handling."""
        try:
            response = self.session.post(
                url, files=files, data=data, timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.warning("Connection refused: %s", url)
            return None
        except requests.exceptions.Timeout:
            logger.warning("Request timed out: %s", url)
            return None
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP error (%s): %s", url, e)
            return None
        except Exception as e:
            logger.error("API error (%s): %s", url, e)
            return None

    # ==================================================================
    # 1. SYSTEM HEALTH
    # ==================================================================
    def check_health(self) -> bool:
        """Check if the backend API is alive.

        Returns True if backend responds with HTTP 200.
        """
        url = Endpoints.url(Endpoints.HEALTH)
        data = self._get(url, timeout=HEALTH_CHECK_TIMEOUT)
        return data is not None

    # ==================================================================
    # 2. INFERENCE & EXPLANATION  (page_inference)
    # ==================================================================
    def analyze_image(
        self,
        file_bytes: bytes,
        filename: str,
        patient_id: str,
        study_date: str,
        modality: str = "X-ray",
        body_part: str = "Chest",
        clinical_history: str = "",
        model_name: str = "convnext_small",
        xai_method: str = "gradcam++",
    ) -> Optional[Dict[str, Any]]:
        """Upload an image and run full AI analysis.

        Endpoint: POST /api/v2/analyze
        """
        url = Endpoints.url(Endpoints.ANALYZE)
        files = {"file": (filename, file_bytes, "image/jpeg")}
        data = {
            "patient_id": patient_id,
            "study_date": study_date,
            "modality": modality,
            "body_part": body_part,
            "clinical_history": clinical_history,
            "model_name": model_name,
            "xai_method": xai_method,
        }
        return self._post_multipart(url, files=files, data=data, timeout=INFERENCE_TIMEOUT)

    def get_explanation(
        self,
        analysis_id: str,
        method: str = "gradcam++",
        threshold: float = 0.5,
        opacity: float = 0.6,
        colormap: str = "jet",
        finding: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Regenerate XAI heatmap with adjustable parameters.

        Endpoint: GET /api/v2/explain/{analysis_id}
        """
        url = Endpoints.url(Endpoints.EXPLAIN, analysis_id=analysis_id)
        params: Dict[str, Any] = {
            "method": method, "threshold": threshold,
            "opacity": opacity, "colormap": colormap,
        }
        if finding:
            params["finding"] = finding
        return self._get(url, params=params, timeout=EXPLAIN_TIMEOUT)

    def send_chat_message(
        self,
        analysis_id: str,
        message: str,
        history: List[Dict[str, str]],
        context_type: str = "clinical",
    ) -> Optional[Dict[str, Any]]:
        """Send a message to the LLM clinical assistant.

        Endpoint: POST /api/v2/chat
        """
        url = Endpoints.url(Endpoints.CHAT)
        payload = {
            "analysis_id": analysis_id,
            "message": message,
            "history": history,
            "context_type": context_type,
        }
        return self._post_json(url, json_data=payload, timeout=CHAT_TIMEOUT)

    def submit_feedback(
        self,
        analysis_id: str,
        feedback_type: str,
        user_id: str = "anonymous",
        notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Submit clinician quality feedback.

        Endpoint: POST /api/v2/feedback
        """
        url = Endpoints.url(Endpoints.FEEDBACK)
        payload = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "feedback_type": feedback_type,
            "notes": notes,
        }
        return self._post_json(url, json_data=payload, timeout=FEEDBACK_TIMEOUT)

    # ==================================================================
    # 3. HISTORICAL COMPARISON  (page_history)
    # ==================================================================
    def compare_images(
        self,
        file_a_bytes: bytes,
        filename_a: str,
        file_b_bytes: bytes,
        filename_b: str,
        model_name: str = "convnext_small",
        xai_method: str = "gradcam++",
    ) -> Optional[Dict[str, Any]]:
        """Upload two images and return side-by-side analysis.

        Endpoint: POST /api/v2/compare
        """
        from config import COMPARE_TIMEOUT
        url = Endpoints.url(Endpoints.COMPARE)
        files = {
            "file_a": (filename_a, file_a_bytes, "image/jpeg"),
            "file_b": (filename_b, file_b_bytes, "image/jpeg"),
        }
        data = {"model_name": model_name, "xai_method": xai_method}
        return self._post_multipart(url, files=files, data=data, timeout=COMPARE_TIMEOUT)

    def get_patient_history(self, patient_id: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch a patient's historical analysis timeline.

        Endpoint: GET /api/v2/history/{patient_id}
        """
        url = Endpoints.url(Endpoints.HISTORY, patient_id=patient_id)
        return self._get(url, timeout=HISTORY_TIMEOUT)

    # ==================================================================
    # 4. REPORT GENERATION  (page_report)
    # ==================================================================
    def generate_report(
        self,
        analysis_ids: List[str],
        template: str = "structured_clinical",
        sections: Optional[List[str]] = None,
        language: str = "en",
        include_uncertainty: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Generate a structured clinical report.

        Endpoint: POST /api/v2/generate-report
        """
        url = Endpoints.url(Endpoints.GENERATE_REPORT)
        payload = {
            "analysis_ids": analysis_ids,
            "template": template,
            "sections": sections or ["findings", "impressions", "recommendations"],
            "language": language,
            "include_uncertainty": include_uncertainty,
        }
        return self._post_json(url, json_data=payload, timeout=REPORT_TIMEOUT)

    # ==================================================================
    # 5. AUDIT & TRANSPARENCY  (page_audit)
    # ==================================================================
    def get_model_card(self) -> Optional[Dict[str, Any]]:
        """Fetch model card and governance metadata.

        Endpoint: GET /api/v2/model-card
        """
        url = Endpoints.url(Endpoints.MODEL_CARD)
        return self._get(url, timeout=MODEL_CARD_TIMEOUT)

    def get_drift_metrics(self, days: int = 30) -> Optional[Dict[str, Any]]:
        """Fetch drift monitoring metrics.

        Endpoint: GET /api/v2/drift-metrics
        """
        url = Endpoints.url(Endpoints.DRIFT_METRICS)
        return self._get(url, params={"days": days}, timeout=DRIFT_TIMEOUT)

    def get_feedback_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch aggregated feedback statistics.

        Endpoint: GET /api/v2/feedback-stats
        """
        url = Endpoints.url(Endpoints.FEEDBACK_STATS)
        return self._get(url, timeout=FEEDBACK_STATS_TIMEOUT)

    def export_report_html(
        self,
        analysis_id: str,
        include_xai: bool = True,
        include_uncertainty: bool = True,
        indication: str = "",
        comments: str = "",
        conversation_log: list | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Export a self-contained HTML clinical report.

        Endpoint: POST /api/v2/export-report
        """
        url = Endpoints.url(Endpoints.EXPORT_REPORT)
        payload = {
            "analysis_id": analysis_id,
            "format": "html",
            "include_xai": include_xai,
            "include_uncertainty": include_uncertainty,
            "indication": indication,
            "comments": comments,
            "conversation_log": conversation_log or [],
        }
        return self._post_json(url, json_data=payload, timeout=EXPORT_REPORT_TIMEOUT)

    def export_report_pdf(
        self,
        analysis_id: str,
        include_xai: bool = True,
        include_uncertainty: bool = True,
        indication: str = "",
        comments: str = "",
        conversation_log: list | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Export a clinical report as PDF (base64-encoded).

        Endpoint: POST /api/v2/export-report
        """
        url = Endpoints.url(Endpoints.EXPORT_REPORT)
        payload = {
            "analysis_id": analysis_id,
            "format": "pdf",
            "include_xai": include_xai,
            "include_uncertainty": include_uncertainty,
            "indication": indication,
            "comments": comments,
            "conversation_log": conversation_log or [],
        }
        return self._post_json(url, json_data=payload, timeout=EXPORT_REPORT_TIMEOUT)

    def export_report_json(
        self,
        analysis_id: str,
        include_xai: bool = True,
        include_uncertainty: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Export a structured JSON clinical report.

        Endpoint: POST /api/v2/export-report
        """
        url = Endpoints.url(Endpoints.EXPORT_REPORT)
        payload = {
            "analysis_id": analysis_id,
            "format": "json",
            "include_xai": include_xai,
            "include_uncertainty": include_uncertainty,
        }
        return self._post_json(url, json_data=payload, timeout=EXPORT_REPORT_TIMEOUT)

    # ==================================================================
    # 6. LLM PROVIDER MANAGEMENT
    # ==================================================================
    def get_llm_providers(self) -> Optional[Dict[str, Any]]:
        """List available LLM providers and current active provider.

        Endpoint: GET /api/v2/llm/providers
        """
        url = Endpoints.url(Endpoints.LLM_PROVIDERS)
        return self._get(url, timeout=HEALTH_CHECK_TIMEOUT)

    def switch_llm_provider(self, provider: str) -> Optional[Dict[str, Any]]:
        """Switch the active LLM provider.

        Endpoint: POST /api/v2/llm/switch
        """
        url = Endpoints.url(Endpoints.LLM_SWITCH)
        try:
            response = self.session.post(
                url, data={"provider": provider}, timeout=HEALTH_CHECK_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to switch LLM provider: %s", e)
            return None

    def get_llm_health(self) -> Optional[Dict[str, Any]]:
        """Health check for all LLM providers.

        Endpoint: GET /api/v2/llm/health
        """
        url = Endpoints.url(Endpoints.LLM_HEALTH)
        return self._get(url, timeout=HEALTH_CHECK_TIMEOUT)

    # ==================================================================
    # 7. STREAMING CHAT
    # ==================================================================
    def stream_chat_message(
        self,
        analysis_id: str,
        message: str,
        history: List[Dict[str, str]],
        context_type: str = "clinical",
    ):
        """Stream a chat response via Server-Sent Events.

        Endpoint: POST /api/v2/chat/stream

        Yields parsed SSE events as dicts with keys:
        - ``event``: event type ("metadata", "token", "done")
        - ``data``: parsed JSON data

        Falls back to non-streaming :meth:`send_chat_message` on error.
        """
        import json

        url = Endpoints.url(Endpoints.CHAT_STREAM)
        payload = {
            "analysis_id": analysis_id,
            "message": message,
            "history": history,
            "context_type": context_type,
        }

        try:
            response = self.session.post(
                url, json=payload, timeout=CHAT_TIMEOUT, stream=True,
            )
            response.raise_for_status()

            event_type = "message"
            data_lines: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if line == "":
                    # Empty line = end of SSE frame → dispatch accumulated data
                    if data_lines:
                        data_str = "\n".join(data_lines)
                        data_lines = []
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            data = {"raw": data_str}
                        yield {"event": event_type, "data": data}
                        event_type = "message"
                    continue
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
                elif line.startswith("data:"):
                    data_lines.append(line[5:])
        except Exception as e:
            logger.warning("Streaming chat failed (%s), falling back to sync", e)
            result = self.send_chat_message(
                analysis_id=analysis_id,
                message=message,
                history=history,
                context_type=context_type,
            )
            if result:
                yield {
                    "event": "token",
                    "data": {"token": result.get("response", "")},
                }
                yield {"event": "done", "data": {"status": "done", "fallback": True}}
