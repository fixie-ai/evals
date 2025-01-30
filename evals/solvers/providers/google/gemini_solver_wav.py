import os
from dataclasses import asdict, dataclass

import google.api_core.exceptions
import google.generativeai as genai
from google.generativeai.client import get_default_generative_client

import numpy as np

from evals.solvers.solver import Solver, SolverResult
from evals.solvers.utils import data_url_to_wav
from evals.task_state import Message, TaskState
from evals.utils.api_utils import create_retrying

from evals.solvers.providers.google.gemini_solver import SAFETY_SETTINGS, GEMINI_RETRY_EXCEPTIONS, GeminiSolver

# Load API key from environment variable
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# TODO: Could we just use google's own types?
# e.g. google.generativeai.types.content_types.ContentType
@dataclass
class GoogleMessage:
    role: str
    parts: list[str]

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_evals_message(msg: Message):
        valid_roles = {"user", "model"}
        to_google_role = {
            "system": "user",  # Google doesn't have a "system" role
            "user": "user",
            "assistant": "model",
        }
        gmsg = GoogleMessage(
            role=to_google_role.get(msg.role, msg.role),
            parts=[msg.content] if isinstance(msg.content, str) else msg.content,
        )
        assert gmsg.role in valid_roles, f"Invalid role: {gmsg.role}"
        return gmsg
    
    

class GeminiSolverWav(GeminiSolver):
    """
    A solver class that uses Google's Gemini API to generate responses.
    """

    def _solve(
        self,
        task_state: TaskState,
        **kwargs,
    ) -> SolverResult:
        # Uncomment if you want to add a system message. 
        # Gemini doesn't have a system role, so we will eventually convert this to a user message. 
        # It seems to degrade performance on some models. 
        # msgs = [
        #     Message(role="system", content=task_state.task_description),
        # ] + task_state.messages
        msgs = task_state.messages
        gmsgs = self._convert_msgs_to_google_format(msgs)
        try:
            glm_model = genai.GenerativeModel(model_name=self.model_name)
            glm_model._client = self.glm_client
            gen_content_resp = create_retrying(
                glm_model.generate_content,
                retry_exceptions=GEMINI_RETRY_EXCEPTIONS,
                **{
                    "contents": gmsgs,
                    "generation_config": self.gen_config,
                    "safety_settings": SAFETY_SETTINGS,
                },
            )
            
            if gen_content_resp.prompt_feedback.block_reason:
                # Blocked by safety filters
                solver_result = SolverResult(
                    str(gen_content_resp.prompt_feedback),
                    error=gen_content_resp.prompt_feedback,
                )
            else:
                # Get text response
                solver_result = SolverResult(
                    gen_content_resp.text,
                    error=gen_content_resp.prompt_feedback,
                )
        except (google.api_core.exceptions.GoogleAPIError,) as e:
            solver_result = SolverResult(
                e.message,
                error=e,
            )
        except ValueError as e:
            # TODO: Why does this error ever occur and how can we handle it better?
            # (See google/generativeai/types/generation_types.py for the triggers)
            known_errors = [
                "The `response.text` quick accessor",
                "The `response.parts` quick accessor",
            ]
            if any(err in str(e) for err in known_errors):
                solver_result = SolverResult(
                    str(e),
                    error=e,
                )
            else:
                raise e

        return solver_result

    @staticmethod
    def _convert_msgs_to_google_format(msgs: list[Message]) -> list[GoogleMessage]:
        """
        Convert messages to Gemini API format and process audio data.
        Currently only supports a single audio message at the end of the conversation.

        Args:
            msgs: List of Message objects. The last message must contain audio data.

        Raises:
            AssertionError: If messages after audio are found or if audio format is invalid.

        Example msgs:
        [Message(role='user', content='You are a helpful assistant.'), 
         Message(role='user', 
                 content=[{'type': 'text', 'text': 'Please translate...'}, 
                         {'type': 'audio_url', 'audio_url': {'url': 'data:audio/x-wav;base64,...'}}])]
        """
        # Add validation for audio message placement
        assert len(msgs) >= 1, "Must provide at least one message"
        last_msg_content = msgs[-1].content

        # Validate message has exactly one text and one audio part
        content_types = [part.get('type') for part in last_msg_content]
        assert content_types.count('text') == content_types.count('audio_url') == 1, (
            "Last message must contain exactly one text and one audio_url content"
        )

        # Make sure audio_url is the last part
        if content_types[-1] != 'audio_url':
            # Swap text and audio parts to ensure audio is last
            text_idx, audio_idx = content_types.index('text'), content_types.index('audio_url')
            last_msg_content[text_idx], last_msg_content[audio_idx] = last_msg_content[audio_idx], last_msg_content[text_idx]

        std_msgs = []
        for msg in msgs:
            gmsg = GoogleMessage.from_evals_message(msg)
            assert gmsg.role in {"user", "model"}, f"Invalid role: {gmsg.role}"
            
            if std_msgs and gmsg.role == std_msgs[-1].role:
                # Combine text content
                if isinstance(gmsg.parts[0], str):
                    std_msgs[-1].parts = ["\n".join(std_msgs[-1].parts + gmsg.parts)]
                # Combine text and preserve audio data
                elif isinstance(gmsg.parts[0], dict):
                    std_msgs[-1].parts = [{
                        "type": "text",
                        "text": std_msgs[-1].parts[0] + "\n" + gmsg.parts[0]["text"]
                    }, gmsg.parts[1]]
            else:
                std_msgs.append(gmsg)

        # Process final audio message
        last_msg = std_msgs[-1].parts
        audio_part = last_msg[1]
        
        # Convert audio to wav format
        wav_data = {
            "mime_type": "audio/wav",
            "data": data_url_to_wav(audio_part["audio_url"]["url"])
        }

        return [last_msg[0]['text'], wav_data]
