import io
import os
from typing import Optional
from elevenlabs import ElevenLabs

from evals.solvers.providers.openai.whisper_solver import TranscriptionSolver

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

class ElevenSolver(TranscriptionSolver):
    def __init__(self, language_code: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.language_code = language_code if language_code else None

    @property
    def name(self) -> str:
        return "scribe"
    
    @property
    def model_version(self) -> str:
        return "scribe-v1"
    
    def _transcribe(self, wav_bytes: bytes) -> str:
        if not self.client:
            self.client = ElevenLabs(
                api_key=ELEVENLABS_API_KEY,
            )

        file = io.BytesIO(wav_bytes)
        if self.language_code:
            # elevenlabs doesn't seem to work with language_code=None
            response = self.client.speech_to_text.convert(
                model_id="scribe_v1",
                file=file,
                language_code=self.language_code
            )
        else:
            response = self.client.speech_to_text.convert(
                model_id="scribe_v1",
                file=file,
            )
        return response.text
