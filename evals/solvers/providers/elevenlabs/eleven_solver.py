import io
import os

from elevenlabs import ElevenLabs

from evals.solvers.providers.openai.whisper_solver import TranscriptionSolver

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

class ElevenSolver(TranscriptionSolver):
    def __init__(self, language_code: str = "eng", **kwargs):
        super().__init__(**kwargs)
        self.language_code = language_code
        
    @property
    def name(self) -> str:
        return "whisper"
    
    @property
    def model_version(self) -> str:
        return "whisper-1"
    
    def _transcribe(self, wav_bytes: bytes) -> str:
        if not self.client:
            self.client = ElevenLabs(
                api_key=ELEVENLABS_API_KEY,
            )

        file = io.BytesIO(wav_bytes)
        response = self.client.speech_to_text.convert(
            model_id="scribe_v1",
            file=file,
            language_code=self.language_code,
        )
        return response.text
