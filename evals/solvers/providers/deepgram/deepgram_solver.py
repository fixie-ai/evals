from evals.solvers.providers.openai.whisper_solver import TranscriptionSolver
import deepgram
import io
import os
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

class DeepgramSolver(TranscriptionSolver):
    def __init__(self, model_name="nova-3", detect_language=False, **kwargs):
        super().__init__(**kwargs)
        self.client_options = None
        self.model_name = model_name
        self.detect_language = detect_language

    @property
    def name(self) -> str:
        return "deepgram"
    
    @property
    def model_version(self) -> str:
        return self.model_name
    
    def _transcribe(self, wav_bytes: bytes) -> str:
        if not self.client:
            self.client = deepgram.DeepgramClient(DEEPGRAM_API_KEY)
            self.client_options = deepgram.PrerecordedOptions(
                model=self.model_name,
                smart_format=True,
                detect_language=self.detect_language,
            )

        file = io.BytesIO(wav_bytes)

        payload: deepgram.FileSource = {
            "buffer": file
        }
        response = self.client.listen.rest.v("1").transcribe_file(payload, self.client_options)
        return response.results.channels[0].alternatives[0].transcript
