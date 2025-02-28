from evals.solvers.providers.openai.whisper_solver import TranscriptionSolver
import deepgram
import io
import os
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

class DeepgramSolver(TranscriptionSolver):
    def __init__(self, model_name="nova-3", **kwargs):
        super().__init__(**kwargs)
        self.client_options = None
        self.model_name = model_name

    @property
    def name(self) -> str:
        return "whisper"
    
    @property
    def model_version(self) -> str:
        return "whisper-1"
    
    def _transcribe(self, wav_bytes: bytes) -> str:
        if not self.client:
            self.client = deepgram.DeepgramClient(DEEPGRAM_API_KEY)
            self.client_options = deepgram.PrerecordedOptions(
                model=self.model_name,
                smart_format=True,
            )

        file = io.BytesIO(wav_bytes)

        payload: deepgram.FileSource = {
            "buffer": file
        }
        response = self.client.listen.rest.v("1").transcribe_file(payload, self.client_options)
        return response.results.channels[0].alternatives[0].transcript
