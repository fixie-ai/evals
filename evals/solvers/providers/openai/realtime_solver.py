import asyncio
import base64
import io
import json
import os
from typing import Any, Dict, Optional, Union

import librosa
import numpy as np
import websockets

from evals.solvers.solver import Solver, SolverResult
from evals.solvers.utils import data_url_to_wav
from evals.task_state import TaskState


def _wav_to_24k_pcm(wav_bytes):
    audio, sr = librosa.load(io.BytesIO(wav_bytes), sr=24000, mono=True)
    return (audio * 32767).astype(np.int16).tobytes()


def _pcm_to_base64(pcm_bytes):
    return base64.b64encode(pcm_bytes).decode("utf-8")


class RealtimeSolver(Solver):
    """ """

    def __init__(
        self,
        completion_fn_options: Dict[str, Any],
        postprocessors: list[str] = [],
        registry: Any = None,
    ):
        super().__init__(postprocessors=postprocessors)
        if "model" not in completion_fn_options:
            raise ValueError("OpenAISolver requires a model to be specified.")
        self.completion_fn_options = completion_fn_options

    @property
    def model(self) -> str:
        """
        Get model name from completion function, e.g. "gpt-3.5-turbo"
        This may not always include the full model version, e.g. "gpt-3.5-turbo-0613"
        so use `self.model_version` if you need the exact snapshot.
        """
        return self.completion_fn_options["model"]

    @property
    def name(self) -> str:
        return self.model

    @property
    def model_version(self) -> Union[str, dict]:
        return self.model

    @property
    def _api_base(self) -> Optional[str]:
        """The base URL for the API"""
        return "wss://api.openai.com/v1/realtime"

    @property
    def _api_key(self) -> Optional[str]:
        """The API key to use for the API"""
        return os.getenv("OPENAI_API_KEY")

    def _solve(self, task_state: TaskState, **kwargs) -> SolverResult:
        raw_msgs = [
            {"role": "system", "content": task_state.task_description},
        ] + [msg.to_dict() for msg in task_state.messages]
        completion_result = asyncio.run(self._ws_completion(raw_msgs))
        completion_content = completion_result["output"][0]["content"]
        completion_item = completion_content[0]
        # When the model yields text output, the text portion is contained in "text";
        # when the model yields audio output, the text portion is contained in "transcript".
        completion_output = completion_item.get("text") or completion_item.get("transcript")
        solver_result = SolverResult(completion_output, raw_completion_result=completion_result)
        return solver_result

    async def _ws_completion(self, messages):
        # When supplying audio input, the API will hang unless you ask for audio output,
        # even if you don't plan to use the audio output. So we ask for audio output
        # anytime we have audio content.
        url = f"{self._api_base}?model={self.model}"
        headers = {"Authorization": f"Bearer {self._api_key}", "OpenAI-Beta": "realtime=v1"}
        async with websockets.connect(url, additional_headers=headers) as websocket:
            system_message = None
            modalities = set(["text"])
            for message in messages:
                message_content_type = type(message["content"])
                role = message["role"]
                content = []
                if message_content_type is str:
                    text = message["content"]
                    if role == "system":
                        system_message = text
                    else:
                        content.append({"type": "input_text", "text": text})
                elif message_content_type is list:
                    for item in message["content"]:
                        if item["type"] == "text":
                            content.append({"type": "input_text", "text": item["text"]})
                        elif item["type"] == "audio_url":
                            wav_bytes = data_url_to_wav(item["audio_url"]["url"])
                            pcm_bytes = _wav_to_24k_pcm(wav_bytes)
                            base64_pcm = _pcm_to_base64(pcm_bytes)
                            content.append({"type": "input_audio", "audio": base64_pcm})
                            modalities.add("audio")
                if content:  # Only send if we have content
                    event = {
                        "type": "conversation.item.create",
                        "item": {"type": "message", "role": role, "content": content},
                    }
                    await websocket.send(json.dumps(event))
                        
            create_response = {
                "type": "response.create",
                "response": {
                    "instructions": system_message,
                    "modalities": list(modalities),
                },
            }
            await websocket.send(json.dumps(create_response))
            while True:
                try:
                    response_json = await websocket.recv()
                    response = json.loads(response_json)
                    if response["type"] == "response.done":
                        return response["response"]
                except websockets.exceptions.ConnectionClosed as e:
                    raise RuntimeError(f"WebSocket connection closed unexpectedly: {e}")
