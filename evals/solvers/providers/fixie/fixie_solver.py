import re
from typing import Any, Optional

from evals.solvers.providers.openai.third_party_solver import ThirdPartySolver

# Matches <think>...</think> block (including newlines) at the start of a response
_THINK_PATTERN = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)


class FixieSolver(ThirdPartySolver):
    AUDIO_PLACEHOLDER = "<|audio|>"

    def __init__(self, api_base: Optional[str] = None, strip_thinking: bool = False, **kwargs):
        self.strip_thinking = strip_thinking
        super().__init__(api_base or "https://api.ultravox.ai/api/", "ULTRAVOX_API_KEY", **kwargs)

    def _make_api_request(self, msgs, is_chat_model: bool, **kwargs) -> tuple[Any, str]:
        completion_result, completion_output = super()._make_api_request(msgs, is_chat_model, **kwargs)
        if self.strip_thinking and isinstance(completion_output, str):
            completion_output = _THINK_PATTERN.sub("", completion_output)
        return completion_result, completion_output

    def _process_msgs(self, raw_msgs: list[dict[str, str]]):
        replaced_messages = []
        for msg in raw_msgs:
            if isinstance(msg.get("content"), list):
                # Use explicit placeholders instead of letting vLLM put them at the top of the message.
                non_text_fragments = [content for content in msg["content"] if content.get("type") != "text"]
                text_fragments = [
                    {
                        "type": "text",
                        "text": self.AUDIO_PLACEHOLDER,
                    }
                    if content.get("type") == "audio_url"
                    else content
                    if content.get("type") == "text"
                    else {"type": "text", "text": ""}
                    for content in msg["content"]
                ]
                combined_text = "".join([content["text"] for content in text_fragments])
                replaced_messages.append(
                    {
                        **msg,
                        "content": non_text_fragments + [{"type": "text", "text": combined_text}],
                    }
                )
            else:
                replaced_messages.append(msg)

        return super()._process_msgs(replaced_messages)
