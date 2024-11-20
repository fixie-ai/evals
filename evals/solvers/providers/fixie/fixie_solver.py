from typing import Optional

from evals.solvers.providers.openai.third_party_solver import ThirdPartySolver


class FixieSolver(ThirdPartySolver):
    AUDIO_PLACEHOLDER = "<|reserved_special_token_0|>"

    def __init__(self, api_base: Optional[str] = None, **kwargs):
        super().__init__(api_base or "https://ultravox.api.fixie.ai/v1", "ULTRAVOX_API_KEY", **kwargs)

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
