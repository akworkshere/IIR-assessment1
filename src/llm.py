import ollama
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)


class LLMClient:

    def __init__(
        self,
        model_name: str = "phi3:mini",
        temperature: float = 0.1,
        max_tokens: int = 512,
        base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

        logger.info(f"Initialized LLM client: {model_name} (temp={temperature})")

        try:
            self._verify_connection()
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            raise ConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                "Please ensure Ollama is running and the model is installed."
            )

    def _verify_connection(self) -> None:
        try:
            models = ollama.list()

            model_names = [m['name'] for m in models.get('models', [])]

            if not any(self.model_name in name for name in model_names):
                logger.warning(
                    f"Model {self.model_name} not found. Available models: {model_names}"
                )
                raise ValueError(
                    f"Model {self.model_name} not found. "
                    f"Please run: ollama pull {self.model_name}"
                )

            logger.info(f"Successfully connected to Ollama. Model {self.model_name} is available.")

        except Exception as e:
            logger.error(f"Ollama connection error: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        try:
            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            messages.append({
                "role": "user",
                "content": prompt
            })

            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )

            answer = response['message']['content']

            logger.debug(f"Generated response ({len(answer)} chars)")

            return answer

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        try:
            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            messages.append({
                "role": "user",
                "content": prompt
            })

            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )

            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']

        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            raise

    def is_available(self) -> bool:
        try:
            self._verify_connection()
            return True
        except:
            return False


_global_llm_client = None


def get_llm_client(
    model_name: str = "phi3:mini",
    temperature: float = 0.1,
    max_tokens: int = 512
) -> LLMClient:
    global _global_llm_client

    if _global_llm_client is None:
        _global_llm_client = LLMClient(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

    return _global_llm_client


def generate_response(prompt: str, system_prompt: Optional[str] = None) -> str:
    client = get_llm_client()
    return client.generate(prompt, system_prompt)


def check_ollama_status() -> dict:
    try:
        models = ollama.list()
        return {
            "status": "online",
            "models": [m['name'] for m in models.get('models', [])]
        }
    except Exception as e:
        return {
            "status": "offline",
            "error": str(e)
        }
