"""Interface for interacting with Ollama models."""
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from config import Config


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.ollama.base_url.rstrip('/')
        self.timeout = config.ollama.timeout
        self.logger = logging.getLogger(__name__)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        for text in texts:
            try:
                response = self._post('/api/embeddings', {
                    'model': self.config.ollama.embedding_model,
                    'prompt': text
                })
                
                if 'embedding' in response:
                    embeddings.append(response['embedding'])
                else:
                    self.logger.error(f"No embedding in response: {response}")
                    embeddings.append([0.0] * 768)  # Fallback empty embedding
                    
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for text: {e}")
                embeddings.append([0.0] * 768)  # Fallback empty embedding
        
        return embeddings
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * 768
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Generate a chat completion using Ollama."""
        try:
            # Convert messages to Ollama format
            prompt = self._format_messages_to_prompt(messages)
            
            response = self._post('/api/generate', {
                'model': self.config.ollama.chat_model,
                'prompt': prompt,
                'stream': False
            })
            
            return response.get('response', '').strip()
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to Ollama API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.post(
                url,
                json=data,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request to {url} failed: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON response: {e}")
            raise
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def check_model_availability(self) -> Dict[str, bool]:
        """Check if required models are available."""
        available_models = self.list_models()
        
        return {
            'embedding_model': self.config.ollama.embedding_model in available_models,
            'chat_model': self.config.ollama.chat_model in available_models,
            'available_models': available_models
        }