"""
Google Gemini API client wrapper for FinancialKG
"""
import asyncio
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import time

from utils.config import get_config
from utils.logging_config import get_logger

logger = get_logger(__name__)
config = get_config()


class GeminiClient:
    """
    Wrapper for Google Gemini API with async support and retry logic
    
    Supports both Gemini 2.0 Flash (fast) and Gemini 1.5 Pro (advanced reasoning)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_fast: Optional[str] = None,
        model_pro: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize Gemini client
        
        Args:
            api_key: Gemini API key (uses config if not provided)
            model_fast: Fast model name (Gemini 2.0 Flash)
            model_pro: Pro model name (Gemini 1.5 Pro)
            embedding_model: Embedding model name
        """
        self.api_key = api_key or config.gemini.api_key
        self.model_fast_name = model_fast or config.gemini.model_fast
        self.model_pro_name = model_pro or config.gemini.model_pro
        self.embedding_model_name = embedding_model or config.gemini.embedding_model
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY in .env file.")
        
        # Configure API
        genai.configure(api_key=self.api_key)
        
        # Initialize models
        self.model_fast = genai.GenerativeModel(self.model_fast_name)
        self.model_pro = genai.GenerativeModel(self.model_pro_name)
        
        logger.info(f"Initialized Gemini client with models: {self.model_fast_name}, {self.model_pro_name}")
    
    async def generate_async(
        self,
        prompt: str,
        use_pro: bool = False,
        temperature: float = 0.0,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Generate text asynchronously with retry logic
        
        Args:
            prompt: Input prompt
            use_pro: Use Pro model (default: Fast model)
            temperature: Generation temperature
            max_retries: Maximum retry attempts
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        model = self.model_pro if use_pro else self.model_fast
        model_name = self.model_pro_name if use_pro else self.model_fast_name
        
        generation_config = GenerationConfig(
            temperature=temperature,
            **kwargs
        )
        
        for attempt in range(max_retries):
            try:
                # Run in thread pool (Gemini SDK doesn't have native async yet)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                )
                
                return response.text
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {model_name}: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retries failed for {model_name}")
                    raise
    
    def generate_sync(
        self,
        prompt: str,
        use_pro: bool = False,
        temperature: float = 0.0,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Generate text synchronously with retry logic
        
        Args:
            prompt: Input prompt
            use_pro: Use Pro model (default: Fast model)
            temperature: Generation temperature
            max_retries: Maximum retry attempts
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        model = self.model_pro if use_pro else self.model_fast
        model_name = self.model_pro_name if use_pro else self.model_fast_name
        
        generation_config = GenerationConfig(
            temperature=temperature,
            **kwargs
        )
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response.text
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {model_name}: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retries failed for {model_name}")
                    raise
    
    async def generate_batch_async(
        self,
        prompts: List[str],
        use_pro: bool = False,
        temperature: float = 0.0,
        max_concurrent: int = 5
    ) -> List[str]:
        """
        Generate text for multiple prompts in parallel (with concurrency limit)
        
        Args:
            prompts: List of input prompts
            use_pro: Use Pro model
            temperature: Generation temperature
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of generated texts
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_limit(prompt: str) -> str:
            async with semaphore:
                return await self.generate_async(prompt, use_pro=use_pro, temperature=temperature)
        
        tasks = [generate_with_limit(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generate embedding for text
        
        Args:
            text: Input text
            task_type: Embedding task type (retrieval_document, retrieval_query, etc.)
            
        Returns:
            Embedding vector
        """
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type=task_type
            )
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    async def embed_text_async(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generate embedding for text asynchronously
        
        Args:
            text: Input text
            task_type: Embedding task type
            
        Returns:
            Embedding vector
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text, task_type)
    
    async def embed_batch_async(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        max_concurrent: int = 10
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in parallel
        
        Args:
            texts: List of input texts
            task_type: Embedding task type
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of embedding vectors
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def embed_with_limit(text: str) -> List[float]:
            async with semaphore:
                return await self.embed_text_async(text, task_type)
        
        tasks = [embed_with_limit(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return results


# Global client instance
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create global Gemini client instance"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


if __name__ == "__main__":
    # Test Gemini client
    async def test_client():
        client = get_gemini_client()
        
        # Test generation
        prompt = "What are the key financial metrics for evaluating a company?"
        response = await client.generate_async(prompt)
        print(f"Response: {response[:200]}...")
        
        # Test embedding
        text = "Reliance Industries is a major energy company in India"
        embedding = await client.embed_text_async(text)
        print(f"Embedding dimension: {len(embedding)}")
    
    asyncio.run(test_client())
