"""
Google Gemini API client wrapper for FinancialKG
"""
import asyncio
import concurrent.futures
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
import time

from .config import get_config
from .logging_config import get_logger

logger = get_logger(__name__)
config = get_config()


class GeminiClient:
    """
    Wrapper for Google Gemini API with async support and retry logic
    
    Supports both Gemini 2.0 Flash (fast) and Gemini 2.5 Pro (advanced reasoning)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_fast: Optional[str] = None,
        model_pro: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        self.api_key = api_key or config.gemini.api_key
        self.model_fast_name = model_fast or config.gemini.model_fast
        self.model_pro_name = model_pro or config.gemini.model_pro
        self.embedding_model_name = embedding_model or config.gemini.embedding_model
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY in .env file.")
        
        # Initialize client with new SDK
        self.client = genai.Client(api_key=self.api_key)

        # Thread pool sized for 1000 RPM: 100 workers → ~100 concurrent requests
        # Each request takes ~0.5–1s → throughput ≈ 100–200 req/s well within limit
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=100, thread_name_prefix="gemini"
        )
        
        logger.info(f"Initialized Gemini client with models: {self.model_fast_name}, {self.model_pro_name}")
    
    async def generate_async(
        self,
        prompt: str,
        use_pro: bool = False,
        temperature: float = 0.0,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """Generate text asynchronously with retry logic"""
        model_name = self.model_pro_name if use_pro else self.model_fast_name
        
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
        )
        
        for attempt in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self._executor,          # use dedicated 100-worker pool
                    lambda: self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=generation_config
                    )
                )
                return response.text
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {model_name}: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
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
        """Generate text synchronously with retry logic"""
        model_name = self.model_pro_name if use_pro else self.model_fast_name
        
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
        )
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=generation_config
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
        max_concurrent: int = 100        # raised from 5 → matches 1000 RPM quota
    ) -> List[str]:
        """Generate text for multiple prompts in parallel (with concurrency limit)"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_limit(prompt: str) -> str:
            async with semaphore:
                return await self.generate_async(prompt, use_pro=use_pro, temperature=temperature)
        
        tasks = [generate_with_limit(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embedding for text"""
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model_name,
                contents=text,
            )
            return result.embeddings[0].values
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    async def embed_text_async(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embedding for text asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text, task_type)
    
    async def embed_batch_async(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        max_concurrent: int = 10
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in parallel"""
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
    async def test_client():
        client = get_gemini_client()
        prompt = "What are the key financial metrics for evaluating a company?"
        response = await client.generate_async(prompt)
        print(f"Response: {response[:200]}...")
    
    asyncio.run(test_client())

