"""
Prompt Cache Module

Provides caching functionality for optimized prompts with persistence to disk.
"""

import os
import json
import time
import threading
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger("prompt-cache")

class PromptCache:
    """Cache for optimized prompts with persistence"""
    
    def __init__(self, cache_dir: str = "./cache"):
        """Initialize the prompt cache
        
        Args:
            cache_dir: Directory to store cached prompts
        """
        self.cache_dir = Path(cache_dir)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache from disk
        self._load_cache()
    
    def _get_cache_file(self, tone_style: str) -> Path:
        """Get the cache file for a specific tone style"""
        # Replace any characters that might be problematic in filenames
        safe_name = tone_style.replace(" ", "_").replace("/", "_").lower()
        return self.cache_dir / f"{safe_name}.json"
    
    def _load_cache(self):
        """Load all cached prompts from disk"""
        try:
            # Look for all JSON files in the cache directory
            for file in self.cache_dir.glob("*.json"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "tone_style" in data and "prompt" in data:
                            tone_style = data["tone_style"]
                            self.cache[tone_style] = data
                            logger.info(f"Loaded cached prompt for '{tone_style}'")
                except Exception as e:
                    logger.warning(f"Failed to load cache file {file}: {e}")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    def get(self, tone_style: str) -> Optional[Dict[str, Any]]:
        """Get a cached prompt by tone style
        
        Args:
            tone_style: The tone style to retrieve
            
        Returns:
            Dict with prompt data or None if not found
        """
        with self.lock:
            return self.cache.get(tone_style)
    
    def save(self, tone_style: str, prompt_data: Dict[str, Any]):
        """Save a prompt to the cache
        
        Args:
            tone_style: The tone style
            prompt_data: Dict containing at least 'prompt' and other metadata
        """
        if "prompt" not in prompt_data:
            raise ValueError("prompt_data must contain 'prompt' field")
        
        # Add timestamp and tone_style if not present
        if "timestamp" not in prompt_data:
            prompt_data["timestamp"] = time.time()
        if "tone_style" not in prompt_data:
            prompt_data["tone_style"] = tone_style
        
        with self.lock:
            # Update memory cache
            self.cache[tone_style] = prompt_data
            
            # Save to disk
            cache_file = self._get_cache_file(tone_style)
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(prompt_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved prompt for '{tone_style}' to cache")
            except Exception as e:
                logger.error(f"Failed to save prompt cache for '{tone_style}': {e}")
    
    def clear(self, tone_style: Optional[str] = None):
        """Clear cache for a specific tone style or all if none specified
        
        Args:
            tone_style: Specific tone style to clear, or None for all
        """
        with self.lock:
            if tone_style is not None:
                # Clear specific tone style
                if tone_style in self.cache:
                    del self.cache[tone_style]
                    cache_file = self._get_cache_file(tone_style)
                    if cache_file.exists():
                        try:
                            cache_file.unlink()
                            logger.info(f"Cleared cache for '{tone_style}'")
                        except Exception as e:
                            logger.error(f"Failed to delete cache file for '{tone_style}': {e}")
            else:
                # Clear all
                self.cache.clear()
                try:
                    for file in self.cache_dir.glob("*.json"):
                        file.unlink()
                    logger.info("Cleared entire prompt cache")
                except Exception as e:
                    logger.error(f"Failed to clear cache directory: {e}")
    
    def list_tones(self) -> Dict[str, Dict[str, Any]]:
        """List all cached tone styles with metadata
        
        Returns:
            Dict mapping tone styles to metadata
        """
        with self.lock:
            # Return a copy to avoid thread safety issues
            return {k: {
                "timestamp": v.get("timestamp", 0),
                "fitness_score": v.get("fitness_score", 0),
                "generation": v.get("generation", 0)
            } for k, v in self.cache.items()}

# Global instance
prompt_cache = PromptCache() 