"""
Configuration Management System

Handles settings for the Research Copilot system including:
- Environment variables and .env files
- Configuration validation
- Default settings
- API key management
- Model selection and parameters
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .summarizer import LLMProvider, SummaryType
from .chunker import ChunkingStrategy

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 120


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    # PDF Processing
    use_ocr: bool = True
    ocr_fallback_threshold: float = 0.1
    
    # Chunking
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    max_chunk_tokens: int = 1000
    overlap_tokens: int = 100
    min_chunk_tokens: int = 50
    
    # Summarization
    summary_type: SummaryType = SummaryType.STRUCTURED
    max_summary_tokens: int = 500
    
    # Vector Index
    embedding_model: str = "all-MiniLM-L6-v2"
    index_type: str = "flat"
    vector_dimension: int = 384
    
    # Performance
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    # Base directories
    data_dir: str = "./data"
    output_dir: str = "./data/processed"
    index_dir: str = "./data/embeddings"
    cache_dir: str = "./data/cache"
    
    # Collector integration
    collector_db: str = "./data/papers.db"
    
    # Logging
    log_file: str = "./logs/research_copilot.log"
    log_level: str = "INFO"


@dataclass
class APIConfig:
    """Configuration for API keys and endpoints."""
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_org_id: Optional[str] = None
    
    # Anthropic
    anthropic_api_key: Optional[str] = None
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    
    # Azure OpenAI
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2023-12-01-preview"


@dataclass
class ResearchCopilotConfig:
    """Complete configuration for Research Copilot."""
    # Sub-configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    apis: APIConfig = field(default_factory=APIConfig)
    
    # General settings
    debug: bool = False
    save_intermediate: bool = True
    auto_cleanup: bool = False
    
    def __post_init__(self):
        """Initialize default models if none provided."""
        if not self.models:
            self.models = self._get_default_models()
    
    def _get_default_models(self) -> Dict[str, ModelConfig]:
        """Get default model configurations."""
        return {
            "ollama_llama2": ModelConfig(
                provider=LLMProvider.OLLAMA,
                model_name="llama2",
                base_url=self.apis.ollama_base_url,
                temperature=0.3,
                max_tokens=2000
            ),
            "openai_gpt35": ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-3.5-turbo",
                api_key=self.apis.openai_api_key,
                temperature=0.3,
                max_tokens=2000
            ),
            "anthropic_claude": ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                api_key=self.apis.anthropic_api_key,
                temperature=0.3,
                max_tokens=2000
            )
        }


class ConfigManager:
    """Manages configuration loading, validation, and saving."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else Path("config.json")
        self.env_file = Path(".env")
        self.config: Optional[ResearchCopilotConfig] = None
        
        # Load environment variables
        self._load_env_vars()
    
    def _load_env_vars(self):
        """Load environment variables from .env file."""
        if load_dotenv and self.env_file.exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
    
    def load_config(self) -> ResearchCopilotConfig:
        """Load configuration from file and environment."""
        if self.config_path.exists():
            logger.info(f"Loading configuration from {self.config_path}")
            config = self._load_from_file()
        else:
            logger.info("Creating default configuration")
            config = ResearchCopilotConfig()
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        # Validate configuration
        self._validate_config(config)
        
        self.config = config
        return config
    
    def _load_from_file(self) -> ResearchCopilotConfig:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            # Convert to dataclass
            config = self._dict_to_config(data)
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
            return ResearchCopilotConfig()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> ResearchCopilotConfig:
        """Convert dictionary to configuration object."""
        # Extract sub-configurations
        models_data = data.get('models', {})
        processing_data = data.get('processing', {})
        paths_data = data.get('paths', {})
        apis_data = data.get('apis', {})
        
        # Convert models
        models = {}
        for name, model_data in models_data.items():
            models[name] = ModelConfig(
                provider=LLMProvider(model_data.get('provider', 'ollama')),
                model_name=model_data.get('model_name', 'llama2'),
                api_key=model_data.get('api_key'),
                base_url=model_data.get('base_url'),
                temperature=model_data.get('temperature', 0.3),
                max_tokens=model_data.get('max_tokens', 2000),
                timeout=model_data.get('timeout', 120)
            )
        
        # Convert processing config
        processing = ProcessingConfig(
            use_ocr=processing_data.get('use_ocr', True),
            ocr_fallback_threshold=processing_data.get('ocr_fallback_threshold', 0.1),
            chunking_strategy=ChunkingStrategy(processing_data.get('chunking_strategy', 'hybrid')),
            max_chunk_tokens=processing_data.get('max_chunk_tokens', 1000),
            overlap_tokens=processing_data.get('overlap_tokens', 100),
            min_chunk_tokens=processing_data.get('min_chunk_tokens', 50),
            summary_type=SummaryType(processing_data.get('summary_type', 'structured')),
            max_summary_tokens=processing_data.get('max_summary_tokens', 500),
            embedding_model=processing_data.get('embedding_model', 'all-MiniLM-L6-v2'),
            index_type=processing_data.get('index_type', 'flat'),
            vector_dimension=processing_data.get('vector_dimension', 384),
            batch_size=processing_data.get('batch_size', 10),
            max_retries=processing_data.get('max_retries', 3),
            retry_delay=processing_data.get('retry_delay', 1.0),
            enable_caching=processing_data.get('enable_caching', True)
        )
        
        # Convert paths config
        paths = PathConfig(**paths_data) if paths_data else PathConfig()
        
        # Convert APIs config
        apis = APIConfig(**apis_data) if apis_data else APIConfig()
        
        return ResearchCopilotConfig(
            models=models,
            processing=processing,
            paths=paths,
            apis=apis,
            debug=data.get('debug', False),
            save_intermediate=data.get('save_intermediate', True),
            auto_cleanup=data.get('auto_cleanup', False)
        )
    
    def _apply_env_overrides(self, config: ResearchCopilotConfig) -> ResearchCopilotConfig:
        """Apply environment variable overrides."""
        # API keys
        if os.getenv('OPENAI_API_KEY'):
            config.apis.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if os.getenv('ANTHROPIC_API_KEY'):
            config.apis.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if os.getenv('AZURE_OPENAI_API_KEY'):
            config.apis.azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        
        # Ollama base URL
        if os.getenv('OLLAMA_BASE_URL'):
            config.apis.ollama_base_url = os.getenv('OLLAMA_BASE_URL')
        
        # Paths
        if os.getenv('RESEARCH_COPILOT_DATA_DIR'):
            config.paths.data_dir = os.getenv('RESEARCH_COPILOT_DATA_DIR')
        
        if os.getenv('RESEARCH_COPILOT_OUTPUT_DIR'):
            config.paths.output_dir = os.getenv('RESEARCH_COPILOT_OUTPUT_DIR')
        
        # Processing settings
        if os.getenv('RESEARCH_COPILOT_DEBUG'):
            config.debug = os.getenv('RESEARCH_COPILOT_DEBUG').lower() == 'true'
        
        # Update model API keys
        for model_config in config.models.values():
            if model_config.provider == LLMProvider.OPENAI and not model_config.api_key:
                model_config.api_key = config.apis.openai_api_key
            elif model_config.provider == LLMProvider.ANTHROPIC and not model_config.api_key:
                model_config.api_key = config.apis.anthropic_api_key
        
        return config
    
    def _validate_config(self, config: ResearchCopilotConfig):
        """Validate configuration settings."""
        errors = []
        
        # Validate paths
        try:
            Path(config.paths.data_dir).mkdir(parents=True, exist_ok=True)
            Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
            Path(config.paths.index_dir).mkdir(parents=True, exist_ok=True)
            Path(config.paths.cache_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create directories: {e}")
        
        # Validate processing settings
        if config.processing.max_chunk_tokens <= 0:
            errors.append("max_chunk_tokens must be positive")
        
        if config.processing.overlap_tokens >= config.processing.max_chunk_tokens:
            errors.append("overlap_tokens must be less than max_chunk_tokens")
        
        if config.processing.min_chunk_tokens <= 0:
            errors.append("min_chunk_tokens must be positive")
        
        # Validate models
        for name, model in config.models.items():
            if model.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC] and not model.api_key:
                logger.warning(f"No API key provided for {name} ({model.provider.value})")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        logger.info("Configuration validation passed")
    
    def save_config(self, config: Optional[ResearchCopilotConfig] = None):
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        if config is None:
            raise ValueError("No configuration to save")
        
        try:
            # Convert to dictionary
            config_dict = self._config_to_dict(config)
            
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _config_to_dict(self, config: ResearchCopilotConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        # Convert models
        models_dict = {}
        for name, model in config.models.items():
            models_dict[name] = {
                'provider': model.provider.value,
                'model_name': model.model_name,
                'api_key': model.api_key,
                'base_url': model.base_url,
                'temperature': model.temperature,
                'max_tokens': model.max_tokens,
                'timeout': model.timeout
            }
        
        return {
            'models': models_dict,
            'processing': {
                'use_ocr': config.processing.use_ocr,
                'ocr_fallback_threshold': config.processing.ocr_fallback_threshold,
                'chunking_strategy': config.processing.chunking_strategy.value,
                'max_chunk_tokens': config.processing.max_chunk_tokens,
                'overlap_tokens': config.processing.overlap_tokens,
                'min_chunk_tokens': config.processing.min_chunk_tokens,
                'summary_type': config.processing.summary_type.value,
                'max_summary_tokens': config.processing.max_summary_tokens,
                'embedding_model': config.processing.embedding_model,
                'index_type': config.processing.index_type,
                'vector_dimension': config.processing.vector_dimension,
                'batch_size': config.processing.batch_size,
                'max_retries': config.processing.max_retries,
                'retry_delay': config.processing.retry_delay,
                'enable_caching': config.processing.enable_caching
            },
            'paths': asdict(config.paths),
            'apis': asdict(config.apis),
            'debug': config.debug,
            'save_intermediate': config.save_intermediate,
            'auto_cleanup': config.auto_cleanup
        }
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        if not self.config:
            self.load_config()
        
        return self.config.models.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """List all available model configurations."""
        if not self.config:
            self.load_config()
        
        return list(self.config.models.keys())
    
    def add_model(self, name: str, model_config: ModelConfig):
        """Add a new model configuration."""
        if not self.config:
            self.load_config()
        
        self.config.models[name] = model_config
        logger.info(f"Added model configuration: {name}")
    
    def remove_model(self, name: str) -> bool:
        """Remove a model configuration."""
        if not self.config:
            self.load_config()
        
        if name in self.config.models:
            del self.config.models[name]
            logger.info(f"Removed model configuration: {name}")
            return True
        
        return False
    
    def create_sample_config(self, output_path: Optional[str] = None):
        """Create a sample configuration file."""
        if output_path is None:
            output_path = "config.sample.json"
        
        # Create sample configuration
        sample_config = ResearchCopilotConfig()
        
        # Save to file
        config_dict = self._config_to_dict(sample_config)
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Sample configuration created at {output_path}")
    
    def create_env_template(self, output_path: Optional[str] = None):
        """Create a .env template file."""
        if output_path is None:
            output_path = ".env.template"
        
        template = """# Research Copilot Environment Variables

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Paths
RESEARCH_COPILOT_DATA_DIR=./data
RESEARCH_COPILOT_OUTPUT_DIR=./data/processed

# Settings
RESEARCH_COPILOT_DEBUG=false
"""
        
        with open(output_path, 'w') as f:
            f.write(template)
        
        logger.info(f"Environment template created at {output_path}")


def get_config(config_path: Optional[str] = None) -> ResearchCopilotConfig:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Loaded configuration
    """
    manager = ConfigManager(config_path)
    return manager.load_config()


def create_sample_config(output_path: str = "config.sample.json"):
    """Create a sample configuration file."""
    manager = ConfigManager()
    manager.create_sample_config(output_path)


def create_env_template(output_path: str = ".env.template"):
    """Create a .env template file."""
    manager = ConfigManager()
    manager.create_env_template(output_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "sample":
            create_sample_config()
            print("Sample configuration created as config.sample.json")
        
        elif command == "env":
            create_env_template()
            print("Environment template created as .env.template")
        
        elif command == "validate":
            config_path = sys.argv[2] if len(sys.argv) > 2 else None
            try:
                config = get_config(config_path)
                print("✓ Configuration is valid")
                print(f"Found {len(config.models)} model configurations")
            except Exception as e:
                print(f"✗ Configuration validation failed: {e}")
                sys.exit(1)
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    
    else:
        print("Usage: python config.py [sample|env|validate] [config_path]")
        sys.exit(1)
