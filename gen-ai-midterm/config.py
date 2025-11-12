"""
Configuration file for UChicago MS-ADS RAG System
Loads environment variables and provides centralized config access
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Centralized configuration for RAG system"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
    
    # LangSmith Configuration
    LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2', 'true').lower() == 'true'
    LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT', 'https://api.smith.langchain.com')
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT', 'uchicago-msads-rag')
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
    CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'uchicago_msads_docs')
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")
        
        if cls.LANGCHAIN_TRACING_V2 and not cls.LANGCHAIN_API_KEY:
            errors.append("LANGCHAIN_API_KEY is not set but tracing is enabled")
        
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    @classmethod
    def setup_langsmith(cls):
        """Setup LangSmith environment variables for tracing"""
        if cls.LANGCHAIN_TRACING_V2 and cls.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = cls.LANGCHAIN_ENDPOINT
            os.environ["LANGCHAIN_API_KEY"] = cls.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = cls.LANGCHAIN_PROJECT
            return True
        return False
    
    @classmethod
    def display_config(cls):
        """Display current configuration (hiding sensitive keys)"""
        def mask_key(key):
            if not key:
                return "Not set"
            return f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
        
        config_info = f"""
╔══════════════════════════════════════════════════════════════╗
║           UChicago MS-ADS RAG System Configuration           ║
╚══════════════════════════════════════════════════════════════╝

API Keys:
  • OpenAI API Key:     {mask_key(cls.OPENAI_API_KEY)}
  • Firecrawl API Key:  {mask_key(cls.FIRECRAWL_API_KEY)}

LangSmith (Observability):
  • Tracing Enabled:    {cls.LANGCHAIN_TRACING_V2}
  • LangChain API Key:  {mask_key(cls.LANGCHAIN_API_KEY)}
  • Project Name:       {cls.LANGCHAIN_PROJECT}

ChromaDB:
  • Database Path:      {cls.CHROMA_DB_PATH}
  • Collection Name:    {cls.CHROMA_COLLECTION_NAME}

Models:
  • Embedding Model:    {cls.EMBEDDING_MODEL}
  • LLM Model:          {cls.LLM_MODEL}
  • LLM Temperature:    {cls.LLM_TEMPERATURE}
        """
        print(config_info)


# Validate configuration on import
try:
    Config.validate()
    print("✓ Configuration loaded successfully")
except ValueError as e:
    print(f"⚠ Configuration warning: {e}")

# Setup LangSmith if enabled
if Config.setup_langsmith():
    print("✓ LangSmith tracing enabled")
