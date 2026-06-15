import os
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base directory of the project
    base_dir: Path = Path(__file__).parent.parent.parent

    # Directory where uploaded assets will be saved
    assets_dir: Path = Field(
        default=Path(__file__).parent.parent / "Assets" / "files",
        description="Directory where uploaded files will be saved"
    )

    # Allowed file extensions (must start with a dot, lowercase)
    allowed_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf", ".txt", ".csv", ".json", ".doc", ".docx", ".md"],
        description="Allowed file extensions for upload"
    )

    # Allowed MIME types (content types)
    allowed_content_types: List[str] = Field(
        default=[
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "application/pdf",
            "text/plain",
            "text/csv",
            "application/json",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/markdown",
            "text/x-markdown"
        ],
        description="Allowed content types for upload"
    )

    # Maximum file size in bytes for standard files (default: 10 MB)
    max_file_size: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum allowed file size in bytes for standard files (default 10MB)"
    )

    # Maximum file size in bytes for large files like PDF, Word, and MD (default: 500 MB)
    max_large_file_size: int = Field(
        default=500 * 1024 * 1024,
        description="Maximum allowed file size in bytes for large file formats like PDF, Word, MD (default 500MB)"
    )

    # List of extensions allowed to be large
    large_file_extensions: List[str] = Field(
        default=[".pdf", ".doc", ".docx", ".md"],
        description="File extensions allowed to use the large file limit"
    )

settings = Settings()
