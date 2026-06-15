import os
from pydantic import BaseModel, Field, field_validator, model_validator
from helpers.config import settings

class FileValidationSchema(BaseModel):
    filename: str = Field(..., description="The original name of the uploaded file")
    content_type: str = Field(..., description="The MIME type of the uploaded file")
    size: int = Field(..., description="The size of the file in bytes")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        if not filename or filename.strip() == "":
            raise ValueError("Filename cannot be empty.")
        
        # Extract base name to prevent directory traversal
        base_name = os.path.basename(filename)
        _, ext = os.path.splitext(base_name.lower())
        
        if not ext:
            raise ValueError("File must have a valid extension.")
            
        if ext not in settings.allowed_extensions:
            raise ValueError(
                f"File extension '{ext}' is not allowed. Allowed extensions: {', '.join(settings.allowed_extensions)}"
            )
            
        return base_name

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, content_type: str) -> str:
        if not content_type or content_type.strip() == "":
            raise ValueError("Content-Type cannot be empty.")
            
        normalized_content_type = content_type.lower().strip()
        if normalized_content_type not in settings.allowed_content_types:
            raise ValueError(
                f"Content-Type '{content_type}' is not allowed. Allowed types: {', '.join(settings.allowed_content_types)}"
            )
            
        return normalized_content_type

    @model_validator(mode="after")
    def validate_size_for_extension(self) -> 'FileValidationSchema':
        filename = self.filename
        size = self.size
        
        if size <= 0:
            raise ValueError("File size must be greater than 0.")
            
        _, ext = os.path.splitext(filename.lower())
        
        if ext in settings.large_file_extensions:
            limit = settings.max_large_file_size
            limit_desc = "large file limit"
        else:
            limit = settings.max_file_size
            limit_desc = "limit"
            
        if size > limit:
            limit_mb = limit / (1024 * 1024)
            raise ValueError(
                f"File size ({size} bytes) exceeds the {limit_desc} of {limit_mb:.2f} MB ({limit} bytes) for extension '{ext}'."
            )
            
        return self
