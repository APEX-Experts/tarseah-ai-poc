import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import ValidationError
from src.helpers.config import settings
from src.models.file_validation import FileValidationSchema

router = APIRouter(prefix="/files", tags=["Files"])

def sanitize_and_unique_filename(filename: str) -> str:
    """
    Sanitize the filename to prevent directory traversal and make it unique.
    """
    base_name, ext = os.path.splitext(filename)
    # Keep only alphanumeric characters, dashes, and underscores
    clean_base = "".join(c for c in base_name if c.isalnum() or c in ("-", "_")).strip()
    if not clean_base:
        clean_base = "uploaded_file"
    
    # Append a short unique suffix to avoid overwriting existing files
    unique_suffix = uuid.uuid4().hex[:8]
    return f"{clean_base}_{unique_suffix}{ext.lower()}"

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, validate it using Pydantic, and save it in the Assets folder.
    """
    # Ensure Assets directory exists
    settings.assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine file size
    # In FastAPI/Starlette, file.size gets populated. If it's not present, we can read/seek.
    file_size = file.size
    if file_size is None:
        try:
            file.file.seek(0, os.SEEK_END)
            file_size = file.file.tell()
            file.file.seek(0)
        except Exception:
            file_size = 0

    # 2. Validate metadata with Pydantic Schema
    try:
        validated_data = FileValidationSchema(
            filename=file.filename or "",
            content_type=file.content_type or "",
            size=file_size
        )
    except ValidationError as e:
        # Format the validation errors nicely
        errors = []
        for err in e.errors():
            loc = err.get("loc", [])
            msg = err.get("msg", "")
            # Clean up default Pydantic Value error prefix if present
            if msg.startswith("Value error, "):
                msg = msg[13:]
            field = loc[0] if loc else "file"
            errors.append(f"{field}: {msg}")
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "File validation failed.",
                "errors": errors
            }
        )

    # 3. Save the file to Assets directory
    unique_filename = sanitize_and_unique_filename(validated_data.filename)
    destination_path = settings.assets_dir / unique_filename

    try:
        async with aiofiles.open(destination_path, "wb") as out_file:
            # Read and write file in 1MB chunks to be memory-efficient
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while saving the file: {str(e)}"
        )
    finally:
        await file.close()

    return {
        "status": "success",
        "message": "File validated and saved successfully.",
        "data": {
            "original_filename": validated_data.filename,
            "saved_filename": unique_filename,
            "content_type": validated_data.content_type,
            "size_bytes": validated_data.size,
            "saved_path": str(destination_path)
        }
    }
