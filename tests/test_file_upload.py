import io
import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Add src folder to the python search path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from src.main import app
from src.helpers.config import settings

class TestFileUpload(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Store original settings to restore them after testing
        self.original_max_file_size = settings.max_file_size
        self.original_allowed_extensions = settings.allowed_extensions.copy()
        self.original_allowed_content_types = settings.allowed_content_types.copy()
        
        # Track uploaded files during the test to clean them up in tearDown
        self.created_files = []

    def tearDown(self):
        # Restore original settings
        settings.max_file_size = self.original_max_file_size
        settings.allowed_extensions = self.original_allowed_extensions
        settings.allowed_content_types = self.original_allowed_content_types
        
        # Clean up files created during the tests
        for file_path in self.created_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete test file {file_path}: {e}")

    def test_upload_success_txt(self):
        file_content = b"This is a sample text file for API validation testing."
        file_name = "sample_test.txt"
        
        response = self.client.post(
            "/files/upload",
            files={"file": (file_name, io.BytesIO(file_content), "text/plain")}
        )
        
        self.assertEqual(response.status_code, 201)
        response_json = response.json()
        self.assertEqual(response_json["status"], "success")
        self.assertEqual(response_json["data"]["original_filename"], file_name)
        self.assertEqual(response_json["data"]["content_type"], "text/plain")
        self.assertEqual(response_json["data"]["size_bytes"], len(file_content))
        
        saved_path = response_json["data"]["saved_path"]
        self.created_files.append(saved_path)
        
        # Verify file exists on the disk and has correct content
        self.assertTrue(os.path.exists(saved_path))
        with open(saved_path, "rb") as f:
            self.assertEqual(f.read(), file_content)

    def test_upload_success_png(self):
        file_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."
        file_name = "test_image.png"
        
        response = self.client.post(
            "/files/upload",
            files={"file": (file_name, io.BytesIO(file_content), "image/png")}
        )
        
        self.assertEqual(response.status_code, 201)
        response_json = response.json()
        self.assertEqual(response_json["status"], "success")
        self.assertEqual(response_json["data"]["content_type"], "image/png")
        
        saved_path = response_json["data"]["saved_path"]
        self.created_files.append(saved_path)
        
        # Verify file exists on disk
        self.assertTrue(os.path.exists(saved_path))

    def test_upload_invalid_extension(self):
        file_content = b"#!/bin/bash\necho 'Hello World'"
        file_name = "exploit.sh"
        
        response = self.client.post(
            "/files/upload",
            files={"file": (file_name, io.BytesIO(file_content), "text/plain")}
        )
        
        self.assertEqual(response.status_code, 400)
        response_json = response.json()
        self.assertIn("detail", response_json)
        self.assertIn("errors", response_json["detail"])
        errors = response_json["detail"]["errors"]
        self.assertTrue(any("filename" in err for err in errors))

    def test_upload_invalid_mime_type(self):
        file_content = b"plain text data"
        file_name = "textfile.txt"
        
        # Send text file but with blocked/dangerous content type
        response = self.client.post(
            "/files/upload",
            files={"file": (file_name, io.BytesIO(file_content), "application/x-msdownload")}
        )
        
        self.assertEqual(response.status_code, 400)
        response_json = response.json()
        self.assertIn("errors", response_json["detail"])
        errors = response_json["detail"]["errors"]
        self.assertTrue(any("content_type" in err for err in errors))

    def test_upload_exceeds_size_limit(self):
        # Configure setting to only allow max 15 bytes
        settings.max_file_size = 15
        
        file_content = b"This string is 30 bytes long!!"
        file_name = "oversized_file.txt"
        
        response = self.client.post(
            "/files/upload",
            files={"file": (file_name, io.BytesIO(file_content), "text/plain")}
        )
        
        self.assertEqual(response.status_code, 400)
        response_json = response.json()
        self.assertIn("errors", response_json["detail"])
        errors = response_json["detail"]["errors"]
        self.assertTrue(any("size" in err for err in errors))

if __name__ == "__main__":
    unittest.main()
