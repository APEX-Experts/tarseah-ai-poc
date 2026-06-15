import io
import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Add src folder to the python search path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from main import app
from helpers.config import settings

class TestFileUpload(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Store original settings to restore them after testing
        self.original_max_file_size = settings.max_file_size
        self.original_max_large_file_size = settings.max_large_file_size
        self.original_allowed_extensions = settings.allowed_extensions.copy()
        self.original_allowed_content_types = settings.allowed_content_types.copy()
        self.original_large_file_extensions = settings.large_file_extensions.copy()
        
        # Track uploaded files during the test to clean them up in tearDown
        self.created_files = []

    def tearDown(self):
        # Restore original settings
        settings.max_file_size = self.original_max_file_size
        settings.max_large_file_size = self.original_max_large_file_size
        settings.allowed_extensions = self.original_allowed_extensions
        settings.allowed_content_types = self.original_allowed_content_types
        settings.large_file_extensions = self.original_large_file_extensions
        
        # Clean up files and folders created during the tests
        for file_path in self.created_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    # Try to remove parent directory (project subfolder) if empty
                    parent_dir = os.path.dirname(file_path)
                    if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                except Exception as e:
                    print(f"Failed to delete test file/folder {file_path}: {e}")

    def test_upload_success_txt(self):
        file_content = b"This is a sample text file for API validation testing."
        file_name = "sample_test.txt"
        project_id = "test_project_123"
        
        response = self.client.post(
            f"/files/upload/{project_id}",
            files={"file": (file_name, io.BytesIO(file_content), "text/plain")}
        )
        
        self.assertEqual(response.status_code, 201)
        response_json = response.json()
        self.assertEqual(response_json["status"], "success")
        self.assertEqual(response_json["data"]["original_filename"], file_name)
        self.assertEqual(response_json["data"]["content_type"], "text/plain")
        self.assertEqual(response_json["data"]["size_bytes"], len(file_content))
        self.assertEqual(response_json["data"]["project_id"], project_id)
        
        saved_path = response_json["data"]["saved_path"]
        self.created_files.append(saved_path)
        
        # Verify file exists on the disk under project ID folder and has correct content
        self.assertTrue(os.path.exists(saved_path))
        self.assertIn(project_id, saved_path)
        with open(saved_path, "rb") as f:
            self.assertEqual(f.read(), file_content)

    def test_upload_success_png(self):
        file_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."
        file_name = "test_image.png"
        project_id = "test_project_png"
        
        response = self.client.post(
            f"/files/upload/{project_id}",
            files={"file": (file_name, io.BytesIO(file_content), "image/png")}
        )
        
        self.assertEqual(response.status_code, 201)
        response_json = response.json()
        self.assertEqual(response_json["status"], "success")
        self.assertEqual(response_json["data"]["content_type"], "image/png")
        
        saved_path = response_json["data"]["saved_path"]
        self.created_files.append(saved_path)
        
        self.assertTrue(os.path.exists(saved_path))
        self.assertIn(project_id, saved_path)

    def test_upload_invalid_extension(self):
        file_content = b"#!/bin/bash\necho 'Hello World'"
        file_name = "exploit.sh"
        
        response = self.client.post(
            "/files/upload/test_project_invalid_ext",
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
            "/files/upload/test_project_invalid_mime",
            files={"file": (file_name, io.BytesIO(file_content), "application/x-msdownload")}
        )
        
        self.assertEqual(response.status_code, 400)
        response_json = response.json()
        self.assertIn("errors", response_json["detail"])
        errors = response_json["detail"]["errors"]
        self.assertTrue(any("content_type" in err for err in errors))

    def test_upload_invalid_project_id(self):
        file_content = b"Some content"
        file_name = "test.txt"
        
        # Empty or spaces project ID (FastAPI URL encodes it or handles as path parameter value)
        response = self.client.post(
            "/files/upload/%20%20%20",
            files={"file": (file_name, io.BytesIO(file_content), "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        
        # Directory traversal attempt
        response = self.client.post(
            "/files/upload/..%2F..%2Fdangerous_dir",
            files={"file": (file_name, io.BytesIO(file_content), "text/plain")}
        )
        # Slashes/dots in the path parameter are intercepted/rejected by the router as a 404
        self.assertEqual(response.status_code, 404)

    def test_upload_exceeds_size_limit_standard_file(self):
        # Configure standard size limit to only allow max 15 bytes
        settings.max_file_size = 15
        settings.max_large_file_size = 100
        
        file_content = b"This string is 30 bytes long!!"
        file_name = "oversized_file.txt"
        
        response = self.client.post(
            "/files/upload/test_project_size",
            files={"file": (file_name, io.BytesIO(file_content), "text/plain")}
        )
        
        self.assertEqual(response.status_code, 400)
        response_json = response.json()
        self.assertIn("errors", response_json["detail"])
        errors = response_json["detail"]["errors"]
        self.assertTrue(any("size" in err for err in errors))

    def test_upload_large_file_success_for_allowed_extensions(self):
        # Configure limits: standard = 15 bytes, large = 100 bytes
        settings.max_file_size = 15
        settings.max_large_file_size = 100
        
        # 30 bytes (exceeds standard 15 bytes, but below large 100 bytes)
        file_content = b"This string is 30 bytes long!!"
        
        # Test PDF (allowed large)
        response_pdf = self.client.post(
            "/files/upload/test_project_large",
            files={"file": ("large_doc.pdf", io.BytesIO(file_content), "application/pdf")}
        )
        self.assertEqual(response_pdf.status_code, 201)
        self.created_files.append(response_pdf.json()["data"]["saved_path"])

        # Test DOCX (allowed large)
        response_docx = self.client.post(
            "/files/upload/test_project_large",
            files={"file": ("large_doc.docx", io.BytesIO(file_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        self.assertEqual(response_docx.status_code, 201)
        self.created_files.append(response_docx.json()["data"]["saved_path"])

        # Test MD (allowed large)
        response_md = self.client.post(
            "/files/upload/test_project_large",
            files={"file": ("large_doc.md", io.BytesIO(file_content), "text/markdown")}
        )
        self.assertEqual(response_md.status_code, 201)
        self.created_files.append(response_md.json()["data"]["saved_path"])

    def test_upload_large_file_exceeds_large_limit(self):
        settings.max_file_size = 15
        settings.max_large_file_size = 25
        
        # 30 bytes (exceeds large 25 bytes)
        file_content = b"This string is 30 bytes long!!"
        
        response = self.client.post(
            "/files/upload/test_project_too_large",
            files={"file": ("too_large_doc.pdf", io.BytesIO(file_content), "application/pdf")}
        )
        self.assertEqual(response.status_code, 400)
        response_json = response.json()
        self.assertIn("errors", response_json["detail"])
        errors = response_json["detail"]["errors"]
        self.assertTrue(any("size" in err for err in errors))

if __name__ == "__main__":
    unittest.main()
