"""
Test-Driven Development: Document Processor Tests
Following TDD principles - RED phase: Write failing tests first
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.processors.document_processor import (
    DocumentProcessor,
    Page,
    ProcessingError,
    DocumentType,
    ExtractionResult,
)


class TestDocumentProcessor:
    """Unit tests for DocumentProcessor following TDD principles"""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing"""
        return DocumentProcessor()

    @pytest.fixture
    def sample_pdf_path(self):
        """Create a temporary PDF file for testing"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n%...")  # Minimal PDF header
            return Path(f.name)

    @pytest.fixture
    def sample_docx_path(self):
        """Create a temporary DOCX file for testing"""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    def sample_txt_path(self):
        """Create a temporary text file for testing"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Sample text document content.\nSecond line of text.")
            return Path(f.name)

    @pytest.mark.unit
    def test_processor_initialization(self, processor):
        """Test that DocumentProcessor initializes correctly"""
        assert processor is not None
        assert hasattr(processor, "supported_formats")
        assert ".pdf" in processor.supported_formats
        assert ".docx" in processor.supported_formats
        assert ".txt" in processor.supported_formats

    @pytest.mark.unit
    def test_detect_document_type(self, processor, sample_pdf_path):
        """Test document type detection based on file extension and content"""
        doc_type = processor.detect_document_type(sample_pdf_path)
        assert doc_type == DocumentType.PDF

    @pytest.mark.unit
    def test_extract_pages_from_pdf(self, processor, sample_pdf_path):
        """Test page extraction from PDF documents"""
        # Since our minimal PDF is invalid, we expect it to raise an error or return empty
        try:
            pages = processor.extract_pages(sample_pdf_path)
            # If it doesn't raise, we should get an empty list for invalid PDF
            assert isinstance(pages, list)
            assert len(pages) == 0  # Invalid PDF should return empty list
        except ProcessingError:
            # This is also acceptable for an invalid PDF
            pass

    @pytest.mark.unit
    def test_extract_pages_from_text(self, processor, sample_txt_path):
        """Test page extraction from text documents"""
        pages = processor.extract_pages(sample_txt_path)
        assert isinstance(pages, list)
        assert len(pages) >= 1
        assert pages[0].content == "Sample text document content.\nSecond line of text."

    @pytest.mark.unit
    def test_extract_metadata_from_page(self, processor):
        """Test metadata extraction from a single page"""
        page = Page(
            page_number=1,
            content="This is a test page with some content.",
            document_path=Path("/tmp/test.pdf"),
        )
        metadata = processor.extract_metadata(page)
        
        assert "word_count" in metadata
        assert "char_count" in metadata
        assert "has_tables" in metadata
        assert "has_images" in metadata
        assert metadata["word_count"] == 8

    @pytest.mark.unit
    def test_process_document_complete_pipeline(self, processor, sample_txt_path):
        """Test complete document processing pipeline"""
        result = processor.process_document(sample_txt_path)
        
        assert isinstance(result, ExtractionResult)
        assert result.success is True
        assert result.document_type == DocumentType.TEXT
        assert len(result.pages) >= 1
        assert result.metadata is not None

    @pytest.mark.unit
    def test_handle_unsupported_format(self, processor):
        """Test handling of unsupported file formats"""
        unsupported_path = Path("/tmp/test.xyz")
        
        with pytest.raises(ProcessingError) as exc_info:
            processor.process_document(unsupported_path)
        
        assert "Unsupported file format" in str(exc_info.value)

    @pytest.mark.unit
    def test_handle_corrupted_document(self, processor):
        """Test handling of corrupted documents"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"Not a valid PDF content")
            corrupted_path = Path(f.name)
        
        result = processor.process_document(corrupted_path)
        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.unit
    def test_extract_text_with_layout_preservation(self, processor):
        """Test text extraction with layout preservation"""
        page = Page(
            page_number=1,
            content="Header\n\n    Indented content\n\nFooter",
            document_path=Path("/tmp/test.pdf"),
        )
        
        layout_analysis = processor.analyze_layout(page)
        assert "structure" in layout_analysis
        assert "sections" in layout_analysis
        assert layout_analysis["structure"]["has_indentation"] is True

    @pytest.mark.unit
    def test_batch_processing(self, processor):
        """Test batch processing of multiple documents"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            files = []
            for i in range(3):
                file_path = Path(tmpdir) / f"test_{i}.txt"
                file_path.write_text(f"Content of document {i}")
                files.append(file_path)
            
            results = processor.batch_process(files)
            assert len(results) == 3
            assert all(r.success for r in results)

    @pytest.mark.unit
    @patch("src.processors.document_processor.logger")
    def test_logging_during_processing(self, mock_logger, processor, sample_txt_path):
        """Test that processing events are properly logged"""
        processor.process_document(sample_txt_path)
        
        # Verify logging calls
        assert mock_logger.info.called
        mock_logger.info.assert_any_call(
            f"Starting document processing", extra={"path": str(sample_txt_path)}
        )

    @pytest.mark.unit
    def test_memory_efficient_processing(self, processor):
        """Test that large documents are processed memory-efficiently"""
        # Create a large text file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            # Write 10MB of text
            large_content = "x" * (10 * 1024 * 1024)
            f.write(large_content)
            large_file = Path(f.name)
        
        # Should process without memory error
        result = processor.process_document(large_file, stream_mode=True)
        assert result.success is True
        assert result.metadata["processing_mode"] == "stream"

    @pytest.mark.unit
    def test_concurrent_processing_safety(self, processor):
        """Test thread-safe concurrent document processing"""
        import concurrent.futures
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files = []
            for i in range(10):
                file_path = Path(tmpdir) / f"concurrent_{i}.txt"
                file_path.write_text(f"Concurrent test {i}")
                files.append(file_path)
            
            # Process concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(processor.process_document, file_path)
                    for file_path in files
                ]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            assert len(results) == 10
            assert all(r.success for r in results)