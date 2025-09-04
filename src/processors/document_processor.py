"""
Document Processor Module
Following TDD principles - GREEN phase: Minimal implementation to pass tests
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import concurrent.futures
from threading import Lock

import pypdf
import docx
from pydantic import BaseModel

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Enumeration of supported document types"""

    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"
    UNKNOWN = "unknown"


class ProcessingError(Exception):
    """Custom exception for document processing errors"""

    pass


@dataclass
class Page:
    """Represents a single page from a document"""

    page_number: int
    content: str
    document_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of document extraction"""

    success: bool
    document_type: DocumentType
    pages: List[Page]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class DocumentProcessor:
    """Main document processing class following TDD implementation"""

    def __init__(self):
        """Initialize the document processor"""
        self.supported_formats = [".pdf", ".docx", ".txt"]
        self._processing_lock = Lock()
        logger.info("DocumentProcessor initialized")

    def detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type based on file extension and content"""
        if not file_path.exists():
            return DocumentType.UNKNOWN

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return DocumentType.PDF
        elif suffix == ".docx":
            return DocumentType.DOCX
        elif suffix in [".txt", ".text"]:
            return DocumentType.TEXT
        else:
            return DocumentType.UNKNOWN

    def extract_pages(self, file_path: Path) -> List[Page]:
        """Extract pages from a document"""
        doc_type = self.detect_document_type(file_path)
        pages = []

        try:
            if doc_type == DocumentType.PDF:
                pages = self._extract_pdf_pages(file_path)
            elif doc_type == DocumentType.DOCX:
                pages = self._extract_docx_pages(file_path)
            elif doc_type == DocumentType.TEXT:
                pages = self._extract_text_pages(file_path)
            else:
                raise ProcessingError(f"Unsupported document type: {doc_type}")
        except ProcessingError:
            # Re-raise ProcessingError to be handled by process_document
            raise
        except Exception as e:
            logger.error(f"Error extracting pages from {file_path}: {e}")
            # Return empty list on error for now
            return []

        return pages

    def _extract_pdf_pages(self, file_path: Path) -> List[Page]:
        """Extract pages from a PDF document"""
        pages = []
        try:
            with open(file_path, "rb") as file:
                # Try to read the file and detect if it's a valid PDF
                file_content = file.read()
                if not file_content.startswith(b"%PDF"):
                    raise ProcessingError("Invalid PDF file format")
                
                # Reset file pointer and read with pypdf
                file.seek(0)
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    pages.append(
                        Page(
                            page_number=page_num,
                            content=text,
                            document_path=file_path,
                        )
                    )
        except pypdf.errors.PdfReadError as e:
            logger.error(f"Corrupted PDF file: {e}")
            raise ProcessingError(f"Corrupted PDF file: {e}")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise ProcessingError(f"Error reading PDF: {e}")

        return pages

    def _extract_docx_pages(self, file_path: Path) -> List[Page]:
        """Extract pages from a DOCX document"""
        pages = []
        try:
            doc = docx.Document(file_path)
            full_text = []
            for paragraph in doc.paragraphs:
                full_text.append(paragraph.text)

            # For DOCX, treat entire document as one page for now
            if full_text:
                pages.append(
                    Page(
                        page_number=1,
                        content="\n".join(full_text),
                        document_path=file_path,
                    )
                )
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")

        return pages

    def _extract_text_pages(self, file_path: Path) -> List[Page]:
        """Extract pages from a text document"""
        pages = []
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                pages.append(
                    Page(
                        page_number=1,
                        content=content,
                        document_path=file_path,
                    )
                )
        except Exception as e:
            logger.error(f"Error reading text file: {e}")

        return pages

    def extract_metadata(self, page: Page) -> Dict[str, Any]:
        """Extract metadata from a single page"""
        content = page.content
        words = content.split()

        metadata = {
            "word_count": len(words),
            "char_count": len(content),
            "has_tables": False,  # Simplified for now
            "has_images": False,  # Simplified for now
            "line_count": content.count("\n") + 1,
        }

        return metadata

    def process_document(
        self, file_path: Path, stream_mode: bool = False
    ) -> ExtractionResult:
        """Process a complete document through the pipeline"""
        logger.info("Starting document processing", extra={"path": str(file_path)})

        # Check if file exists and is a file (not a Path object only)
        try:
            exists = file_path.exists()
        except OSError:
            exists = False
            
        if not exists:
            # Check if format is supported first for non-existent files
            if file_path.suffix.lower() not in self.supported_formats:
                raise ProcessingError(f"Unsupported file format: {file_path.suffix}")
            return ExtractionResult(
                success=False,
                document_type=DocumentType.UNKNOWN,
                pages=[],
                metadata={},
                error_message=f"File not found: {file_path}",
            )

        # Check if format is supported
        if file_path.suffix.lower() not in self.supported_formats:
            raise ProcessingError(f"Unsupported file format: {file_path.suffix}")

        doc_type = self.detect_document_type(file_path)

        try:
            # Extract pages
            pages = self.extract_pages(file_path)

            # Extract metadata for each page
            for page in pages:
                page.metadata = self.extract_metadata(page)

            # Document-level metadata
            doc_metadata = {
                "total_pages": len(pages),
                "document_path": str(file_path),
                "document_type": doc_type.value,
            }

            if stream_mode:
                doc_metadata["processing_mode"] = "stream"

            return ExtractionResult(
                success=True,
                document_type=doc_type,
                pages=pages,
                metadata=doc_metadata,
            )

        except ProcessingError as e:
            logger.error(f"Processing error: {e}")
            return ExtractionResult(
                success=False,
                document_type=doc_type,
                pages=[],
                metadata={},
                error_message=str(e),
            )
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return ExtractionResult(
                success=False,
                document_type=doc_type,
                pages=[],
                metadata={},
                error_message=str(e),
            )

    def analyze_layout(self, page: Page) -> Dict[str, Any]:
        """Analyze the layout of a page"""
        content = page.content
        lines = content.split("\n")

        # Simple layout analysis
        has_indentation = any(line.startswith("    ") or line.startswith("\t") for line in lines)

        layout_analysis = {
            "structure": {
                "has_indentation": has_indentation,
                "line_count": len(lines),
                "average_line_length": (
                    sum(len(line) for line in lines) / len(lines) if lines else 0
                ),
            },
            "sections": self._identify_sections(content),
        }

        return layout_analysis

    def _identify_sections(self, content: str) -> List[Dict[str, Any]]:
        """Identify sections in the content"""
        sections = []
        lines = content.split("\n")

        current_section = None
        for i, line in enumerate(lines):
            # Simple heuristic: lines with less indentation might be headers
            if line and not line.startswith(" ") and not line.startswith("\t"):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "title": line.strip(),
                    "start_line": i,
                    "content": [],
                }
            elif current_section:
                current_section["content"].append(line)

        if current_section:
            sections.append(current_section)

        return sections

    def batch_process(self, file_paths: List[Path]) -> List[ExtractionResult]:
        """Process multiple documents in batch"""
        results = []
        for file_path in file_paths:
            result = self.process_document(file_path)
            results.append(result)
        return results