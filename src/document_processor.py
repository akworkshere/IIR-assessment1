import pdfplumber
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentProcessor:

    def __init__(self, min_text_length: int = 50):
        self.min_text_length = min_text_length

    def process_pdf(self, file_path: Path) -> Dict:
        try:
            logger.info(f"Processing PDF: {file_path.name}")

            if not file_path.exists():
                return {"success": False, "error": "File not found"}

            with pdfplumber.open(file_path) as pdf:
                metadata = self._extract_metadata(pdf, file_path)

                pages = []
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_data = self._process_page(page, page_num)
                    if page_data:
                        pages.append(page_data)

                total_text = "".join([p["text"] for p in pages])
                if len(total_text.strip()) < self.min_text_length:
                    return {
                        "success": False,
                        "error": "PDF appears to be empty or contains no extractable text"
                    }

                logger.info(f"Successfully processed {len(pages)} pages from {file_path.name}")

                return {
                    "success": True,
                    "document_name": file_path.stem,
                    "file_path": str(file_path),
                    "pages": pages,
                    "metadata": metadata,
                    "total_pages": len(pages),
                    "total_characters": len(total_text),
                }

        except pdfplumber.pdfminer.pdfparser.PDFSyntaxError:
            logger.error(f"Invalid PDF format: {file_path.name}")
            return {"success": False, "error": "Invalid or corrupted PDF file"}
        except Exception as e:
            logger.error(f"Error processing PDF {file_path.name}: {str(e)}")
            return {"success": False, "error": f"Processing error: {str(e)}"}

    def _process_page(self, page, page_num: int) -> Optional[Dict]:
        try:
            text = page.extract_text()
            if not text:
                text = ""

            tables = []
            extracted_tables = page.extract_tables()
            if extracted_tables:
                for table_idx, table in enumerate(extracted_tables):
                    table_text = self._table_to_text(table, table_idx)
                    tables.append({
                        "table_id": table_idx,
                        "text": table_text,
                        "rows": len(table),
                        "cols": len(table[0]) if table else 0
                    })

            width = page.width
            height = page.height

            return {
                "page_num": page_num,
                "text": text.strip(),
                "tables": tables,
                "has_tables": len(tables) > 0,
                "char_count": len(text),
                "dimensions": {"width": width, "height": height}
            }

        except Exception as e:
            logger.warning(f"Error processing page {page_num}: {str(e)}")
            return None

    def _table_to_text(self, table: List[List], table_idx: int) -> str:
        if not table:
            return ""

        lines = [f"[Table {table_idx + 1}]"]

        header = table[0]
        header_str = " | ".join([str(cell) if cell else "" for cell in header])
        lines.append(header_str)
        lines.append("-" * len(header_str))

        for row in table[1:]:
            row_str = " | ".join([str(cell) if cell else "" for cell in row])
            lines.append(row_str)

        return "\n".join(lines)

    def _extract_metadata(self, pdf, file_path: Path) -> Dict:
        metadata = {
            "filename": file_path.name,
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "processed_date": datetime.now().isoformat(),
        }

        if hasattr(pdf, 'metadata') and pdf.metadata:
            pdf_info = pdf.metadata
            metadata.update({
                "title": pdf_info.get("Title", ""),
                "author": pdf_info.get("Author", ""),
                "subject": pdf_info.get("Subject", ""),
                "creator": pdf_info.get("Creator", ""),
                "producer": pdf_info.get("Producer", ""),
                "creation_date": pdf_info.get("CreationDate", ""),
            })

        return metadata

    def get_full_text(self, processed_doc: Dict) -> str:
        if not processed_doc.get("success"):
            return ""

        text_parts = []
        for page in processed_doc["pages"]:
            text_parts.append(page["text"])

            for table in page.get("tables", []):
                text_parts.append(table["text"])

        return "\n\n".join(text_parts)

    def get_page_text(self, processed_doc: Dict, page_num: int) -> str:
        if not processed_doc.get("success"):
            return ""

        for page in processed_doc["pages"]:
            if page["page_num"] == page_num:
                text = page["text"]
                for table in page.get("tables", []):
                    text += "\n\n" + table["text"]
                return text

        return ""


def process_pdf_file(file_path: Path) -> Dict:
    processor = DocumentProcessor()
    return processor.process_pdf(file_path)
