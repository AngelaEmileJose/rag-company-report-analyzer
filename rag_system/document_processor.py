import logging
import hashlib
import aiofiles
import aiohttp
import re
import pymupdf
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timedelta
from typing import Tuple

from .config import RAGConfig

class DocumentProcessor:
    """Handles document downloading and text extraction"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger("rag_system.document_processor")

    async def process_source(self, source: str) -> Tuple[str, str]:
        """Process a source (URL or file path) and extract text"""
        if self._is_url(source):
            return await self._download_and_extract(source)
        else:
            return await self._extract_from_file(source)

    def _is_url(self, source: str) -> bool:
        """Check if source is a URL"""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    async def _download_and_extract(self, url: str) -> Tuple[str, str]:
        """Download PDF from URL and extract text"""
        # Generate cache key
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.config.cache_dir / f"{cache_key}.txt"

        # Check cache
        if self.config.enable_cache and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=self.config.cache_ttl_hours):
                self.logger.info(f"Using cached content for {url}")
                async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                    return await f.read(), url

        # Download file/content
        self.logger.info(f"Downloading {url}")
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        connector = aiohttp.TCPConnector(ssl=False)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/"
            }
            async with session.get(url, headers=headers) as response:
                if response.status == 403:
                     # Fallback to urllib for 403 (often works where aiohttp fails)
                     self.logger.warning("aiohttp failed with 403, trying fallback...")
                     import urllib.request
                     req = urllib.request.Request(
                         url, 
                         data=None, 
                         headers={
                             'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                         }
                     )
                     with urllib.request.urlopen(req) as f:
                         content = f.read()
                         content_type = f.headers.get_content_type()
                         
                elif response.status != 200:
                    raise Exception(f"Failed to download: HTTP {response.status}")
                else:
                    content_type = response.headers.get('Content-Type', '').lower()
                    content = await response.read()

                # Check file size (approximate for urllib)
                if len(content) > self.config.max_file_size_mb * 1024 * 1024:
                     raise Exception(f"File too large: {len(content) / (1024*1024):.1f}MB")


        # Handle Content Type
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            # Save PDF
            temp_file = self.config.cache_dir / f"{cache_key}.pdf"
            async with aiofiles.open(temp_file, 'wb') as f:
                await f.write(content)
            
            try:
                text = self._extract_pdf_text(temp_file)
            finally:
                if temp_file.exists():
                    temp_file.unlink()
        else:
            # Assume HTML/Text
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                    script.extract()
                
                # Get text
                text = soup.get_text(separator='\n\n')
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
            except ImportError:
                # Fallback if bs4 not installed
                text = content.decode('utf-8', errors='ignore')
                
        # Cache the extracted text
        if self.config.enable_cache:
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(text)

        return text, url

    async def _extract_from_file(self, file_path: str) -> Tuple[str, str]:
        """Extract text from local file"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == '.pdf':
            text = self._extract_pdf_text(path)
        elif path.suffix.lower() == '.txt':
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                text = await f.read()
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return text, str(path)

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = pymupdf.open(str(pdf_path))
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Clean up the text
                text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
                text = re.sub(r'\s+', ' ', text)   # Multiple spaces to single
                text = text.strip()

                if text:  # Only add non-empty pages
                    text_parts.append(text)

            doc.close()
            return '\n\n'.join(text_parts)

        except Exception as e:
            raise Exception(f"Failed to extract PDF text: {str(e)}")
