"""
Credit Bond Risk - Filing Parser Module

Parses corporate filings, announcements, and financial reports.

Supported Filing Types:
- Annual Reports (年报)
- Semi-annual Reports (半年报)
- Bond Prospectuses (募集说明书)
- Rating Reports (评级报告)
- Material Event Announcements (重大事项公告)
- SEC/EDGAR Filings (10-K, 10-Q, 8-K)

Usage:
    from data.filing_parser import FilingParser, FilingType
    parser = FilingParser()
    result = parser.parse_file("path/to/annual_report.pdf")
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FilingType(str, Enum):
    """Types of corporate filings"""
    # China
    ANNUAL_REPORT = "annual_report"           # 年报
    SEMI_ANNUAL_REPORT = "semi_annual_report" # 半年报
    QUARTERLY_REPORT = "quarterly_report"     # 季报
    BOND_PROSPECTUS = "bond_prospectus"       # 募集说明书
    RATING_REPORT = "rating_report"           # 评级报告
    MATERIAL_EVENT = "material_event"         # 重大事项公告
    AUDIT_REPORT = "audit_report"             # 审计报告

    # US/International
    SEC_10K = "sec_10k"      # Annual report (SEC)
    SEC_10Q = "sec_10q"      # Quarterly report (SEC)
    SEC_8K = "sec_8k"        # Material event (SEC)
    EARNINGS_RELEASE = "earnings_release"
    INVESTOR_PRESENTATION = "investor_presentation"

    # Credit-specific
    COVENANT_COMPLIANCE = "covenant_compliance"
    DEBT_MATURITY_SCHEDULE = "debt_maturity_schedule"


class FilingSource(str, Enum):
    """Sources for filings"""
    COMPANY_WEBSITE = "company_website"
    CNINFO = "cninfo"           # 巨潮资讯
    EDGAR = "edgar"             # SEC EDGAR
    HKEX = "hkex"               # 港交所
    BLOOMBERG = "bloomberg"
    WIND = "wind"


@dataclass
class FinancialMetrics:
    """Extracted financial metrics from filings"""
    # P&L
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    ebitda: float | None = None

    # Balance Sheet
    total_assets: float | None = None
    total_liabilities: float | None = None
    total_equity: float | None = None
    total_debt: float | None = None
    cash_and_equivalents: float | None = None
    short_term_debt: float | None = None
    long_term_debt: float | None = None

    # Cash Flow
    operating_cash_flow: float | None = None
    capex: float | None = None
    free_cash_flow: float | None = None

    # Ratios
    debt_to_equity: float | None = None
    debt_to_ebitda: float | None = None
    interest_coverage: float | None = None
    current_ratio: float | None = None

    # Credit specific
    bond_outstanding: float | None = None
    credit_facilities: float | None = None
    restricted_cash: float | None = None

    currency: str = "CNY"
    unit: str = "亿"  # Billions, Millions, etc.
    period_end: date | None = None


@dataclass
class DebtMaturity:
    """Debt maturity schedule entry"""
    amount: float
    maturity_date: date
    instrument_type: str  # Bond, Loan, CP, MTN, etc.
    currency: str = "CNY"
    description: str = ""


@dataclass
class CovenantStatus:
    """Covenant compliance status"""
    covenant_name: str
    threshold: float
    actual_value: float
    is_compliant: bool
    headroom_pct: float | None = None
    notes: str = ""


@dataclass
class ParsedFiling:
    """Result of parsing a filing"""
    # Metadata
    filing_id: str
    filing_type: FilingType
    obligor_id: str | None = None
    obligor_name: str | None = None
    source: FilingSource = FilingSource.COMPANY_WEBSITE

    # Dates
    filing_date: date | None = None
    period_end: date | None = None
    parsed_at: datetime = field(default_factory=datetime.now)

    # Content
    title: str = ""
    raw_text: str = ""
    summary: str | None = None

    # Extracted data
    financials: FinancialMetrics | None = None
    debt_maturities: list[DebtMaturity] = field(default_factory=list)
    covenants: list[CovenantStatus] = field(default_factory=list)

    # AI analysis
    key_points: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    sentiment: str | None = None  # POSITIVE, NEUTRAL, NEGATIVE

    # Raw extracted tables
    tables: list[dict] = field(default_factory=list)

    # Errors
    parse_errors: list[str] = field(default_factory=list)


class FilingParserConfig(BaseModel):
    """Configuration for filing parser"""
    # OCR settings
    enable_ocr: bool = True
    ocr_language: str = "chi_sim+eng"

    # LLM settings
    enable_llm_extraction: bool = True
    llm_model: str = "claude-3-5-haiku-20241022"

    # Table extraction
    extract_tables: bool = True
    table_detection_threshold: float = 0.5

    # Output
    save_raw_text: bool = True
    max_text_length: int = 100000


# =============================================================================
# Abstract Filing Parser
# =============================================================================


class BaseFilingParser(ABC):
    """Abstract base class for filing parsers"""

    SUPPORTED_TYPES: list[FilingType] = []

    def __init__(self, config: FilingParserConfig | None = None):
        self.config = config or FilingParserConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def parse(self, file_path: str | Path, filing_type: FilingType | None = None) -> ParsedFiling:
        """Parse a filing document"""
        pass

    @abstractmethod
    def extract_financials(self, text: str) -> FinancialMetrics:
        """Extract financial metrics from text"""
        pass

    def detect_filing_type(self, file_path: str | Path, text: str | None = None) -> FilingType | None:
        """Auto-detect filing type from filename or content"""
        path = Path(file_path)
        name_lower = path.stem.lower()

        # Check filename patterns
        patterns = {
            "annual_report": [r"年报", r"annual.?report", r"10-?k"],
            "semi_annual_report": [r"半年报", r"semi.?annual", r"中报"],
            "quarterly_report": [r"季报", r"quarterly", r"10-?q"],
            "bond_prospectus": [r"募集说明书", r"prospectus", r"offering"],
            "rating_report": [r"评级报告", r"rating.?report"],
            "material_event": [r"公告", r"8-?k", r"announcement"],
        }

        for filing_type, keywords in patterns.items():
            for kw in keywords:
                if re.search(kw, name_lower):
                    return FilingType(filing_type)

        return None


# =============================================================================
# PDF Parser Implementation
# =============================================================================


class PDFFilingParser(BaseFilingParser):
    """
    Parser for PDF filings using multiple extraction methods.

    Requires:
    - pdfplumber: pip install pdfplumber
    - pytesseract (optional, for OCR): pip install pytesseract

    Usage:
        parser = PDFFilingParser()
        result = parser.parse("annual_report.pdf")
    """

    SUPPORTED_TYPES = list(FilingType)

    def parse(self, file_path: str | Path, filing_type: FilingType | None = None) -> ParsedFiling:
        """Parse a PDF filing"""
        path = Path(file_path)

        if not path.exists():
            return ParsedFiling(
                filing_id=path.stem,
                filing_type=filing_type or FilingType.ANNUAL_REPORT,
                parse_errors=[f"File not found: {file_path}"],
            )

        # Detect filing type
        if filing_type is None:
            filing_type = self.detect_filing_type(path) or FilingType.ANNUAL_REPORT

        # Extract text
        text = self._extract_text(path)

        # Create result
        result = ParsedFiling(
            filing_id=path.stem,
            filing_type=filing_type,
            title=path.stem,
            raw_text=text[:self.config.max_text_length] if self.config.save_raw_text else "",
        )

        # Extract financials
        if text:
            try:
                result.financials = self.extract_financials(text)
            except Exception as e:
                result.parse_errors.append(f"Financial extraction failed: {e}")

            # Extract tables
            if self.config.extract_tables:
                try:
                    result.tables = self._extract_tables(path)
                except Exception as e:
                    result.parse_errors.append(f"Table extraction failed: {e}")

        return result

    def _extract_text(self, path: Path) -> str:
        """Extract text from PDF"""
        try:
            import pdfplumber

            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return "\n".join(text_parts)

        except ImportError:
            self.logger.warning("pdfplumber not installed. Run: pip install pdfplumber")
            return ""
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return ""

    def _extract_tables(self, path: Path) -> list[dict]:
        """Extract tables from PDF"""
        try:
            import pdfplumber

            tables = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for j, table in enumerate(page_tables):
                        tables.append({
                            "page": i + 1,
                            "table_index": j,
                            "data": table,
                        })

            return tables

        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
            return []

    def extract_financials(self, text: str) -> FinancialMetrics:
        """Extract financial metrics using regex patterns"""
        metrics = FinancialMetrics()

        # Chinese patterns
        patterns_cn = {
            "revenue": [r"营业收入[：:]\s*([\d,.]+)", r"营收[：:]\s*([\d,.]+)"],
            "net_income": [r"净利润[：:]\s*([\d,.]+)", r"归母净利润[：:]\s*([\d,.]+)"],
            "total_assets": [r"总资产[：:]\s*([\d,.]+)", r"资产总额[：:]\s*([\d,.]+)"],
            "total_debt": [r"总负债[：:]\s*([\d,.]+)", r"有息负债[：:]\s*([\d,.]+)"],
            "total_equity": [r"净资产[：:]\s*([\d,.]+)", r"所有者权益[：:]\s*([\d,.]+)"],
        }

        # English patterns
        patterns_en = {
            "revenue": [r"Revenue[:\s]*([\d,.]+)", r"Total Revenue[:\s]*([\d,.]+)"],
            "net_income": [r"Net Income[:\s]*([\d,.]+)", r"Net Profit[:\s]*([\d,.]+)"],
            "total_assets": [r"Total Assets[:\s]*([\d,.]+)"],
            "total_debt": [r"Total Debt[:\s]*([\d,.]+)", r"Total Borrowings[:\s]*([\d,.]+)"],
            "ebitda": [r"EBITDA[:\s]*([\d,.]+)"],
        }

        # Try Chinese patterns first
        for metric, pats in patterns_cn.items():
            for pat in pats:
                match = re.search(pat, text)
                if match:
                    try:
                        value = float(match.group(1).replace(",", ""))
                        setattr(metrics, metric, value)
                        break
                    except ValueError:
                        pass

        # Try English patterns
        for metric, pats in patterns_en.items():
            if getattr(metrics, metric) is None:  # Don't override if already found
                for pat in pats:
                    match = re.search(pat, text, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1).replace(",", ""))
                            setattr(metrics, metric, value)
                            break
                        except ValueError:
                            pass

        return metrics


# =============================================================================
# EDGAR Parser (SEC Filings)
# =============================================================================


class EDGARParser(BaseFilingParser):
    """
    Parser for SEC EDGAR filings (10-K, 10-Q, 8-K).

    Uses EDGAR API to fetch and parse filings.
    """

    SUPPORTED_TYPES = [FilingType.SEC_10K, FilingType.SEC_10Q, FilingType.SEC_8K]

    EDGAR_BASE_URL = "https://data.sec.gov"

    def parse(self, file_path: str | Path, filing_type: FilingType | None = None) -> ParsedFiling:
        """Parse an EDGAR filing (can be URL or local file)"""
        # If it's a URL, fetch it
        if str(file_path).startswith("http"):
            text = self._fetch_from_edgar(str(file_path))
        else:
            path = Path(file_path)
            if path.exists():
                text = path.read_text()
            else:
                return ParsedFiling(
                    filing_id=str(file_path),
                    filing_type=filing_type or FilingType.SEC_10K,
                    parse_errors=[f"File not found: {file_path}"],
                )

        result = ParsedFiling(
            filing_id=Path(file_path).stem if not str(file_path).startswith("http") else str(file_path),
            filing_type=filing_type or FilingType.SEC_10K,
            source=FilingSource.EDGAR,
            raw_text=text[:self.config.max_text_length] if self.config.save_raw_text else "",
        )

        if text:
            result.financials = self.extract_financials(text)

        return result

    def _fetch_from_edgar(self, url: str) -> str:
        """Fetch filing from EDGAR"""
        try:
            import requests
            headers = {"User-Agent": "CreditRiskPlatform/1.0"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.error(f"EDGAR fetch failed: {e}")
            return ""

    def extract_financials(self, text: str) -> FinancialMetrics:
        """Extract financials from SEC filing"""
        # EDGAR filings often use XBRL tags
        metrics = FinancialMetrics(currency="USD", unit="millions")

        patterns = {
            "revenue": [r"Revenues?\s*[:\$]*\s*([\d,.]+)", r"Total\s+revenues?\s*[:\$]*\s*([\d,.]+)"],
            "net_income": [r"Net\s+income\s*[:\$]*\s*([\d,.]+)", r"Net\s+earnings\s*[:\$]*\s*([\d,.]+)"],
            "total_assets": [r"Total\s+assets\s*[:\$]*\s*([\d,.]+)"],
            "total_debt": [r"Total\s+debt\s*[:\$]*\s*([\d,.]+)", r"Long-term\s+debt\s*[:\$]*\s*([\d,.]+)"],
        }

        for metric, pats in patterns.items():
            for pat in pats:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1).replace(",", ""))
                        setattr(metrics, metric, value)
                        break
                    except ValueError:
                        pass

        return metrics


# =============================================================================
# Unified Filing Parser
# =============================================================================


class FilingParser:
    """
    Unified filing parser with automatic format detection.

    Usage:
        parser = FilingParser()
        result = parser.parse_file("annual_report.pdf")
        result = parser.parse_url("https://sec.gov/...")
    """

    def __init__(self, config: FilingParserConfig | None = None):
        self.config = config or FilingParserConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize parsers
        self._pdf_parser = PDFFilingParser(self.config)
        self._edgar_parser = EDGARParser(self.config)

    def parse_file(
        self,
        file_path: str | Path,
        filing_type: FilingType | None = None,
    ) -> ParsedFiling:
        """
        Parse a filing from a local file.

        Args:
            file_path: Path to the filing
            filing_type: Optional explicit filing type

        Returns:
            ParsedFiling with extracted data
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._pdf_parser.parse(path, filing_type)
        elif ext in [".html", ".htm", ".txt"]:
            return self._edgar_parser.parse(path, filing_type)
        else:
            return ParsedFiling(
                filing_id=path.stem,
                filing_type=filing_type or FilingType.ANNUAL_REPORT,
                parse_errors=[f"Unsupported file format: {ext}"],
            )

    def parse_url(
        self,
        url: str,
        filing_type: FilingType | None = None,
    ) -> ParsedFiling:
        """
        Parse a filing from URL.

        Args:
            url: URL to the filing
            filing_type: Optional explicit filing type

        Returns:
            ParsedFiling with extracted data
        """
        if "sec.gov" in url or "edgar" in url.lower():
            return self._edgar_parser.parse(url, filing_type)
        else:
            # Try to download and parse
            return ParsedFiling(
                filing_id=url,
                filing_type=filing_type or FilingType.ANNUAL_REPORT,
                parse_errors=["URL parsing not yet implemented for non-EDGAR sources"],
            )

    def extract_debt_schedule(self, filing: ParsedFiling) -> list[DebtMaturity]:
        """
        Extract debt maturity schedule from a parsed filing.

        Args:
            filing: Previously parsed filing

        Returns:
            List of DebtMaturity entries
        """
        maturities = []

        # Look for debt maturity tables
        for table in filing.tables:
            data = table.get("data", [])
            if not data:
                continue

            # Check if this looks like a maturity table
            header = data[0] if data else []
            header_str = " ".join(str(h).lower() for h in header if h)

            if any(kw in header_str for kw in ["maturity", "到期", "due", "repayment"]):
                # Parse the table
                for row in data[1:]:
                    try:
                        # Try to extract amount and date
                        # This is simplified - real implementation would be more robust
                        pass
                    except Exception:
                        pass

        return maturities


# =============================================================================
# Convenience Functions
# =============================================================================


def parse_annual_report(file_path: str | Path) -> ParsedFiling:
    """Quick helper to parse an annual report"""
    parser = FilingParser()
    return parser.parse_file(file_path, FilingType.ANNUAL_REPORT)


def parse_prospectus(file_path: str | Path) -> ParsedFiling:
    """Quick helper to parse a bond prospectus"""
    parser = FilingParser()
    return parser.parse_file(file_path, FilingType.BOND_PROSPECTUS)


def parse_sec_10k(file_path: str | Path) -> ParsedFiling:
    """Quick helper to parse SEC 10-K"""
    parser = FilingParser()
    return parser.parse_file(file_path, FilingType.SEC_10K)
