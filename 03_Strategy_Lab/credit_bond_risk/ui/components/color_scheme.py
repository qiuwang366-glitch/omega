"""
Credit Bond Risk - Color Scheme

Institutional-grade color palettes and Plotly layout templates.
"""

from dataclasses import dataclass


@dataclass
class ColorScheme:
    """Premium dark theme color scheme"""

    # Background
    bg_primary: str = "#0d1117"
    bg_secondary: str = "#161b22"
    bg_tertiary: str = "#21262d"

    # Text
    text_primary: str = "#f0f6fc"
    text_secondary: str = "#8b949e"
    text_muted: str = "#6e7681"

    # Accent colors
    accent_blue: str = "#58a6ff"
    accent_green: str = "#3fb950"
    accent_yellow: str = "#d29922"
    accent_orange: str = "#db6d28"
    accent_red: str = "#f85149"
    accent_purple: str = "#a371f7"

    # Severity colors
    severity_critical: str = "#f85149"
    severity_warning: str = "#d29922"
    severity_info: str = "#58a6ff"
    severity_success: str = "#3fb950"

    # Rating colors (gradient from AAA to CCC)
    rating_aaa: str = "#238636"
    rating_aa: str = "#3fb950"
    rating_a: str = "#7ee787"
    rating_bbb: str = "#d29922"
    rating_bb: str = "#db6d28"
    rating_b: str = "#f85149"
    rating_ccc: str = "#da3633"

    # Sector colors
    sector_lgfv: str = "#58a6ff"
    sector_soe: str = "#a371f7"
    sector_financial: str = "#3fb950"
    sector_corp: str = "#d29922"

    @classmethod
    def get_severity_color(cls, severity: str) -> str:
        """Get color for severity level"""
        scheme = cls()
        mapping = {
            "CRITICAL": scheme.severity_critical,
            "WARNING": scheme.severity_warning,
            "INFO": scheme.severity_info,
            "SUCCESS": scheme.severity_success,
        }
        return mapping.get(severity.upper(), scheme.text_secondary)

    @classmethod
    def get_rating_color(cls, rating: str) -> str:
        """Get color for credit rating"""
        scheme = cls()
        rating_upper = rating.upper().replace("+", "_PLUS").replace("-", "_MINUS")

        if "AAA" in rating_upper:
            return scheme.rating_aaa
        elif "AA" in rating_upper:
            return scheme.rating_aa
        elif rating_upper.startswith("A"):
            return scheme.rating_a
        elif "BBB" in rating_upper:
            return scheme.rating_bbb
        elif "BB" in rating_upper:
            return scheme.rating_bb
        elif rating_upper.startswith("B"):
            return scheme.rating_b
        else:
            return scheme.rating_ccc

    @classmethod
    def get_sector_color(cls, sector: str) -> str:
        """Get color for sector"""
        scheme = cls()
        mapping = {
            "LGFV": scheme.sector_lgfv,
            "SOE": scheme.sector_soe,
            "FINANCIAL": scheme.sector_financial,
            "CORP": scheme.sector_corp,
        }
        return mapping.get(sector.upper(), scheme.text_secondary)


def get_premium_layout(title: str = "", height: int = 400) -> dict:
    """
    Get premium Plotly layout template

    Args:
        title: Chart title
        height: Chart height in pixels

    Returns:
        Plotly layout dict
    """
    scheme = ColorScheme()

    return {
        "title": {
            "text": title,
            "font": {"size": 16, "color": scheme.text_primary},
            "x": 0.02,
            "xanchor": "left",
        },
        "paper_bgcolor": scheme.bg_primary,
        "plot_bgcolor": scheme.bg_secondary,
        "height": height,
        "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
        "font": {
            "family": "Inter, SF Pro Display, -apple-system, sans-serif",
            "color": scheme.text_secondary,
        },
        "xaxis": {
            "gridcolor": scheme.bg_tertiary,
            "linecolor": scheme.bg_tertiary,
            "tickfont": {"color": scheme.text_secondary},
        },
        "yaxis": {
            "gridcolor": scheme.bg_tertiary,
            "linecolor": scheme.bg_tertiary,
            "tickfont": {"color": scheme.text_secondary},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": scheme.text_secondary},
        },
        "hoverlabel": {
            "bgcolor": scheme.bg_tertiary,
            "font": {"color": scheme.text_primary},
        },
    }
