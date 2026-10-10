"""
setup_check.py — Dependency verification script for JobFair AI.
Run before first use: python setup_check.py
"""

import importlib
import io
import sys


def _check(package: str, import_name: str | None = None) -> tuple[bool, str]:
    """Try to import a package and return (success, version_or_error)."""
    mod_name = import_name or package
    try:
        mod = importlib.import_module(mod_name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError as exc:
        return False, str(exc)


def main() -> int:
    """Run all checks and print a summary. Returns exit code (0=ok, 1=critical failures)."""
    # Force UTF-8 output on Windows to handle unicode symbols
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    checks = [
        # (display_name, import_name, critical)
        ("pandas", "pandas", True),
        ("numpy", "numpy", True),
        ("scikit-learn", "sklearn", True),
        ("xgboost", "xgboost", False),
        ("IBM AI Fairness 360 (aif360)", "aif360", False),
        ("Fairlearn", "fairlearn", False),
        ("SHAP", "shap", True),
        ("Streamlit", "streamlit", True),
        ("Plotly", "plotly", True),
        ("SQLAlchemy", "sqlalchemy", True),
        ("Jinja2", "jinja2", True),
        ("WeasyPrint", "weasyprint", False),
        ("ReportLab", "reportlab", False),
        ("scipy", "scipy", True),
    ]

    print("\n" + "=" * 60)
    print("  JobFair AI — Dependency Check")
    print("=" * 60)

    critical_failures: list[str] = []
    warnings: list[str] = []

    for display_name, import_name, critical in checks:
        ok, info = _check(import_name)
        if ok:
            print(f"  ✓  {display_name:<35} v{info}")
        else:
            marker = "✗" if critical else "⚠"
            severity = "CRITICAL" if critical else "WARNING"
            print(f"  {marker}  {display_name:<35} [{severity}] {info[:60]}")
            if critical:
                critical_failures.append(display_name)
            else:
                warnings.append(display_name)

    print("=" * 60)

    # Fairness toolkit fallback logic
    aif_ok, _ = _check("aif360")
    fl_ok, _ = _check("fairlearn")
    if not aif_ok and not fl_ok:
        print("  ✗  FATAL: Neither aif360 nor fairlearn is available.")
        print("     Install at least one: pip install fairlearn")
        critical_failures.append("aif360 / fairlearn")
    elif not aif_ok and fl_ok:
        print("  ⚠  aif360 not found — Fairlearn fallback will be used for all fairness metrics.")
        print("     This is fully supported. All features remain available.")
    elif aif_ok and not fl_ok:
        print("  ⚠  fairlearn not found — ExponentiatedGradient (in-processing) will be unavailable.")

    # PDF generation fallback
    wp_ok, _ = _check("weasyprint")
    rl_ok, _ = _check("reportlab")
    if not wp_ok and not rl_ok:
        print("  ✗  FATAL: Neither weasyprint nor reportlab is available.")
        print("     Install at least one: pip install reportlab")
        critical_failures.append("weasyprint / reportlab")
    elif not wp_ok and rl_ok:
        print("  ⚠  weasyprint not found — ReportLab fallback will be used for PDF export.")
        print("     HTML export will work regardless.")

    print()
    if critical_failures:
        print(f"  ✗  {len(critical_failures)} critical package(s) missing. Fix before running the app.")
        print(f"     Missing: {', '.join(critical_failures)}")
        print("     Run: pip install -r requirements.txt")
        return 1
    elif warnings:
        print(f"  ⚠  {len(warnings)} optional package(s) unavailable. Fallbacks are in place.")
        print("  ✓  Core system is ready. Run: streamlit run app/Home.py")
        return 0
    else:
        print("  ✓  All dependencies satisfied. Run: streamlit run app/Home.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
