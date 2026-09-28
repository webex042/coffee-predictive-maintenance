"""Web-API service layer for the Coffee Line predictive-maintenance system.

This package holds the *single-snapshot* prediction service used by the Vercel
web API. It deliberately reuses the trained model bundle, the machine-type
baselines in ``config.py`` and the health / maintenance-rule logic in ``src/``,
so the web API scores a machine with the SAME model and SAME rules as the
original Streamlit dashboard.
"""
