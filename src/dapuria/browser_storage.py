"""Mount the trusted IndexedDB bridge; conversation data is passed only as data."""

from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

from .browser_history import BrowserHistoryStore

_component = components.component(
    "dapuria_history_storage",
    js=(Path(__file__).resolve().parent / "assets/history_storage.js").read_text(encoding="utf-8"),
)


def _noop() -> None:
    pass


def connect_browser_history() -> BrowserHistoryStore:
    if "browser_history" not in st.session_state:
        st.session_state.browser_history = BrowserHistoryStore()
    store = st.session_state.browser_history
    request = store.request()
    response = _component(
        key="history_storage", data={"client_id": store.client_id, "request": request},
        height=0, on_response_change=_noop, on_changed_change=store.refresh,
    )
    try:
        received = store.receive(response.response)
    except (ValueError, TypeError, RecursionError):
        store.error = "Data riwayat browser tidak dapat dibaca. Coba muat ulang browser."
        received = False
    if received and not store.error:
        st.rerun()
    if not store.ready:
        if store.error:
            st.warning(store.error)
            st.button("Coba buka riwayat lagi", on_click=store.retry, key="retry_history")
        else:
            st.info("Membuka riwayat dari browser…")
        st.stop()
    return store
