"""Presentation only: display recipe data without inventing quantities or steps."""

import re
from base64 import b64encode
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

ASSET_DIR = Path(__file__).resolve().parent / "assets"
MAIN_LOGO = ASSET_DIR / "image" / "logo_utama.png"
SECONDARY_LOGO = ASSET_DIR / "image" / "magnifying_glass_icon.png"


@lru_cache(maxsize=2)
def _image_uri(path: Path) -> str:
    """Embed bundled images in custom HTML without relying on the working directory."""
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def ingredient_names(recipe: dict) -> list[str]:
    names = recipe.get("ingredient_names")
    if not isinstance(names, list) or not names:
        names = list(
            dict.fromkeys(
                recipe.get("matched_recipe_items", []) + recipe.get("missing_items", [])
            )
        )
    return [str(name) for name in names]


def chips(items: list[str], missing: bool = False) -> str:
    cls = "chip missing" if missing else "chip"
    return (
        '<div class="chips">'
        + "".join(f'<span class="{cls}">{_text(item)}</span>' for item in items)
        + "</div>"
    )


def inject_styles() -> None:
    # Style-only HTML uses Streamlit's event container and leaves no layout gap.
    st.html(ASSET_DIR / "style.css")


def render_header() -> None:
    st.markdown(
        f"""<header class="brand"><img class="brand-icon" src="{_image_uri(MAIN_LOGO)}" alt="Logo Dapuria" width="64" height="64">
    <div><h1>Dapuria</h1><p>Asisten virtual resep masakan</p></div>
    <span class="brand-note"><span class="status-dot"></span>Dari dapurmu, jadi menumu</span></header>""",
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.markdown(
        f'<div class="sidebar-heading"><div class="sidebar-brand"><img src="{_image_uri(MAIN_LOGO)}" '
        'alt="" width="34" height="34"><span>Dapur pribadimu</span></div>',
        # '<p class="sidebar-description">Simpan, kelola, dan jelajahi resep yang kamu suka.</p></div>',
        unsafe_allow_html=True,
    )


def render_progress(phase: str, has_result: bool) -> None:
    active = 2 if has_result else (1 if phase == "await_confirmation" else 0)
    labels = ["01 · Ceritakan bahanmu", "02 · Cek pelengkap", "03 · Pilih resep"]
    html = '<nav class="flow" aria-label="Tahap pencarian">'
    for i, label in enumerate(labels):
        state = "active" if i == active else ("complete" if i < active else "")
        current = ' aria-current="step"' if i == active else ""
        html += f'<span class="{state}"{current}>{label}</span>'
    st.markdown(html + "</nav>", unsafe_allow_html=True)


@contextmanager
def chat_bubble(role: str, key: str) -> Iterator[None]:
    with (
        st.container(key=f"bubble_{role}_{key}"),
        st.chat_message(
            role,
            avatar=str(MAIN_LOGO) if role == "assistant" else "👤",
            width="content",
        ),
    ):
        yield


def render_welcome() -> None:
    st.markdown(
        f"""<section class="hero">
      <div class="hero-copy"><div class="eyebrow"><span class="tiny-line"></span>Mulai dari yang kamu punya</div>
      <h2>Ada apa di dapur<br><em>hari ini?</em></h2>
      <p>Bahan sederhana bisa jadi menu istimewa. Ceritakan isi dapurmu,
      kita temukan ide masaknya bersama.</p>
      <div class="hero-note"><span>✦</span> Resep dari bahanmu. Inspirasi untuk harimu.</div></div>
      <div class="kitchen-scene" aria-hidden="true">
        <div class="scene-orbit"></div><div class="scene-orbit second"></div>
        <img class="welcome-logo" src="{_image_uri(SECONDARY_LOGO)}" alt="" width="280" height="280">
        <div class="floating-label label-egg">✦ <span>Ide masak untukmu</span></div>
        <div class="floating-label label-leaf">✧ <span>Dari bahan di dapurmu</span></div>
        <div class="scene-caption">Sedikit bahan, banyak kemungkinan.</div>
      </div></section>""",
        unsafe_allow_html=True,
    )


@contextmanager
def activity(label: str) -> Iterator[None]:
    placeholder = st.empty()
    placeholder.markdown(
        f'<div class="activity" role="status" aria-live="polite">'
        f'<span class="activity-orbit" aria-hidden="true"></span>'
        f'<div><strong>{_text(label)}</strong><span class="activity-note">'
        'Sebentar, sedang disiapkan untukmu<span class="loading-dots" aria-hidden="true">'
        "<i></i><i></i><i></i></span></span></div></div>",
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


def split_cooking_steps(text: str | None) -> list[str]:
    """Separate source step numbers without splitting quantities or sentences."""
    if not text or not text.strip():
        return []
    text = text.strip()
    markers = []
    for match in re.finditer(r"(?<!\S)(\d+)[.)](?:\s+|(?=[^\W\d_]))(?=\S)", text):
        if int(match[1]) == len(markers) + 1 and (markers or match.start() == 0):
            markers.append(match)
    if markers:
        return [
            text[marker.end():markers[index + 1].start() if index + 1 < len(markers) else len(text)].strip()
            for index, marker in enumerate(markers)
        ]
    return [re.sub(r"^[-*•]\s+", "", line.strip()) for line in text.splitlines() if line.strip()]


def recipe_html(recipe: dict, rank: int) -> str:
    names = ingredient_names(recipe)
    available = recipe.get("matched_recipe_items", [])
    missing = recipe.get("missing_items", [])
    ingredients = "".join(f"<li>{_text(x)}</li>" for x in names)
    if not ingredients:
        ingredients = "<li>Daftar bahan belum tersedia.</li>"
    available_html = (
        chips(available)
        if available
        else '<p class="small-note">Belum ada bahan yang cocok tercatat.</p>'
    )
    missing_html = (
        chips(missing, True)
        if missing
        else '<p class="small-note">Tidak ada bahan tambahan yang tercatat.</p>'
    )
    steps = split_cooking_steps(recipe.get("steps_text"))
    steps_html = (
        '<ol class="steps-list">' + "".join(f"<li>{_text(step)}</li>" for step in steps) + "</ol>"
        if steps else '<p class="small-note">Langkah memasak belum tersedia pada data resep ini.</p>'
    )
    label = (
        "Pilihan utama · Peringkat 01"
        if rank == 1
        else f"Resep lainnya · Peringkat {rank:02}"
    )
    available_count = len(available)
    total = len(names)
    percentage = round(min(available_count / total, 1) * 100) if total else 0
    availability_meter = (
        f'<div class="availability-meter"><div class="meter-label">'
        f"<span>Bahan tersedia</span><strong>{available_count} dari {total}</strong></div>"
        f'<div class="meter-track" role="meter" aria-label="Jumlah bahan resep yang tersedia" '
        f'aria-valuenow="{available_count}" aria-valuemin="0" aria-valuemax="{total}">'
        f'<span style="width:{percentage}%"></span></div></div>'
        if total
        else ""
    )
    return f"""<article class="recipe-full">
      <div class="recipe-head"><div class="eyebrow">{label}</div>
      <h2>{_text(recipe.get("title") or "Resep tanpa judul")}</h2>
      <p>{len(names)} bahan dalam resep · {len(missing)} bahan belum tercatat tersedia</p>{availability_meter}</div>
      <div class="recipe-body">
        <section class="recipe-section"><h3>Bahan-bahan</h3><ul class="ingredients">{ingredients}</ul>
        </section>
        <section class="recipe-section availability"><div><h4>Bahan yang kamu punya</h4>{available_html}</div>
        <div><h4>Yang perlu disiapkan</h4>{missing_html}</div></section>
        <section><h3>Langkah memasak</h3>{steps_html}</section>
      </div></article>"""


def render_recipe(recipe: dict, rank: int) -> None:
    st.markdown(recipe_html(recipe, rank), unsafe_allow_html=True)
    # with st.expander("Detail penelitian", expanded=False):
    #     st.caption(
    #         "Skor digunakan untuk mengurutkan resep; bukan persentase kecocokan."
    #     )
    #     fields = [
    #         "sbert_score",
    #         "sbert_score_norm",
    #         "coverage_recipe",
    #         "coverage_user",
    #         "ingredient_efficiency",
    #         "final_score",
    #         "retrieval_source",
    #     ]
    #     st.json({key: recipe[key] for key in fields if key in recipe})


@st.dialog("Detail resep", width="large")
def recipe_dialog(recipe: dict, rank: int) -> None:
    render_recipe(recipe, rank)
    if st.button(
        "Kembali ke rekomendasi", use_container_width=True, key="close_recipe"
    ):
        st.rerun()


def render_results(result: dict) -> None:
    recipes = result.get("results", [])
    if not recipes:
        st.info("Belum ada resep untuk ditampilkan. Coba kombinasi bahan lain, ya.")
        return
    if result.get("user_ingredients"):
        st.caption("Bahan yang terbaca: " + ", ".join(result["user_ingredients"]))
        st.markdown(
            '<div class="pantry-summary"><span class="pantry-label">Dapurmu hari ini</span>'
            + chips(result["user_ingredients"])
            + "</div>",
            unsafe_allow_html=True,
        )
    if result.get("excluded_ingredients"):
        st.caption(
            "Batasan pada pencarian ini: " + ", ".join(result["excluded_ingredients"])
        )
    st.markdown(
        f"""<div class="result-intro"><div class="eyebrow">Dari bahan yang kamu punya</div>
      <h2>Yuk, masak yang ini.</h2><p>{len(recipes)} resep ditemukan. Pilihan utama diutamakan dari bahan yang paling siap.</p></div>""",
        unsafe_allow_html=True,
    )
    render_recipe(recipes[0], 1)
    if len(recipes) > 1:
        st.markdown(
            f"""<div class="other-heading"><h2>Resep lainnya</h2><span>{len(recipes) - 1} alternatif untukmu</span></div>""",
            unsafe_allow_html=True,
        )
        for rank, recipe in enumerate(recipes[1:], 2):
            with st.container(border=True):
                text_col, button_col = st.columns([4, 1], vertical_alignment="center")
                with text_col:
                    names = (
                        ", ".join(ingredient_names(recipe)) or "Bahan belum tersedia"
                    )
                    st.markdown(
                        f"""<div class="other-row"><span class="rank">{rank:02}</span><div>
                    <h3>{_text(recipe.get("title") or "Resep tanpa judul")}</h3>
                    <p><strong>Bahan:</strong> {_text(names)}</p></div></div>""",
                        unsafe_allow_html=True,
                    )
                    available = len(recipe.get("matched_recipe_items", []))
                    missing = len(recipe.get("missing_items", []))
                    st.caption(
                        f"{available} bahan tersedia · {missing} perlu disiapkan"
                    )
                with button_col:
                    if st.button(
                        "Lihat detail",
                        key=f"detail_recipe_{rank}",
                        use_container_width=True,
                        help=f"Buka resep peringkat {rank}: {recipe.get('title', '')}",
                    ):
                        recipe_dialog(recipe, rank)
    st.caption(
        "Pilih Lanjutkan percakapan untuk menambah bahan atau batasan, atau Cari resep baru untuk mulai lagi."
    )
