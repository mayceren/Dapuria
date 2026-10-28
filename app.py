from __future__ import annotations

import logging
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapuria import RecipeService, Settings
from dapuria.chat_history import STATE_KEYS, ChatHistoryStore
from dapuria.llama_service import LlamaService
from dapuria.pantry import update_inventory
from dapuria.preferences import ingredient_is_excluded, update_preferences
from dapuria.preprocessing import parse_user_ingredients
from dapuria.ui import (
    MAIN_LOGO,
    activity,
    chat_bubble,
    inject_styles,
    render_header,
    render_progress,
    render_results,
    render_sidebar_brand,
    render_welcome,
)

st.set_page_config(page_title="Dapuria · Masak dari bahanmu", page_icon=str(MAIN_LOGO), layout="wide", initial_sidebar_state="auto")
inject_styles()
LOGGER = logging.getLogger(__name__)
OWNER_ID = "local"


@st.cache_resource(show_spinner=False)
def load_services():
    settings = Settings()
    service = RecipeService(settings)
    llama = LlamaService(settings)
    return settings, service, llama


@st.cache_resource(show_spinner=False)
def load_history(path: str) -> ChatHistoryStore:
    return ChatHistoryStore(Path(path))


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.phase = "await_query"
    st.session_state.pending = None
    st.session_state.latest_result = None
    st.session_state.selected_result = None
    st.session_state.inventory = []
    st.session_state.submission = None
    st.session_state.excluded_ingredients = []
    st.session_state.conversation_id = str(uuid4())
    st.query_params["chat"] = st.session_state.conversation_id


def persist_chat() -> None:
    if history is None:
        return
    try:
        history.save(OWNER_ID, st.session_state.conversation_id, dict(st.session_state))
        st.session_state.history_error = None
    except (OSError, sqlite3.Error, ValueError, TypeError):
        LOGGER.exception("Could not save conversation")
        st.session_state.history_error = "Riwayat belum tersimpan. Percakapan tetap bisa dilanjutkan di sesi ini."


def finish_turn() -> None:
    st.session_state.submission = None
    messages = st.session_state.messages
    if (messages and messages[-1]["role"] == "assistant"
            and st.session_state.latest_result and st.session_state.phase == "await_query"):
        messages[-1]["result"] = st.session_state.latest_result
    persist_chat()
    st.rerun()


def open_chat(conversation_id: str) -> bool:
    if history is None:
        return False
    try:
        snapshot = history.load(OWNER_ID, conversation_id)
    except (OSError, sqlite3.Error, ValueError):
        LOGGER.exception("Could not open conversation")
        st.warning("Percakapan ini belum dapat dibuka. Coba percakapan lain atau mulai yang baru.")
        return False
    if snapshot is None:
        return False
    for key in STATE_KEYS:
        st.session_state[key] = snapshot[key]
    st.session_state.conversation_id = conversation_id
    st.session_state.selected_result = None
    st.session_state.submission = None
    st.query_params["chat"] = conversation_id
    return True


def search_recipes(service, query: str) -> dict:
    try:
        if st.session_state.excluded_ingredients:
            return service.handle_user_input(query, excluded_ingredients=st.session_state.excluded_ingredients)
        return service.handle_user_input(query)
    except Exception:
        LOGGER.exception("Recipe search failed")
        return {"status": "error", "message": "Pencarian resep sedang mengalami kendala. Coba kirim bahanmu lagi, ya."}


@contextmanager
def processing_guard():
    try:
        yield
    except Exception:
        LOGGER.exception("Could not complete chat turn")
        st.session_state.messages.append({"role": "assistant", "content":
            "Prosesnya belum berhasil diselesaikan. Bahanmu tetap tersimpan; coba lagi, ya."})
        finish_turn()
    finally:
        st.session_state.submission = None


def queue_message(text: str | None = None) -> None:
    if st.session_state.submission is None:
        value = text if text is not None else st.session_state.get("chat_composer", "")
        if value and value.strip():
            st.session_state.submission = {"kind": "message", "text": value.strip()}


def continue_chat() -> None:
    st.session_state.phase = "await_query"
    st.session_state.latest_result = None
    st.session_state.selected_result = None
    persist_chat()


def close_rename_dialog() -> None:
    st.session_state.renaming_chat_id = None


def close_history_menu(chat_id: str) -> None:
    st.session_state[f"history_menu_{chat_id}"] = False


def begin_rename(chat_id: str) -> None:
    close_history_menu(chat_id)
    st.session_state.renaming_chat_id = chat_id


@st.dialog("Ubah nama percakapan", width="small", on_dismiss=close_rename_dialog)
def rename_chat_dialog(item: dict) -> None:
    chat_id = item["id"]
    with st.form(f"rename_chat_{chat_id}", border=False, enter_to_submit=True):
        title = st.text_input("Judul percakapan", value=item["title"], max_chars=72,
                              key=f"chat_title_{chat_id}", disabled=busy)
        # Enter submits the first submit button; keep Save before Cancel.
        save_col, cancel_col = st.columns(2, wrap=False)
        save = save_col.form_submit_button("Simpan judul", key=f"rename_chat_{chat_id}",
                                           type="primary", disabled=busy, use_container_width=True)
        cancel = cancel_col.form_submit_button("Batal", key=f"cancel_rename_{chat_id}",
                                               use_container_width=True)
        if cancel:
            close_rename_dialog()
            st.rerun()
        if save:
            try:
                history.rename(OWNER_ID, chat_id, title)
                close_rename_dialog()
                st.rerun()
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "Judul belum dapat disimpan.")


def history_actions(item: dict) -> None:
    chat_id = item["id"]
    if st.button("Lepas pin" if item["pinned"] else "Pin chat", key=f"pin_chat_{chat_id}",
                 icon=":material/keep_off:" if item["pinned"] else ":material/keep:",
                 disabled=busy, use_container_width=True, on_click=close_history_menu, args=(chat_id,)):
        try:
            history.set_pinned(OWNER_ID, chat_id, not item["pinned"])
            st.rerun()
        except (OSError, sqlite3.Error):
            st.error("Status pin belum dapat disimpan. Coba lagi nanti.")
    st.button("Ubah nama", key=f"open_rename_{chat_id}", icon=":material/edit:",
              disabled=busy, use_container_width=True, on_click=begin_rename, args=(chat_id,))
    if st.button("Hapus chat", key=f"delete_chat_{chat_id}", icon=":material/delete:",
                 disabled=busy, use_container_width=True, on_click=close_history_menu, args=(chat_id,)):
        try:
            history.delete(OWNER_ID, chat_id)
            if chat_id == st.session_state.conversation_id:
                reset_chat()
            st.rerun()
        except (OSError, sqlite3.Error):
            st.error("Percakapan belum dapat dihapus. Coba lagi nanti.")


for key, default in {
    "messages": [],
    "phase": "await_query",
    "pending": None,
    "latest_result": None,
    "selected_result": None,
    "history_error": None,
    "inventory": [],
    "submission": None,
    "excluded_ingredients": [],
    "renaming_chat_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

render_header()

try:
    history = load_history(os.getenv("DAPURIA_HISTORY_PATH", str(ROOT / "data" / "local_history" / "chats.sqlite3")))
except (OSError, sqlite3.Error):
    LOGGER.exception("Could not initialize history")
    history = None
    st.session_state.history_error = "Penyimpanan riwayat belum tersedia. Chat tetap bisa digunakan selama sesi ini."

requested_chat = st.query_params.get("chat")
if "conversation_id" not in st.session_state:
    if not requested_chat or not open_chat(requested_chat):
        reset_chat()
elif (requested_chat and requested_chat != st.session_state.conversation_id
      and not open_chat(requested_chat)):
    reset_chat()

busy = st.session_state.submission is not None
conversations = []
if history is not None:
    try:
        conversations = history.list(OWNER_ID)
    except (OSError, sqlite3.Error):
        st.session_state.history_error = "Daftar riwayat belum dapat dibuka. Chat tetap bisa digunakan."

with st.sidebar:
    render_sidebar_brand()
    if st.button("Mulai percakapan baru", use_container_width=True, type="primary", icon=":material/add:", disabled=busy):
        persist_chat()
        reset_chat()
        st.rerun()
    st.divider()
    st.markdown('<div class="sidebar-kicker">Riwayat percakapan</div>', unsafe_allow_html=True)
    search = st.text_input("Cari percakapan", placeholder="Cari judul percakapan…", key="history_search", label_visibility="collapsed", disabled=busy)
    filtered = [item for item in conversations if search.strip().casefold() in item["title"].casefold()]
    if not filtered:
        st.caption("Tidak ada percakapan yang cocok." if search.strip() else "Percakapanmu akan tersimpan otomatis di sini.")
    group = None
    jakarta = timezone(timedelta(hours=7))
    today = datetime.now(jakarta).date()
    for item in filtered:
        updated = datetime.fromisoformat(item["updated_at"]).astimezone(jakarta)
        day = updated.date()
        label = ("Disematkan" if item["pinned"] else "Hari ini" if day == today
                 else "Kemarin" if day == today - timedelta(days=1) else day.strftime("%d/%m/%Y"))
        if label != group:
            st.markdown(f'<div class="history-group">{label}</div>', unsafe_allow_html=True)
            group = label
        active = item["id"] == st.session_state.conversation_id
        with st.container(key=f"history_row_{item['id']}"):
            title_col, menu_col = st.columns([5, 1], gap="small", vertical_alignment="center", wrap=False)
            with title_col:
                if (st.button(item["title"], key=f"history_{item['id']}", type="primary" if active else "secondary",
                              icon=":material/keep:" if item["pinned"] else ":material/chat_bubble_outline:",
                              use_container_width=True, help=item["title"], disabled=busy)
                        and open_chat(item["id"])):
                    st.rerun()
            with menu_col, st.popover("⋯", key=f"history_menu_{item['id']}",
                                     help="Kelola percakapan",
                                     width="stretch", disabled=busy, on_change="rerun"), \
                    st.container(key=f"history_actions_{item['id']}"):
                history_actions(item)
        # st.divider()
        # st.caption("Riwayat tersimpan lokal di perangkat yang menjalankan Dapuria.")
    if st.session_state.history_error:
        st.warning(st.session_state.history_error)

if st.session_state.renaming_chat_id:
    rename_item = next((item for item in conversations if item["id"] == st.session_state.renaming_chat_id), None)
    if rename_item is not None and not busy:
        rename_chat_dialog(rename_item)
    else:
        close_rename_dialog()

completed = st.session_state.phase == "await_query" and st.session_state.latest_result is not None
render_progress(st.session_state.phase, completed)
if not st.session_state.messages:
    render_welcome()
    with chat_bubble("assistant", "welcome"):
        st.write("Halo! Aku bantu carikan resep dari bahan yang kamu punya. Mau masak pakai apa hari ini?")
    st.caption("Mulai dari inspirasi kecil, pilih bahan yang ada di dapurmu:")
    examples = ["telur, bawang putih, tomat", "ayam, jeruk nipis, mentega", "nasi, kecap asin, wortel"]
    with st.container(key="example_cards"):
        for col, example in zip(st.columns(3), examples):
            col.button(example, key=f"example_{example}", use_container_width=True,
                       disabled=busy, on_click=queue_message, args=(example,))
    st.caption("Boleh juga bercerita: “aku punya nasi sama telur”.")

# Render conversation history.
last_user_index = next((i for i in range(len(st.session_state.messages) - 1, -1, -1)
                        if st.session_state.messages[i]["role"] == "user"), -1)
for index, message in enumerate(st.session_state.messages):
    bubble_key = "latest" if index in {len(st.session_state.messages) - 1, last_user_index} else str(index)
    with chat_bubble(message["role"], bubble_key):
        st.markdown(message["content"])
        saved_result = message.get("result")
        if (saved_result and saved_result != st.session_state.latest_result
                and st.session_state.phase == "await_query"
                and st.button("Lihat rekomendasi ini", key=f"past_result_{index}", icon=":material/restaurant:", disabled=busy)):
            st.session_state.selected_result = saved_result

if st.session_state.excluded_ingredients:
    st.caption("Bahan yang dihindari: " + ", ".join(st.session_state.excluded_ingredients))
    st.caption(f'Untuk menghapus batasan, tulis "boleh pakai {st.session_state.excluded_ingredients[0]}" atau "hapus batasan".')
if not completed and st.session_state.inventory:
    st.caption("Bahan saat ini: " + ", ".join(st.session_state.inventory))

prompt_text = "Aku punya nasi sama telur..."
if st.session_state.phase == "await_confirmation":  
    prompt_text = "Ada bahan pelengkapnya? Boleh sebut bahan lain juga"
elif st.session_state.inventory:
    prompt_text = "Tambahkan bahan atau batasan, atau ketik cari resep"

if st.session_state.phase == "await_confirmation":
    prompt_text = "Ada bahan pelengkapnya? Boleh sebut bahan lain juga"

if st.session_state.phase == "await_confirmation":
    st.info("Cek satu bahan pelengkap dulu, ya. Kamu juga boleh menyebut bahan lain yang tersedia.", icon="🧄")
    pending_candidates = (st.session_state.pending or {}).get("candidate_staples", [])
    if len(pending_candidates) == 1:
        yes_col, no_col = st.columns(2)
        yes_col.button("Ada, tambahkan", use_container_width=True, type="primary",
                       disabled=busy, on_click=queue_message, args=("iya",))
        no_col.button("Tidak ada, lanjut saja", use_container_width=True,
                      disabled=busy, on_click=queue_message, args=("tidak ada",))

if not completed:
    st.chat_input(prompt_text, key="chat_composer", disabled=busy,
                  submit_mode="disable", on_submit=queue_message)

submission = st.session_state.submission
if submission:
    with processing_guard():
        st.session_state.selected_result = None
        user_message = submission["text"]
        st.session_state.messages.append({"role": "user", "content": user_message})
        preferences = update_preferences(st.session_state.excluded_ingredients, user_message)
        st.session_state.excluded_ingredients = preferences.excluded
        is_confirmation = st.session_state.phase == "await_confirmation" and not preferences.has_directive
        if not is_confirmation:
            if st.session_state.pending and preferences.has_directive:
                candidates = st.session_state.pending.get("candidate_staples", [])
                if preferences.inventory_text.strip(" ,.!?").lower() in {"ada", "iya", "ya", "punya"}:
                    st.session_state.inventory = list(dict.fromkeys(st.session_state.inventory + [
                        item for item in candidates if not ingredient_is_excluded(item, preferences.excluded)
                    ]))
            inventory_text = preferences.inventory_text
            if preferences.has_directive and inventory_text.strip(" ,.!?").lower() in {"iya", "ya", "oke", "ok"}:
                inventory_text = ""
            st.session_state.inventory = update_inventory(st.session_state.inventory, inventory_text)
            st.session_state.phase = "await_query"
            st.session_state.pending = None
            st.session_state.latest_result = None
            if not any(not ingredient_is_excluded(item, preferences.excluded) for item in st.session_state.inventory):
                content = ("Batasan bahanmu sudah dicatat. Bahan apa yang kamu punya untuk dimasak?"
                           if preferences.excluded else "Belum ada bahan yang bisa dicari. Sebutkan bahan yang kamu punya, ya.")
                st.session_state.messages.append({"role": "assistant", "content": content})
                finish_turn()
        with chat_bubble("user", "incoming"):
            st.write(user_message)
        persist_chat()

        try:
            with activity("Menyiapkan dapur dan pencarian resep…"):
                settings, service, llama = load_services()
        except Exception:
            LOGGER.exception("Could not initialize recipe services")
            st.session_state.messages.append({
                "role": "assistant", "content": "Dapur pencarian belum siap. Pastikan dependensi dan model tersedia, lalu coba kirim bahanmu lagi. Riwayatmu tetap bisa dibuka."
            })
            finish_turn()

        if not is_confirmation:
            # Avoid showing the previous query's cards during a new clarification.
            st.session_state.latest_result = None
            with activity("Mencocokkan bahanmu dengan resep…"):
                result = search_recipes(service, ", ".join(st.session_state.inventory))

            if result["status"] != "ok":
                assistant_text = result["message"]
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                st.session_state.latest_result = None
            else:
                candidates = service.clarification_candidates(result)
                if candidates:
                    with activity("Mengecek satu bahan pelengkap…"):
                        question = llama.clarification_question(
                            result["user_ingredients"], candidates
                        )
                    st.session_state.pending = {
                        "original_query": ", ".join(st.session_state.inventory),
                        "base_inventory": list(st.session_state.inventory),
                        "excluded_ingredients": list(st.session_state.excluded_ingredients),
                        "candidate_staples": candidates,
                        "initial_result": result,
                        "additional_ingredients": [],
                    }
                    st.session_state.phase = "await_confirmation"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": question}
                    )
                else:
                    payload = service.build_grounded_payload(result)
                    with activity("Merangkai pilihan resep untukmu…"):
                        assistant_text = llama.grounded_summary(payload)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_text}
                    )
                    st.session_state.latest_result = result
                    st.session_state.phase = "await_query"
        else:
            pending = st.session_state.pending or {}
            candidates = pending.get("candidate_staples", [])
            reply = llama.interpret_clarification_reply(user_message, candidates)
            confirmation = reply.confirmation
            previous_extras = pending.get("additional_ingredients", [])
            updated_extras = update_inventory(previous_extras, user_message)
            additional = list(dict.fromkeys(
                [item for item in previous_extras if item in updated_extras]
                + reply.additional_ingredients
            ))
            initial_result = pending.get("initial_result")
            original_ingredients = pending.get("base_inventory", (initial_result or {}).get(
                "user_ingredients", parse_user_ingredients(pending.get("original_query", ""))
            ))
            previous = pending.get("retained_ingredients", original_ingredients)
            updated = update_inventory(previous, user_message)
            retained = [item for item in previous if item in updated]
            pending["retained_ingredients"] = retained
            st.session_state.inventory = list(dict.fromkeys(retained + additional))
            pending["additional_ingredients"] = additional
            if confirmation.status == "unclear":
                st.session_state.pending = pending
                assistant_text = (
                    f"Aku belum bisa memastikan dari jawabanmu. Ada **{', '.join(candidates)}** juga? "
                    "Kalau adanya jenis tertentu, sebutkan saja, ya."
                )
                if additional:
                    assistant_text = (
                        f"Aku catat bahan tambahanmu: **{', '.join(additional)}**. "
                        + assistant_text
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                finish_turn()
            confirmed = confirmation.ingredients
            added = [item for item in dict.fromkeys(confirmed + additional)
                     if item not in retained]
            st.session_state.inventory = list(dict.fromkeys(retained + added))
            combined_query = ", ".join(st.session_state.inventory)

            # Reuse the exact initial result when no ingredient was added.
            if st.session_state.inventory == original_ingredients and initial_result is not None:
                result = initial_result
            else:
                with activity("Mencocokkan bahan tambahan dengan resep…"):
                    result = search_recipes(service, combined_query)
            if result["status"] == "error":
                st.session_state.pending = pending
                st.session_state.messages.append({"role": "assistant", "content": result["message"]})
                finish_turn()

            if result["status"] == "ok":
                payload = service.build_grounded_payload(result)
                with activity("Merangkai pilihan resep untukmu…"):
                    assistant_text = llama.grounded_summary(payload)
                if added:
                    assistant_text = (
                        f"Siap, aku tambahkan **{', '.join(added)}** juga, ya.\n\n"
                        + assistant_text
                    )
                elif retained != original_ingredients:
                    assistant_text = "Siap, aku perbarui daftar bahanmu, ya.\n\n" + assistant_text
                else:
                    assistant_text = "Oke, kita lanjut dengan bahan awal, ya.\n\n" + assistant_text
                st.session_state.latest_result = result
            else:
                assistant_text = result["message"]
                st.session_state.latest_result = None

            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_text}
            )
            st.session_state.phase = "await_query"
            st.session_state.pending = None

        finish_turn()

# Cards and dialogs use the original ranked data, never LLM-generated details.
result = st.session_state.selected_result or (
    st.session_state.latest_result if st.session_state.phase == "await_query" else None
)
if st.session_state.selected_result:
    st.info("Kamu sedang melihat rekomendasi sebelumnya dalam percakapan ini.", icon="🕰️")
    if st.button("Kembali ke hasil terbaru", key="show_latest"):
        st.session_state.selected_result = None
        st.rerun()
if result and result.get("status") == "ok":
    render_results(result)

if completed:
    with st.container(key="result_actions", border=True):
        st.write("Pilihan resepmu sudah siap. Mau lanjut dengan cara apa?")
        continue_col, new_col = st.columns(2)
        continue_col.button("Lanjutkan percakapan", key="continue_chat", icon=":material/chat:",
                            use_container_width=True, disabled=busy, on_click=continue_chat)
        if new_col.button("Cari resep baru", key="fresh_search", icon=":material/add:",
                          use_container_width=True, disabled=busy):
            persist_chat()
            reset_chat()
            st.rerun()
