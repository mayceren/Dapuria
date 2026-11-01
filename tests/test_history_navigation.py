"""Saved recommendations remain accessible while a conversation continues."""

from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dapuria.chat_history import STATE_KEYS, ChatHistoryStore


def recommendation(title):
    return {
        "status": "ok",
        "user_ingredients": ["ayam", "bawang putih"],
        "results": [{
            "id": title,
            "title": title,
            "ingredient_names": ["ayam", "bawang putih", "air"],
            "matched_recipe_items": ["ayam", "bawang putih"],
            "missing_items": ["air"],
            "steps_text": "Masak hingga matang.",
        }, {
            "id": f"{title}-alternative",
            "title": f"Alternatif {title}",
            "ingredient_names": ["ayam", "bawang putih"],
            "matched_recipe_items": ["ayam", "bawang putih"],
            "missing_items": [],
            "steps_text": "Kukus hingga matang.",
        }],
    }


def open_history(monkeypatch, tmp_path, phase, latest):
    previous = recommendation("Ayam Kecap Bombay")
    newer = recommendation("Ayam Kukus Jahe")
    inventory = ["ayam", "bawang putih", "air"]
    state = {
        "messages": [
            {"role": "user", "content": "Aku punya ayam dan bawang putih."},
            {"role": "assistant", "content": "Ini rekomendasimu.", "result": previous},
            {"role": "user", "content": "Aku ada air juga."},
            {"role": "assistant", "content": "Ada bawang merah juga?"},
        ],
        "phase": phase,
        "pending": {
            "original_query": ", ".join(inventory),
            "base_inventory": inventory,
            "candidate_staples": ["bawang merah"],
            "initial_result": newer,
            "additional_ingredients": [],
        } if phase == "await_confirmation" else None,
        "latest_result": {"none": None, "same": previous, "new": newer}[latest],
        "inventory": inventory,
        "excluded_ingredients": ["cabai"],
    }
    if phase == "await_query" and latest == "new":
        state["messages"][-1] = {
            "role": "assistant", "content": "Ini rekomendasi terbarumu.", "result": newer,
        }
    store = ChatHistoryStore(tmp_path / "history.sqlite3")
    store.save("local", "history-navigation", state)
    monkeypatch.setenv("DAPURIA_HISTORY_PATH", str(store.path))
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"))
    app.query_params["chat"] = "history-navigation"
    app.run()
    assert not app.exception
    return app, store, previous, newer


def snapshot(app):
    return deepcopy({key: app.session_state[key] for key in STATE_KEYS})


@pytest.mark.parametrize(("phase", "latest"), [
    ("await_confirmation", "none"),
    ("await_confirmation", "same"),
    ("await_query", "none"),
    ("await_query", "new"),
])
def test_opening_saved_recommendations_preserves_the_active_conversation(
    monkeypatch, tmp_path, phase, latest,
):
    app, store, previous, newer = open_history(monkeypatch, tmp_path, phase, latest)
    before = snapshot(app)
    persisted = store.load("local", "history-navigation")

    app.button(key="past_result_1").click().run()

    assert not app.exception
    assert snapshot(app) == before
    assert store.load("local", "history-navigation") == persisted
    assert app.session_state.submission is None
    cards = [element.value for element in app.markdown if '<article class="recipe-full">' in element.value]
    assert len(cards) == 1
    assert previous["results"][0]["title"] in cards[0]
    assert newer["results"][0]["title"] not in cards[0]
    # Put the opened recipes next to the clicked message, before the follow-up.
    markdown = [element.value for element in app.markdown]
    assert markdown.index(cards[0]) < markdown.index(before["messages"][2]["content"])
    app.button(key="detail_recipe_2").click().run()
    assert not app.exception
    assert snapshot(app) == before
    assert any("Alternatif Ayam Kecap Bombay" in element.value for element in app.markdown)
    app.button(key="close_recipe").click().run()
    assert not app.exception

    back = app.button(key="show_latest")
    assert back.label == ("Kembali ke hasil terbaru" if latest == "new" else "Kembali ke percakapan")
    back.click().run()

    assert not app.exception
    assert snapshot(app) == before
    assert store.load("local", "history-navigation") == persisted
    cards = [element.value for element in app.markdown if '<article class="recipe-full">' in element.value]
    if latest == "new":
        assert len(cards) == 1
        assert newer["results"][0]["title"] in cards[0]
    else:
        assert not cards
        assert len(app.chat_input) == 1
    if phase == "await_confirmation":
        assert any(button.label == "Ada, tambahkan" for button in app.button)


def test_identical_saved_results_can_be_opened_without_duplicate_widgets(monkeypatch, tmp_path):
    app, store, previous, _ = open_history(monkeypatch, tmp_path, "await_query", "new")
    state = snapshot(app)
    state["messages"].insert(2, {"role": "assistant", "content": "Salinan hasil.", "result": previous})
    store.save("local", "history-navigation", state)
    app.session_state.messages = deepcopy(state["messages"])
    app.run()

    for index in (2, 1, 2):
        app.button(key=f"past_result_{index}").click().run()
        assert not app.exception
        cards = [element.value for element in app.markdown if '<article class="recipe-full">' in element.value]
        assert len(cards) == 1
        assert previous["results"][0]["title"] in cards[0]
        assert len([button for button in app.button if button.key == "detail_recipe_2"]) == 1
        assert snapshot(app) == state
