"""Real Chromium/IndexedDB smoke test; run with a Python that has Playwright."""

from __future__ import annotations

import base64
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DATABASE = "dapuria-chat-history-v1"
READ_RECORDS = """async () => {
  const db = await new Promise((resolve, reject) => {
    const request = indexedDB.open('dapuria-chat-history-v1', 1);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  try {
    return await new Promise((resolve, reject) => {
      const request = db.transaction('conversations').objectStore('conversations').getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  } finally { db.close(); }
}"""


class Server:
    def __init__(self, directory: Path):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.wrapper = directory / "app.py"
        self.wrapper.write_text("""
import runpy
from copy import deepcopy
import dapuria.recipe_service as recipes
from dapuria.llama_service import LlamaService

class Retriever:
    def __init__(self, settings):
        pass
    def retrieve(self, query, ingredients):
        return deepcopy([
            {"id": "one", "title": "Jamur Kukus", "ingredient_names": ["jamur", "tomat", "bawang putih"],
             "steps_text": "Kukus hingga matang.", "sbert_score": 0.9},
            {"id": "two", "title": "Tahu Saus Tomat", "ingredient_names": ["tahu", "jamur", "tomat", "bawang putih"],
             "steps_text": "Masak hingga matang.", "sbert_score": 0.8},
        ])

recipes.HybridRetriever = Retriever
LlamaService.available = lambda self: False
runpy.run_path(APP_PATH, run_name="__main__")
""".replace("APP_PATH", repr(str(ROOT / "app.py"))), encoding="utf-8")
        self.logs = directory / "server.log"
        self.process = None
        self.log_handle = None

    def start(self):
        interpreter = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
        # Windows' venv redirector creates another process. Start its base Python
        # with the same dependencies so this test can stop its own process handle.
        if os.name == "nt":
            interpreter = Path(sys._base_executable)
        package_path = ROOT / (".venv/Lib/site-packages" if os.name == "nt" else "src")
        environment = dict(os.environ, DAPURIA_HISTORY_BACKEND="browser", PYTHONDONTWRITEBYTECODE="1",
                           PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(package_path)]))
        environment.pop("DAPURIA_HISTORY_PATH", None)
        self.log_handle = self.logs.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(interpreter), "-B", "-m", "streamlit", "run", str(self.wrapper),
             "--server.port", str(self.port), "--server.address", "127.0.0.1", "--server.headless", "true",
             "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none"],
            cwd=ROOT, env=environment, stdout=self.log_handle, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(self.logs.read_text(encoding="utf-8"))
            try:
                with urlopen(self.url + "/_stcore/health", timeout=1) as response:
                    if response.status == 200:
                        return
            except (OSError, URLError):
                time.sleep(0.2)
        raise TimeoutError("Test server did not start")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=15)
        if self.log_handle is not None:
            self.log_handle.close()


def ready(page, url):
    page.goto(url, wait_until="domcontentloaded")
    expect(page.get_by_text("Riwayat tersimpan di browser ini.", exact=True)).to_be_visible(timeout=30000)
    expect(page.get_by_test_id("stException")).to_have_count(0)
    expect(page.get_by_role("button", name="Ekspor riwayat", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Impor riwayat", exact=True)).to_have_count(0)


def send(page, text):
    composer = page.get_by_test_id("stChatInputTextArea")
    expect(composer).to_be_enabled(timeout=30000)
    composer.fill(text)
    composer.press("Enter")


def wait_records(page, predicate):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        records = page.evaluate(READ_RECORDS)
        if predicate(records):
            return records
        page.wait_for_timeout(100)
    raise TimeoutError("IndexedDB did not reach the expected state")


def active_count(page, count):
    return wait_records(page, lambda records: len([record for record in records if not record.get("deleted")]) == count)


def native_transactions(page, record):
    javascript = (ROOT / "src/dapuria/assets/history_storage.js").read_bytes()
    module_url = "data:text/javascript;base64," + base64.b64encode(javascript).decode()
    page.evaluate("async url => { window.historyTestModule = await import(url); }", module_url)

    def execute(request):
        return page.evaluate("""request => new Promise(resolve => {
          let cleanup;
          cleanup = window.historyTestModule.default({
            data: {client_id: 'native-test', request},
            setStateValue: (_name, response) => { resolve(response); setTimeout(() => cleanup(), 0); },
            setTriggerValue: () => {},
          });
        })""", request)

    record = {**record, "id": "native-chat", "revision": 0}
    operations = [{"id": "op-1", "kind": "put", "conversation_id": "native-chat", "expected_revision": 0, "record": record},
                  {"id": "op-2", "kind": "put", "conversation_id": "native-chat", "expected_revision": 1,
                   "record": {**record, "title": "Native updated"}}]
    request = {"id": "native-batch", "kind": "apply", "operations": operations}
    first = execute(request)
    assert first["records"][0]["revision"] == 2
    assert execute(request)["records"] == first["records"]
    conflict = execute({**request, "id": "conflicting-batch"})
    assert conflict["error"] == "conflict"
    assert conflict["records"] == first["records"]
    page.evaluate("""() => {
      window.originalPut = IDBObjectStore.prototype.put;
      let puts = 0;
      IDBObjectStore.prototype.put = function(...args) {
        if (this.name === 'conversations' && ++puts === 2) throw new DOMException('Full', 'QuotaExceededError');
        return window.originalPut.apply(this, args);
      };
    }""")
    failed = execute({"id": "atomic-batch", "kind": "apply", "operations": [
        {**operations[0], "expected_revision": 2},
        {**operations[0], "conversation_id": "other-native", "record": {**record, "id": "other-native"}},
    ]})
    assert failed["error"] == "quota"
    page.evaluate("() => { IDBObjectStore.prototype.put = window.originalPut; }")
    assert execute({"id": "read-after-failure", "kind": "read"})["records"] == first["records"]


def main():
    with tempfile.TemporaryDirectory(prefix="dapuria-browser-test-") as temporary:
        directory = Path(temporary).resolve()
        assert directory.is_relative_to(Path(tempfile.gettempdir()).resolve())
        assert directory.name.startswith("dapuria-browser-test-")
        server = Server(directory)
        try:
            server.start()
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, executable_path=playwright.chromium.executable_path)
                context = browser.new_context(viewport={"width": 1440, "height": 1000})
                page = context.new_page()
                ready(page, server.url)
                send(page, "jamur, tomat")
                expect(page.get_by_text("Cek satu bahan pelengkap dulu, ya.", exact=False)).to_be_visible(timeout=30000)
                send(page, "ada dongg")
                expect(page.get_by_role("heading", name="Yuk, masak yang ini.")).to_be_visible(timeout=30000)
                active_count(page, 1)
                page.get_by_role("button", name="⋯", exact=True).first.click()
                page.get_by_role("button", name="Ubah nama", exact=True).click()
                page.get_by_role("textbox", name="Judul percakapan").fill("Percakapan jamur")
                page.get_by_role("button", name="Simpan judul", exact=True).click()
                expect(page.get_by_role("button", name="Percakapan jamur", exact=True)).to_be_enabled()
                page.get_by_role("button", name="⋯", exact=True).first.click()
                page.get_by_role("button", name="Pin chat", exact=True).click()
                records = wait_records(page, lambda records: any(record.get("pinned") and record["title"] == "Percakapan jamur" for record in records))
                saved = next(record for record in records if record.get("pinned"))
                assert saved["title"] == "Percakapan jamur" and saved["pinned"], saved
                assert saved["snapshot"]["latest_result"]["results"][0]["steps_text"] == "Kukus hingga matang."
                print("PASS: IndexedDB save, confirmation, rename, and pin", flush=True)

                page.get_by_role("button", name="Mulai percakapan baru", exact=True).click()
                send(page, "jamur, tomat")
                expect(page.get_by_text("Cek satu bahan pelengkap dulu, ya.", exact=False)).to_be_visible(timeout=30000)
                active_count(page, 2)
                address = page.url
                all_records = page.evaluate(READ_RECORDS)
                pending = next(record for record in all_records if record["snapshot"]["phase"] == "await_confirmation")
                assert pending["snapshot"]["pending"]["candidate_staples"] == ["bawang putih"]
                page.close()
                server.stop()
                server.start()
                page = context.new_page()
                ready(page, address)
                expect(page.get_by_text("Cek satu bahan pelengkap dulu, ya.", exact=False)).to_be_visible()
                send(page, "ada dongg")
                expect(page.get_by_role("heading", name="Yuk, masak yang ini.")).to_be_visible(timeout=30000)
                print("PASS: history and pending confirmation survive a fresh server process", flush=True)

                tab = context.new_page()
                ready(tab, page.url)
                expect(tab.get_by_role("heading", name="Yuk, masak yang ini.")).to_be_visible(timeout=30000)
                page.get_by_role("button", name="Lanjutkan percakapan", exact=True).click()
                send(page, "aku punya bawang bombay")
                expect(tab.get_by_text("Bahan yang terbaca: jamur, tomat, bawang putih, bawang bombay", exact=True)).to_be_visible(timeout=30000)
                tab.get_by_role("button", name="Lanjutkan percakapan", exact=True).click()
                send(tab, "aku punya micin")
                expect(page.get_by_text("Bahan yang terbaca: jamur, tomat, bawang putih, bawang bombay, msg", exact=True)).to_be_visible(timeout=30000)
                tab.close()
                print("PASS: two tabs refresh the active conversation before continuing it", flush=True)

                second = browser.new_context(viewport={"width": 1440, "height": 1000})
                other = second.new_page()
                ready(other, server.url)
                assert not other.evaluate(READ_RECORDS)
                send(other, "jamur, tomat")
                expect(other.get_by_text("Cek satu bahan pelengkap dulu, ya.", exact=False)).to_be_visible(timeout=30000)
                own_records = active_count(other, 1)
                own_id = next(record["id"] for record in own_records if not record.get("deleted"))
                row = other.locator(f".st-key-history_row_{own_id}")
                row.get_by_role("button", name="⋯", exact=True).click()
                other.get_by_role("button", name="Hapus chat", exact=True).click()
                active_count(other, 0)
                other.reload(wait_until="domcontentloaded")
                expect(other.get_by_text("Riwayat tersimpan di browser ini.", exact=True)).to_be_visible()
                expect(other.locator(f".st-key-history_row_{own_id}")).to_have_count(0)
                active_count(page, 2)
                print("PASS: browser isolation and persistent deletion", flush=True)

                other.evaluate("""() => {
                  window.failHistoryWrites = true;
                  const put = IDBObjectStore.prototype.put;
                  IDBObjectStore.prototype.put = function(...args) {
                    if (window.failHistoryWrites && this.name === 'conversations') throw new DOMException('Full', 'QuotaExceededError');
                    return put.apply(this, args);
                  };
                }""")
                other.get_by_role("button", name="Mulai percakapan baru", exact=True).click()
                send(other, "jamur, tomat")
                expect(other.get_by_text("Riwayat belum tersimpan di browser.", exact=False)).to_be_visible(timeout=30000)
                active_count(other, 0)
                expect(other.get_by_text("Bahan saat ini: jamur, tomat", exact=True)).to_be_visible()
                other.evaluate("() => { window.failHistoryWrites = false; }")
                other.get_by_role("button", name="Coba simpan lagi", exact=True).click()
                active_count(other, 1)
                other.reload(wait_until="domcontentloaded")
                expect(other.get_by_text("Riwayat tersimpan di browser ini.", exact=True)).to_be_visible(timeout=30000)
                expect(other.get_by_text("Cek satu bahan pelengkap dulu, ya.", exact=False)).to_be_visible()
                print("PASS: failed writes preserve the working chat and can be retried", flush=True)

                native = browser.new_context()
                native_page = native.new_page()
                ready(native_page, server.url)
                native_transactions(native_page, saved)
                print("PASS: native IndexedDB transactions are atomic, retry-safe, and reject conflicting writes", flush=True)
                browser.close()
        except BaseException:
            print(server.logs.read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
            raise
        finally:
            server.stop()


if __name__ == "__main__":
    main()
