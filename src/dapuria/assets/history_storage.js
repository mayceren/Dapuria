// IndexedDB is authoritative. Server sessions only hold a temporary working copy.
const DATABASE = "dapuria-chat-history-v1";
const running = new Map();

function requestValue(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function completed(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error || new Error("Transaction aborted"));
  });
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    let expired = false;
    const timer = setTimeout(() => {
      expired = true;
      reject(new Error("Browser storage did not respond"));
    }, 8000);
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("conversations", { keyPath: "id" });
      request.result.createObjectStore("receipts", { keyPath: "id" });
    };
    request.onsuccess = () => {
      clearTimeout(timer);
      if (expired) return request.result.close();
      request.result.onversionchange = () => request.result.close();
      resolve(request.result);
    };
    request.onerror = () => {
      clearTimeout(timer);
      reject(request.error);
    };
  });
}

async function readRecords(database) {
  const transaction = database.transaction("conversations", "readonly");
  const done = completed(transaction);
  const records = await requestValue(transaction.objectStore("conversations").getAll());
  await done;
  return records;
}

async function execute(request, clientId) {
  let database;
  let writeTransaction;
  let writeDone;
  try {
    database = await openDatabase();
    if (request.kind === "apply") {
      const transaction = database.transaction(["conversations", "receipts"], "readwrite");
      const done = completed(transaction);
      writeTransaction = transaction;
      writeDone = done;
      // An explicit abort below is handled here, rather than becoming unhandled.
      done.catch(() => {});
      const conversations = transaction.objectStore("conversations");
      const receipts = transaction.objectStore("receipts");
      const receipt = await requestValue(receipts.get(request.id));
      if (!receipt) {
        const records = new Map((await requestValue(conversations.getAll())).map(record => [record.id, record]));
        for (const operation of request.operations) {
          const current = records.get(operation.conversation_id);
          if ((current?.revision || 0) !== operation.expected_revision) {
            transaction.abort();
            await done.catch(() => {});
            return { id: request.id, error: "conflict", records: await readRecords(database) };
          }
          const record = operation.kind === "delete"
            ? { id: operation.conversation_id, deleted: true }
            : structuredClone(operation.record);
          record.revision = operation.expected_revision + 1;
          records.set(record.id, record);
          await requestValue(conversations.put(record));
        }
        await requestValue(receipts.put({ id: request.id, created_at: Date.now() }));
        const retained = (await requestValue(receipts.getAll())).sort((a, b) => b.created_at - a.created_at);
        for (const old of retained.slice(128)) receipts.delete(old.id);
      }
      await done;
      if (typeof BroadcastChannel !== "undefined") {
        const channel = new BroadcastChannel(DATABASE);
        channel.postMessage({ clientId });
        channel.close();
      }
    }
    return { id: request.id, records: await readRecords(database) };
  } catch (error) {
    if (writeTransaction) {
      try { writeTransaction.abort(); } catch { /* It may already be closed. */ }
      await writeDone.catch(() => {});
    }
    return { id: request.id, error: error?.name === "QuotaExceededError" ? "quota" : "storage" };
  } finally {
    database?.close();
  }
}

export default function(component) {
  const { data, setStateValue, setTriggerValue } = component;
  let disposed = false;
  let channel;
  if (typeof BroadcastChannel !== "undefined") {
    channel = new BroadcastChannel(DATABASE);
    channel.onmessage = event => {
      if (event.data?.clientId !== data.client_id) {
        setTriggerValue("changed", crypto.randomUUID());
      }
    };
  }
  const request = data.request;
  if (request) {
    if (!running.has(request.id)) {
      running.set(request.id, execute(request, data.client_id));
      if (running.size > 32) running.delete(running.keys().next().value);
    }
    running.get(request.id).then(response => {
      if (!disposed) setStateValue("response", response);
    });
  }
  return () => {
    disposed = true;
    channel?.close();
  };
}
