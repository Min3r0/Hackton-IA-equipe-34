// script.js
const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// Historique de la conversation (envoyé à chaque requête pour garder le contexte)
let history = [];

function addMessage(role, content, { error = false, thinking = false, id = null } = {}) {
  const el = document.createElement("div");
  el.className = `message ${role}${error ? " error" : ""}${thinking ? " thinking" : ""}`;
  if (id) el.id = id;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  el.appendChild(bubble);
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    if (data.connected) {
      statusDot.className = "dot connected";
      statusText.textContent = data.modelAvailable
        ? `Connecté (${data.configuredModel})`
        : `Connecté — modèle "${data.configuredModel}" introuvable sur Ollama`;
    } else {
      statusDot.className = "dot disconnected";
      statusText.textContent = "Déconnecté du serveur Ollama";
    }
  } catch {
    statusDot.className = "dot disconnected";
    statusText.textContent = "Déconnecté (serveur devweb injoignable)";
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  addMessage("user", text);
  history.push({ role: "user", content: text });
  messageInput.value = "";
  sendBtn.disabled = true;

  const thinkingId = "thinking-" + Date.now();
  addMessage("assistant", "L'assistant réfléchit...", { thinking: true, id: thinkingId });

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });

    const data = await res.json();
    removeMessage(thinkingId);

    if (!res.ok) {
      addMessage("assistant", data.error || "Erreur inconnue", { error: true });
    } else {
      addMessage("assistant", data.content);
      history.push({ role: "assistant", content: data.content });
    }
  } catch (err) {
    removeMessage(thinkingId);
    addMessage("assistant", "Impossible de contacter le serveur devweb.", { error: true });
  } finally {
    sendBtn.disabled = false;
    messageInput.focus();
  }
});

checkStatus();
setInterval(checkStatus, 5000);
