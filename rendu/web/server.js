// server.js
// Interface DEV WEB - Challenge IA TechCorp
// Sert le front (public/) et fait le proxy vers le serveur Ollama de l'équipe INFRA.

const express = require("express");
const path = require("path");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// URL du serveur Ollama monté par l'équipe INFRA.
// Modifiable via variable d'env si jamais ils exposent une autre IP/port.
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

// Nom du modèle créé par l'équipe INFRA avec `ollama create <nom> -f Modelfile`.
// A adapter si le nom choisi par l'INFRA est différent.
const MODEL_NAME = process.env.MODEL_NAME || "phi3.5-financial";

const PORT = process.env.PORT || 3000;

// --- Statut de connexion au serveur Ollama ---
app.get("/api/status", async (req, res) => {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const models = (data.models || []).map((m) => m.name);
    res.json({
      connected: true,
      ollamaUrl: OLLAMA_URL,
      configuredModel: MODEL_NAME,
      modelAvailable: models.includes(MODEL_NAME),
      availableModels: models,
    });
  } catch (err) {
    res.json({
      connected: false,
      ollamaUrl: OLLAMA_URL,
      configuredModel: MODEL_NAME,
      error: err.message,
    });
  }
});

// --- Envoi d'un message au modèle (proxy vers Ollama /api/chat) ---
app.post("/api/chat", async (req, res) => {
  const { messages } = req.body;

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "messages manquant ou vide" });
  }

  try {
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL_NAME,
        messages,
        stream: false,
      }),
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Ollama a répondu ${response.status}: ${text}`);
    }

    const data = await response.json();
    res.json({
      role: "assistant",
      content: data.message?.content || "(réponse vide)",
    });
  } catch (err) {
    res.status(502).json({ error: `Impossible de contacter Ollama: ${err.message}` });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Interface DEV WEB lancée sur http://localhost:${PORT}`);
  console.log(`   -> Proxy vers Ollama: ${OLLAMA_URL} (modèle: ${MODEL_NAME})`);
});
