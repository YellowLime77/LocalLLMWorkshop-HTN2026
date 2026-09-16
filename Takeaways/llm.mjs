// llm.mjs: call a cloud LLM, and fall back to local Ollama when the cloud stops working
// npm install openai. Use on the server only (Node, API routes), not in the browser
import OpenAI from "openai";

export const config = {
  cooldownMs: 60_000,
  fallbackStatusCodes: new Set([401, 402, 403, 404, 408, 429]), // plus every 5xx
  localOptions: { reasoning_effort: "none" }, // Gemma 4: skip thinking
};

let clients = null;
let cloudRetryAt = 0;

function getClients() {
  if (!clients) {
    const env = process.env;
    clients = {
      cloud: env.CLOUD_API_KEY
        ? new OpenAI({ baseURL: env.CLOUD_BASE_URL || "https://api.openai.com/v1", apiKey: env.CLOUD_API_KEY, timeout: 30_000, maxRetries: 0 })
        : null,
      local: new OpenAI({ baseURL: env.LOCAL_BASE_URL || "http://localhost:11434/v1", apiKey: "ollama" }),
    };
  }
  return clients;
}

function cloudIsDown(err) {
  if (err instanceof OpenAI.APIConnectionError) return true; // offline, DNS failure, or timeout
  return err instanceof OpenAI.APIError && typeof err.status === "number" &&
    (config.fallbackStatusCodes.has(err.status) || err.status >= 500);
}

export async function complete(messages, options = {}) {
  const { cloud, local } = getClients();

  if (cloud && Date.now() >= cloudRetryAt) {
    const model = process.env.CLOUD_MODEL || "gpt-5-mini";
    try {
      const response = await cloud.chat.completions.create({ ...options, model, messages });
      return { text: response.choices[0].message.content ?? "", provider: "cloud", model, response };
    } catch (err) {
      if (!cloudIsDown(err)) throw err;
      console.warn(`[llm] Cloud failed (${err.status ?? err.name}). Using local Ollama for ${config.cooldownMs / 1000}s.`);
      cloudRetryAt = Date.now() + config.cooldownMs;
    }
  }

  const model = process.env.LOCAL_MODEL || "gemma4:e2b";
  try {
    const response = await local.chat.completions.create({ ...config.localOptions, ...options, model, messages });
    return { text: response.choices[0].message.content ?? "", provider: "local", model, response };
  } catch (err) {
    if (err instanceof OpenAI.APIConnectionError) {
      throw new Error("Cloud is unavailable and local Ollama isn't reachable. Is Ollama running?", { cause: err });
    }
    throw err;
  }
}

export async function chat(messages, options = {}) {
  return (await complete(messages, options)).text;
}