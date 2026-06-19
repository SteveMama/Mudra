export class TranslationService {
  async translate(text, lang) {
    if (!text || lang === "en") return text;
    try {
      const resp = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error ?? "Translation failed");
      return data.translated;
    } catch (err) {
      console.warn("[mudra] translation error:", err.message);
      return text;
    }
  }
}
