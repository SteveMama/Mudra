import { DEMO_PHRASES } from "./demoPhrases.js";

const phraseLookup = new Map(DEMO_PHRASES.map((phrase) => [phrase.english, phrase]));

export class TranslationService {
  async translate(text, language) {
    const phrase = phraseLookup.get(text);
    if (phrase?.translations[language]) {
      return phrase.translations[language];
    }

    if (language === "en") {
      return text;
    }

    return text;
  }
}
