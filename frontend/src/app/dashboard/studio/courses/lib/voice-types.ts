/**
 * Voice Types for Course Generation
 */

export interface Voice {
  id: string;
  name: string;
  provider: 'elevenlabs' | 'openai';
  gender: 'male' | 'female' | 'neutral';
  language: string;
  style: string;
  description: string;
}

export interface VoicesResponse {
  voices: Voice[];
  supported_languages: string[];
  current_language: string;
  current_provider: string;
}

export interface LanguageInfo {
  code: string;
  name: string;
  flag: string;
}

export const SUPPORTED_LANGUAGES: LanguageInfo[] = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'fr', name: 'Français', flag: '🇫🇷' },
  { code: 'fr-CA', name: 'Français (Québec)', flag: '🇨🇦' },
  { code: 'fr-AF', name: 'Français (Afrique)', flag: '🌍' },
  { code: 'es', name: 'Español', flag: '🇪🇸' },
  { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
  { code: 'pt', name: 'Português', flag: '🇵🇹' },
  { code: 'it', name: 'Italiano', flag: '🇮🇹' },
  { code: 'nl', name: 'Nederlands', flag: '🇳🇱' },
  { code: 'pl', name: 'Polski', flag: '🇵🇱' },
  { code: 'ru', name: 'Русский', flag: '🇷🇺' },
  { code: 'zh', name: '中文', flag: '🇨🇳' },
  { code: 'sw', name: 'Kiswahili', flag: '🇰🇪' },
  { code: 'ha', name: 'Hausa', flag: '🇳🇬' },
  { code: 'yo', name: 'Yorùbá', flag: '🇳🇬' },
];

export function getLanguageName(code: string): string {
  const lang = SUPPORTED_LANGUAGES.find(l => l.code === code);
  return lang ? lang.name : code;
}

export function getLanguageFlag(code: string): string {
  const lang = SUPPORTED_LANGUAGES.find(l => l.code === code);
  return lang ? lang.flag : '🌐';
}

export function getGenderLabel(gender: string): string {
  switch (gender) {
    case 'male': return 'Homme';
    case 'female': return 'Femme';
    case 'neutral': return 'Neutre';
    default: return gender;
  }
}
