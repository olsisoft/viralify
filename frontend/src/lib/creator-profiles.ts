// Creator Profile Types and Management

export interface CreatorProfile {
  id: string;
  name: string;
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;

  // Brand Identity
  brandName: string;
  tagline: string;
  uniqueValue: string;

  // Content Style
  niche: string;
  subNiches: string[];
  tone: ContentTone;
  hookStyle: HookStyle;
  ctaStyle: string;

  // Target Audience
  audienceDescription: string;
  audienceAgeRange: AgeRange;
  audienceLevel: AudienceLevel;
  audiencePainPoints: string[];
  audienceGoals: string[];
  languageLevel: LanguageLevel;

  // Content Goals
  primaryGoal: ContentGoal;
  secondaryGoals: ContentGoal[];

  // Inspiration
  competitors: string[];
  styleNotes: string;
}

export type ContentTone =
  | 'professional'
  | 'casual'
  | 'motivational'
  | 'educational'
  | 'provocative'
  | 'storytelling'
  | 'humorous'
  | 'authoritative';

export type HookStyle =
  | 'question'
  | 'controversial'
  | 'statistic'
  | 'story'
  | 'pattern-interrupt'
  | 'curiosity-gap'
  | 'bold-claim';

export type AgeRange = '18-24' | '25-34' | '35-44' | '45-54' | '55+' | 'all';

export type AudienceLevel = 'beginner' | 'intermediate' | 'advanced' | 'mixed';

export type LanguageLevel = 'simple' | 'moderate' | 'technical' | 'jargon-heavy';

export type ContentGoal =
  | 'educate'
  | 'entertain'
  | 'inspire'
  | 'sell'
  | 'build-authority'
  | 'go-viral'
  | 'build-community'
  | 'drive-traffic';

// Niche options
export const NICHE_OPTIONS = [
  'Technology',
  'Business & Entrepreneurship',
  'Finance & Investing',
  'Health & Fitness',
  'Lifestyle',
  'Education',
  'Comedy & Entertainment',
  'Gaming',
  'Beauty & Fashion',
  'Food & Cooking',
  'Travel',
  'Personal Development',
  'Marketing',
  'Software Development',
  'AI & Machine Learning',
  'Crypto & Web3',
  'Real Estate',
  'Parenting',
  'Relationships',
  'Other',
];

// Tone options with descriptions
export const TONE_OPTIONS: { id: ContentTone; name: string; description: string; emoji: string }[] = [
  { id: 'professional', name: 'Professional', description: 'Polished, credible, trustworthy', emoji: '💼' },
  { id: 'casual', name: 'Casual & Fun', description: 'Relaxed, friendly, approachable', emoji: '😊' },
  { id: 'motivational', name: 'Motivational', description: 'Inspiring, energizing, uplifting', emoji: '🔥' },
  { id: 'educational', name: 'Educational', description: 'Clear, informative, patient', emoji: '📚' },
  { id: 'provocative', name: 'Provocative', description: 'Bold, challenging, thought-provoking', emoji: '💥' },
  { id: 'storytelling', name: 'Storytelling', description: 'Narrative, engaging, emotional', emoji: '📖' },
  { id: 'humorous', name: 'Humorous', description: 'Funny, witty, entertaining', emoji: '😂' },
  { id: 'authoritative', name: 'Authoritative', description: 'Expert, commanding, decisive', emoji: '👑' },
];

// Hook style options
export const HOOK_OPTIONS: { id: HookStyle; name: string; example: string }[] = [
  { id: 'question', name: 'Question', example: '"Want to know the secret to...?"' },
  { id: 'controversial', name: 'Controversial', example: '"Everyone is wrong about..."' },
  { id: 'statistic', name: 'Statistic', example: '"90% of people don\'t know..."' },
  { id: 'story', name: 'Story', example: '"Last week, something crazy happened..."' },
  { id: 'pattern-interrupt', name: 'Pattern Interrupt', example: '"STOP! Before you scroll..."' },
  { id: 'curiosity-gap', name: 'Curiosity Gap', example: '"This changed everything for me..."' },
  { id: 'bold-claim', name: 'Bold Claim', example: '"This is the ONLY way to..."' },
];

// Goal options
export const GOAL_OPTIONS: { id: ContentGoal; name: string; emoji: string }[] = [
  { id: 'educate', name: 'Educate', emoji: '🎓' },
  { id: 'entertain', name: 'Entertain', emoji: '🎬' },
  { id: 'inspire', name: 'Inspire', emoji: '✨' },
  { id: 'sell', name: 'Sell', emoji: '💰' },
  { id: 'build-authority', name: 'Build Authority', emoji: '👑' },
  { id: 'go-viral', name: 'Go Viral', emoji: '🚀' },
  { id: 'build-community', name: 'Build Community', emoji: '🤝' },
  { id: 'drive-traffic', name: 'Drive Traffic', emoji: '🔗' },
];

// Demo profiles
export const DEMO_PROFILES: CreatorProfile[] = [
  {
    id: 'profile-1',
    name: 'Tech Expert',
    isDefault: true,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
    brandName: 'Tech with Alex',
    tagline: 'Making tech simple for everyone',
    uniqueValue: 'I explain complex tech concepts in 60 seconds or less',
    niche: 'Technology',
    subNiches: ['Software Architecture', 'Startups', 'Clean Code'],
    tone: 'educational',
    hookStyle: 'pattern-interrupt',
    ctaStyle: 'Follow for daily tech tips!',
    audienceDescription: 'Developers and tech enthusiasts who want to level up',
    audienceAgeRange: '25-34',
    audienceLevel: 'intermediate',
    audiencePainPoints: ['Overwhelmed by tech choices', 'Don\'t know best practices', 'Want to build faster'],
    audienceGoals: ['Build better apps', 'Get promoted', 'Start a tech business'],
    languageLevel: 'moderate',
    primaryGoal: 'educate',
    secondaryGoals: ['build-authority', 'build-community'],
    competitors: ['@techcreator', '@codingwithmike'],
    styleNotes: 'Direct, no fluff, practical examples',
  },
  {
    id: 'profile-2',
    name: 'Business Coach',
    isDefault: false,
    createdAt: '2024-01-02T00:00:00Z',
    updatedAt: '2024-01-02T00:00:00Z',
    brandName: 'Scale Fast',
    tagline: 'From zero to 7 figures',
    uniqueValue: 'Real strategies from someone who built 3 successful businesses',
    niche: 'Business & Entrepreneurship',
    subNiches: ['Startups', 'E-commerce', 'Personal Branding'],
    tone: 'motivational',
    hookStyle: 'bold-claim',
    ctaStyle: 'DM me "SCALE" for my free guide!',
    audienceDescription: 'Aspiring entrepreneurs and small business owners',
    audienceAgeRange: '25-34',
    audienceLevel: 'beginner',
    audiencePainPoints: ['Stuck in 9-5', 'Don\'t know where to start', 'Fear of failure'],
    audienceGoals: ['Quit their job', 'Financial freedom', 'Build a legacy'],
    languageLevel: 'simple',
    primaryGoal: 'inspire',
    secondaryGoals: ['sell', 'build-community'],
    competitors: ['@garyvee', '@hormozi'],
    styleNotes: 'High energy, lots of proof, transformation stories',
  },
];

// Local storage key
const PROFILES_STORAGE_KEY = 'viralify_creator_profiles';

// Get all profiles
export function getCreatorProfiles(): CreatorProfile[] {
  if (typeof window === 'undefined') return DEMO_PROFILES;

  const stored = localStorage.getItem(PROFILES_STORAGE_KEY);
  if (!stored) {
    localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(DEMO_PROFILES));
    return DEMO_PROFILES;
  }

  try {
    return JSON.parse(stored);
  } catch {
    return DEMO_PROFILES;
  }
}

// Save profiles
export function saveCreatorProfiles(profiles: CreatorProfile[]): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(profiles));
}

// Get default profile
export function getDefaultProfile(): CreatorProfile | null {
  const profiles = getCreatorProfiles();
  return profiles.find(p => p.isDefault) || profiles[0] || null;
}

// Create new profile
export function createProfile(profile: Omit<CreatorProfile, 'id' | 'createdAt' | 'updatedAt'>): CreatorProfile {
  const profiles = getCreatorProfiles();

  const newProfile: CreatorProfile = {
    ...profile,
    id: `profile-${Date.now()}`,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };

  // If this is the first profile or marked as default, update others
  if (newProfile.isDefault || profiles.length === 0) {
    profiles.forEach(p => p.isDefault = false);
    newProfile.isDefault = true;
  }

  profiles.push(newProfile);
  saveCreatorProfiles(profiles);

  return newProfile;
}

// Update profile
export function updateProfile(id: string, updates: Partial<CreatorProfile>): CreatorProfile | null {
  const profiles = getCreatorProfiles();
  const index = profiles.findIndex(p => p.id === id);

  if (index === -1) return null;

  // If setting as default, unset others
  if (updates.isDefault) {
    profiles.forEach(p => p.isDefault = false);
  }

  profiles[index] = {
    ...profiles[index],
    ...updates,
    updatedAt: new Date().toISOString(),
  };

  saveCreatorProfiles(profiles);
  return profiles[index];
}

// Delete profile
export function deleteProfile(id: string): boolean {
  const profiles = getCreatorProfiles();
  const index = profiles.findIndex(p => p.id === id);

  if (index === -1) return false;

  const wasDefault = profiles[index].isDefault;
  profiles.splice(index, 1);

  // If deleted profile was default, set first remaining as default
  if (wasDefault && profiles.length > 0) {
    profiles[0].isDefault = true;
  }

  saveCreatorProfiles(profiles);
  return true;
}

// Build system prompt from profile
export function buildSystemPrompt(profile: CreatorProfile, topic: string, duration: number): string {
  const toneDescription = TONE_OPTIONS.find(t => t.id === profile.tone)?.description || profile.tone;
  const hookDescription = HOOK_OPTIONS.find(h => h.id === profile.hookStyle)?.example || '';

  return `You are a content scriptwriter for a TikTok/short-form video creator.

## CREATOR PROFILE
- Brand: ${profile.brandName}
- Tagline: "${profile.tagline}"
- Unique Value: ${profile.uniqueValue}
- Niche: ${profile.niche} (${profile.subNiches.join(', ')})

## CONTENT STYLE
- Tone: ${profile.tone} (${toneDescription})
- Hook Style: ${profile.hookStyle} - Example: ${hookDescription}
- CTA Style: "${profile.ctaStyle}"
- Style Notes: ${profile.styleNotes}

## TARGET AUDIENCE
- Who: ${profile.audienceDescription}
- Age Range: ${profile.audienceAgeRange}
- Level: ${profile.audienceLevel}
- Pain Points: ${profile.audiencePainPoints.join(', ')}
- Goals: ${profile.audienceGoals.join(', ')}
- Language Level: ${profile.languageLevel}

## CONTENT GOALS
- Primary: ${profile.primaryGoal}
- Secondary: ${profile.secondaryGoals.join(', ')}

## TASK
Create a ${duration}-second video script about: "${topic}"

## OUTPUT FORMAT
Return a structured script as a JSON array with scenes. Each scene should have:
- time: timestamp range (e.g., "0:00-0:05")
- visual: detailed description of what should appear on screen
- audio: the exact script/voiceover text

The script should:
1. Start with a strong hook matching the creator's hook style
2. Deliver value matching the audience's pain points and goals
3. Use language appropriate for the audience level
4. End with a CTA matching the creator's style
5. Match the specified tone throughout

Example output format:
[
  { "time": "0:00-0:05", "visual": "Hook visual description", "audio": "Hook script text" },
  { "time": "0:05-0:15", "visual": "Content visual description", "audio": "Content script text" },
  ...
]`;
}

// Generate empty profile template
export function createEmptyProfile(): Omit<CreatorProfile, 'id' | 'createdAt' | 'updatedAt'> {
  return {
    name: '',
    isDefault: false,
    brandName: '',
    tagline: '',
    uniqueValue: '',
    niche: '',
    subNiches: [],
    tone: 'educational',
    hookStyle: 'question',
    ctaStyle: 'Follow for more!',
    audienceDescription: '',
    audienceAgeRange: '25-34',
    audienceLevel: 'intermediate',
    audiencePainPoints: [],
    audienceGoals: [],
    languageLevel: 'moderate',
    primaryGoal: 'educate',
    secondaryGoals: [],
    competitors: [],
    styleNotes: '',
  };
}
