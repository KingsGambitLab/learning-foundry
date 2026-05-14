export interface VivaQuestion {
  prompt: string;
}

export interface ChatReply {
  reply: string;
  hint_tier: number | null;
}

export interface SubmitResult {
  test_results: { passed: boolean; details: string };
  viva_questions: VivaQuestion[];
}
