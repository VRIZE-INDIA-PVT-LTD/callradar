// Shared contract between pipeline and frontend.
// Kept in sync by tests/test_contract.py — change both in the same commit.
// NOTE: `audioUrl` removed. Build it from the id: `/api/calls/${id}/audio`.

export type Speaker = 'agent' | 'customer';

export interface TranscriptTurn {
  speaker: Speaker;
  startSec: number;
  endSec: number;
  text: string;
}

export interface Evidence {
  timestampSec: number;   // -1 on moodShift means: no shift detected
  quote: string;          // guaranteed verbatim from the transcript
  rationale: string;
}

export interface MoodPoint {
  minute: number;         // fractional by default (0.08 = 4.8s)
  mood: number;           // 0 = furious, 100 = delighted
  label: string;
}

export interface CallAnalysisEvidence {
  intent: Evidence;
  moodShift: Evidence;
  outcome: Evidence;
  attention: Evidence;
}

export interface CallResponseEvent { submitTimeMs: number; }

export interface CallSurveyResponse {
  submitTimeMs: number;
  data: Record<string, string>;
}

export interface CallPartyMetadata {
  arrivalTimeMs: number;
  hangupTimeMs: number;
  metadata: Record<string, string>;   // inner keys keep original casing,
                                      // e.g. "first and last name"
  responses: CallResponseEvent[];
  speakerId: number;
  surveyResponse: CallSurveyResponse;
}

export interface CallQualityLabels {
  lhvbScript: number;
  callerMos: number;
  agentMos: number;
}

export interface CallMetadata {
  agent: CallPartyMetadata;
  caller: CallPartyMetadata;
  endTimeMs: number;
  sid: string;
  startTimeMs: number;
  labels: CallQualityLabels;
  session: string;
}

/** Additive: powers the "why is this an 87?" breakdown. Safe to ignore. */
export interface AttentionFactor {
  key: string;
  points: number;
  label: string;
}

export interface CallRecord {
  id: string;
  customerId: string;
  customerName: string;
  agentId: string;
  agentName: string;
  startedAt: string;
  durationSec: number;
  summary: string;
  intent: string;
  resolved: boolean;
  needsAttention: number;      // 0-100
  moodShiftSec: number;        // -1 = none detected; hide the marker
  moodBefore: string;
  moodAfter: string;
  issueTag: string;
  transcript: TranscriptTurn[];
  moodTimeline: MoodPoint[];
  evidence: CallAnalysisEvidence;
  metadata: CallMetadata;
  needsAttentionFactors?: AttentionFactor[];
  needsReview?: boolean;
}

export interface Customer { id: string; name: string; callCount?: number; worstAttention?: number; }
export interface Agent { id: string; name: string; }

export interface TrendingIssue {
  issueTag: string;
  count: number;
  deltaVsLastWeek: number;
}

export interface AgentMetric {
  agentId: string;
  agentName: string;
  callVolume: number;
  avgHandleTimeSec: number;
  resolvedPct: number;
  escalations: number;
}
