// Support AI System Types

export interface KBArticle {
  id: string;
  title: string;
  summary: string;
  content: string;
  steps: string[];
  placeholders: string[];
  confidence: number;
  version: number;
  source: {
    type: 'ticket' | 'conversation' | 'script';
    id: string;
    date: string;
  };
  tags: string[];
  status: 'approved' | 'pending_review' | 'rejected';
  createdAt: string;
  updatedAt: string;
}

export interface SearchResult {
  article: KBArticle;
  similarity: number;
  confidence: number;
}

export interface Ticket {
  id: string;
  issueText: string;
  status: 'open' | 'in_progress' | 'resolved' | 'closed';
  aiAnalysis?: string;
  resolution?: string;
  kbDraft?: Partial<KBArticle>;
  createdAt: string;
  updatedAt: string;
  linkedConversationId?: string;
}

export interface LearningEvent {
  id: string;
  type: 'search_success' | 'search_failure' | 'ticket_created' | 'kb_generated' | 'kb_approved' | 'training_started' | 'training_completed';
  ticketId?: string;
  articleId?: string;
  timestamp: string;
  status: 'pending' | 'completed';
}

export interface SystemMetrics {
  kbCount: number;
  coveragePercent: number;
  pendingTickets: number;
  learningProgress: number;
  lastCheckpoint: string;
  accuracyTrend: number[];
}

// Auto-Retraining Types
export type TrainingStatus = 'idle' | 'pending' | 'in_progress' | 'completed' | 'failed' | 'paused';

export interface TrainingCheckpoint {
  id: string;
  timestamp: string;
  kbCountAtCheckpoint: number;
  vectorIndexVersion: number;
  promptVersion: string;
  metrics: {
    hitAt1: number;
    hitAt3: number;
    mrr: number;
    coverage: number;
    kbQualityScore: number;
  };
  status: 'complete' | 'partial' | 'failed';
  resumeData?: {
    lastCompletedStep: TrainingStep;
    pendingKBIds: string[];
  };
}

export type TrainingStep = 
  | 'lock_ingestion'
  | 'create_checkpoint'
  | 'retrieval_evaluation'
  | 'prompt_optimization'
  | 'embedding_refresh'
  | 'quality_validation'
  | 'update_metrics'
  | 'unlock_system';

export interface TrainingState {
  status: TrainingStatus;
  newKBCountSinceLastTraining: number;
  triggerThreshold: number;
  currentStep?: TrainingStep;
  stepProgress: number; // 0-100
  estimatedTimeRemaining?: number; // seconds
  lastTrainingResult?: {
    startTime: string;
    endTime: string;
    kbsProcessed: number;
    accuracyBefore: number;
    accuracyAfter: number;
    coverageBefore: number;
    coverageAfter: number;
    promptsUpdated: boolean;
    embeddingsRefreshed: number;
  };
  error?: string;
}

export interface TrainingTriggerCondition {
  newKBCount: number;
  threshold: number;
  shouldTrigger: boolean;
  lastChecked: string;
}
