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
  type: 'search_success' | 'search_failure' | 'ticket_created' | 'kb_generated' | 'kb_approved';
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
