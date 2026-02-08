import { useState, useCallback, useEffect, useRef } from 'react';
import type { SearchResult, Ticket, SystemMetrics, KBArticle, TrainingState, TrainingStep, TrainingCheckpoint } from '@/types/support';

// Auto-retraining constants
const KB_TRAINING_THRESHOLD = 100;

// Mock data for demonstration
const mockKBArticles: KBArticle[] = [
  {
    id: 'KB-640CAF35',
    title: 'Invalid Backend Voucher Reference for Date Advance',
    summary: 'Resolution for voucher reference errors when processing date advances in PropertySuite.',
    content: 'This issue occurs when the backend voucher reference becomes out of sync with the certification data.',
    steps: [
      'Navigate to the Certification module',
      'Select the affected voucher record',
      'Click "Refresh Backend Reference"',
      'Verify the {{VOUCHER_ID}} matches the certification',
      'Reprocess the date advance'
    ],
    placeholders: ['VOUCHER_ID'],
    confidence: 0.92,
    version: 2,
    source: { type: 'ticket', id: 'TKT-2024-1842', date: '2024-01-15' },
    tags: ['voucher', 'backend', 'date-advance'],
    status: 'approved',
    createdAt: '2024-01-15T10:30:00Z',
    updatedAt: '2024-02-08T14:20:00Z'
  },
  {
    id: 'KB-9E50469F',
    title: 'Invalid Backend Certification Reference',
    summary: 'Fix for certification reference mismatches in affordable housing workflows.',
    content: 'Certification references can become invalid when backend data is modified outside the normal workflow.',
    steps: [
      'Open the affected unit in PropertySuite Affordable',
      'Go to Certifications tab',
      'Locate the certification with status "Error"',
      'Click "Validate References"',
      'If validation fails, contact {{ADMIN_EMAIL}} for backend correction'
    ],
    placeholders: ['ADMIN_EMAIL'],
    confidence: 0.88,
    version: 1,
    source: { type: 'conversation', id: 'CONV-8834', date: '2024-01-20' },
    tags: ['certification', 'backend', 'affordable'],
    status: 'approved',
    createdAt: '2024-01-20T09:15:00Z',
    updatedAt: '2024-01-20T09:15:00Z'
  }
];

const mockTickets: Ticket[] = [
  {
    id: 'TKT-2024-2103',
    issueText: 'Customer cannot generate HAP voucher after interim recertification. System shows "Reference mismatch" error.',
    status: 'open',
    aiAnalysis: 'Likely related to certification sync issue. Similar to KB-9E50469F but involves HAP specifically.',
    createdAt: '2024-02-08T11:30:00Z',
    updatedAt: '2024-02-08T11:30:00Z',
    linkedConversationId: 'CONV-9102'
  },
  {
    id: 'TKT-2024-2098',
    issueText: 'Move-out process stuck at final statement generation. No error shown but button is greyed out.',
    status: 'in_progress',
    aiAnalysis: 'Possibly blocked by pending ledger items or deposit reconciliation.',
    resolution: 'Check for pending ledger items and ensure all deposits are reconciled before attempting move-out.',
    createdAt: '2024-02-07T16:45:00Z',
    updatedAt: '2024-02-08T10:00:00Z'
  }
];

const mockMetrics: SystemMetrics = {
  kbCount: 3207,
  coveragePercent: 78.4,
  pendingTickets: 12,
  learningProgress: 94,
  lastCheckpoint: '2024-02-08T12:52:21Z',
  accuracyTrend: [72, 74, 75, 76, 78, 78.4]
};

const initialTrainingState: TrainingState = {
  status: 'idle',
  newKBCountSinceLastTraining: 87, // Demo: close to threshold
  triggerThreshold: KB_TRAINING_THRESHOLD,
  stepProgress: 0
};

// Training step descriptions for UI
const TRAINING_STEPS: Record<TrainingStep, { label: string; description: string }> = {
  lock_ingestion: { label: 'Locking Ingestion', description: 'Pausing new KB approvals...' },
  create_checkpoint: { label: 'Creating Checkpoint', description: 'Saving system state...' },
  retrieval_evaluation: { label: 'Evaluating Retrieval', description: 'Running Hit@K and MRR tests...' },
  prompt_optimization: { label: 'Optimizing Prompts', description: 'Adjusting extraction prompts...' },
  embedding_refresh: { label: 'Refreshing Embeddings', description: 'Re-embedding new KBs...' },
  quality_validation: { label: 'Validating Quality', description: 'Checking KB quality scores...' },
  update_metrics: { label: 'Updating Metrics', description: 'Persisting new metrics...' },
  unlock_system: { label: 'Unlocking System', description: 'Resuming normal operations...' }
};

const TRAINING_STEP_ORDER: TrainingStep[] = [
  'lock_ingestion',
  'create_checkpoint',
  'retrieval_evaluation',
  'prompt_optimization',
  'embedding_refresh',
  'quality_validation',
  'update_metrics',
  'unlock_system'
];

export function useSupportSystem() {
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [tickets, setTickets] = useState<Ticket[]>(mockTickets);
  const [metrics, setMetrics] = useState<SystemMetrics>(mockMetrics);
  const [trainingState, setTrainingState] = useState<TrainingState>(initialTrainingState);
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpoint[]>([]);
  const trainingAbortRef = useRef(false);

  // Check if training should be triggered
  const shouldTriggerTraining = trainingState.newKBCountSinceLastTraining >= trainingState.triggerThreshold;

  // Simulate training step execution
  const executeTrainingStep = async (step: TrainingStep, stepIndex: number): Promise<boolean> => {
    if (trainingAbortRef.current) return false;

    const baseProgress = (stepIndex / TRAINING_STEP_ORDER.length) * 100;
    setTrainingState(prev => ({
      ...prev,
      currentStep: step,
      stepProgress: baseProgress,
      estimatedTimeRemaining: (TRAINING_STEP_ORDER.length - stepIndex) * 15
    }));

    // Simulate step processing with incremental progress
    for (let i = 0; i <= 10; i++) {
      if (trainingAbortRef.current) return false;
      await new Promise(resolve => setTimeout(resolve, 200));
      setTrainingState(prev => ({
        ...prev,
        stepProgress: baseProgress + (i / 10) * (100 / TRAINING_STEP_ORDER.length)
      }));
    }

    return true;
  };

  // Main training cycle
  const runTrainingCycle = useCallback(async () => {
    if (trainingState.status === 'in_progress') return;

    trainingAbortRef.current = false;
    const startTime = new Date().toISOString();
    const accuracyBefore = metrics.accuracyTrend[metrics.accuracyTrend.length - 1] || 78;
    const coverageBefore = metrics.coveragePercent;

    setTrainingState(prev => ({
      ...prev,
      status: 'in_progress',
      stepProgress: 0,
      error: undefined
    }));

    try {
      // Execute each training step
      for (let i = 0; i < TRAINING_STEP_ORDER.length; i++) {
        const step = TRAINING_STEP_ORDER[i];
        const success = await executeTrainingStep(step, i);
        
        if (!success) {
          // Training was paused/aborted - save checkpoint for resume
          const checkpoint: TrainingCheckpoint = {
            id: `CP-${Date.now()}`,
            timestamp: new Date().toISOString(),
            kbCountAtCheckpoint: metrics.kbCount,
            vectorIndexVersion: 1,
            promptVersion: 'v1.2',
            metrics: {
              hitAt1: 0.72,
              hitAt3: 0.85,
              mrr: 0.78,
              coverage: metrics.coveragePercent,
              kbQualityScore: 0.82
            },
            status: 'partial',
            resumeData: {
              lastCompletedStep: TRAINING_STEP_ORDER[Math.max(0, i - 1)],
              pendingKBIds: []
            }
          };
          setCheckpoints(prev => [...prev, checkpoint]);
          setTrainingState(prev => ({
            ...prev,
            status: 'paused',
            currentStep: step
          }));
          return;
        }
      }

      // Training completed successfully
      const accuracyAfter = accuracyBefore + 1.2 + Math.random() * 0.5;
      const coverageAfter = Math.min(100, coverageBefore + 0.5);
      
      const finalCheckpoint: TrainingCheckpoint = {
        id: `CP-${Date.now()}`,
        timestamp: new Date().toISOString(),
        kbCountAtCheckpoint: metrics.kbCount,
        vectorIndexVersion: 2,
        promptVersion: 'v1.3',
        metrics: {
          hitAt1: 0.74,
          hitAt3: 0.87,
          mrr: 0.80,
          coverage: coverageAfter,
          kbQualityScore: 0.85
        },
        status: 'complete'
      };
      setCheckpoints(prev => [...prev, finalCheckpoint]);

      setTrainingState(prev => ({
        ...prev,
        status: 'completed',
        newKBCountSinceLastTraining: 0,
        stepProgress: 100,
        currentStep: undefined,
        estimatedTimeRemaining: 0,
        lastTrainingResult: {
          startTime,
          endTime: new Date().toISOString(),
          kbsProcessed: prev.newKBCountSinceLastTraining,
          accuracyBefore,
          accuracyAfter,
          coverageBefore,
          coverageAfter,
          promptsUpdated: true,
          embeddingsRefreshed: prev.newKBCountSinceLastTraining
        }
      }));

      // Update metrics with improvements
      setMetrics(prev => ({
        ...prev,
        lastCheckpoint: new Date().toISOString(),
        coveragePercent: coverageAfter,
        accuracyTrend: [...prev.accuracyTrend.slice(-5), accuracyAfter]
      }));

      // Reset to idle after a short delay
      setTimeout(() => {
        setTrainingState(prev => ({ ...prev, status: 'idle' }));
      }, 3000);

    } catch (error) {
      setTrainingState(prev => ({
        ...prev,
        status: 'failed',
        error: error instanceof Error ? error.message : 'Training failed'
      }));
    }
  }, [trainingState.status, metrics]);

  // Resume from checkpoint
  const resumeTraining = useCallback(async () => {
    const lastCheckpoint = checkpoints[checkpoints.length - 1];
    if (!lastCheckpoint?.resumeData) {
      // No resume data, start fresh
      return runTrainingCycle();
    }

    const resumeFromStep = TRAINING_STEP_ORDER.indexOf(lastCheckpoint.resumeData.lastCompletedStep);
    if (resumeFromStep === -1) {
      return runTrainingCycle();
    }

    trainingAbortRef.current = false;
    setTrainingState(prev => ({
      ...prev,
      status: 'in_progress',
      stepProgress: ((resumeFromStep + 1) / TRAINING_STEP_ORDER.length) * 100
    }));

    // Continue from next step
    for (let i = resumeFromStep + 1; i < TRAINING_STEP_ORDER.length; i++) {
      const step = TRAINING_STEP_ORDER[i];
      const success = await executeTrainingStep(step, i);
      if (!success) {
        setTrainingState(prev => ({ ...prev, status: 'paused' }));
        return;
      }
    }

    // Complete as normal
    setTrainingState(prev => ({
      ...prev,
      status: 'completed',
      newKBCountSinceLastTraining: 0,
      stepProgress: 100
    }));
  }, [checkpoints, runTrainingCycle]);

  // Pause training
  const pauseTraining = useCallback(() => {
    trainingAbortRef.current = true;
  }, []);

  const searchKnowledge = useCallback(async (query: string): Promise<SearchResult | null> => {
    setIsSearching(true);
    setSearchResult(null);
    
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const queryLower = query.toLowerCase();
    const matched = mockKBArticles.find(kb => 
      kb.title.toLowerCase().includes(queryLower) ||
      kb.content.toLowerCase().includes(queryLower) ||
      kb.tags.some(tag => queryLower.includes(tag))
    );
    
    if (matched && Math.random() > 0.3) {
      const result: SearchResult = {
        article: matched,
        similarity: 0.75 + Math.random() * 0.2,
        confidence: matched.confidence
      };
      setSearchResult(result);
      setIsSearching(false);
      return result;
    }
    
    setIsSearching(false);
    return null;
  }, []);

  const raiseTicket = useCallback(async (issueText: string): Promise<Ticket> => {
    const newTicket: Ticket = {
      id: `TKT-2024-${Date.now().toString().slice(-4)}`,
      issueText,
      status: 'open',
      aiAnalysis: 'Awaiting preliminary analysis...',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    
    setTickets(prev => [newTicket, ...prev]);
    setMetrics(prev => ({ ...prev, pendingTickets: prev.pendingTickets + 1 }));
    
    return newTicket;
  }, []);

  const submitFeedback = useCallback(async (articleId: string, helpful: boolean) => {
    console.log(`Feedback for ${articleId}: ${helpful ? 'helpful' : 'not helpful'}`);
  }, []);

  const resolveTicket = useCallback(async (ticketId: string, resolution: string) => {
    setTickets(prev => prev.map(t => 
      t.id === ticketId 
        ? { ...t, resolution, status: 'resolved' as const, updatedAt: new Date().toISOString() }
        : t
    ));
  }, []);

  const generateKBDraft = useCallback(async (ticketId: string): Promise<Partial<KBArticle>> => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const ticket = tickets.find(t => t.id === ticketId);
    if (!ticket) throw new Error('Ticket not found');
    
    const draft: Partial<KBArticle> = {
      title: `Resolution for ${ticket.issueText.slice(0, 50)}...`,
      summary: ticket.resolution || '',
      content: ticket.resolution || '',
      steps: ticket.resolution?.split('.').filter(s => s.trim()).map(s => s.trim()) || [],
      placeholders: [],
      confidence: 0.7 + Math.random() * 0.25,
      tags: ['auto-generated'],
      status: 'pending_review'
    };
    
    setTickets(prev => prev.map(t => 
      t.id === ticketId ? { ...t, kbDraft: draft } : t
    ));
    
    return draft;
  }, [tickets]);

  const approveKB = useCallback(async (ticketId: string, editedDraft?: Partial<KBArticle>) => {
    // Check if training is in progress (soft lock)
    if (trainingState.status === 'in_progress') {
      console.warn('KB approval paused - learning cycle in progress');
      return;
    }

    setTickets(prev => prev.map(t => 
      t.id === ticketId 
        ? { ...t, status: 'closed' as const, kbDraft: { ...t.kbDraft, ...editedDraft, status: 'approved' as const } }
        : t
    ));
    setMetrics(prev => ({
      ...prev,
      kbCount: prev.kbCount + 1,
      pendingTickets: Math.max(0, prev.pendingTickets - 1),
      coveragePercent: Math.min(100, prev.coveragePercent + 0.1)
    }));

    // Increment new KB counter for auto-retraining trigger
    setTrainingState(prev => ({
      ...prev,
      newKBCountSinceLastTraining: prev.newKBCountSinceLastTraining + 1
    }));
  }, [trainingState.status]);

  const rejectKB = useCallback(async (ticketId: string, reason: string) => {
    console.log(`KB rejected for ${ticketId}: ${reason}`);
    setTickets(prev => prev.map(t => 
      t.id === ticketId 
        ? { ...t, kbDraft: { ...t.kbDraft, status: 'rejected' as const } }
        : t
    ));
  }, []);

  // Auto-trigger training when threshold is reached
  useEffect(() => {
    if (shouldTriggerTraining && trainingState.status === 'idle') {
      // In production, this would auto-trigger. For demo, we show a pending state.
      setTrainingState(prev => ({ ...prev, status: 'pending' }));
    }
  }, [shouldTriggerTraining, trainingState.status]);

  return {
    isSearching,
    searchResult,
    tickets,
    metrics,
    trainingState,
    checkpoints,
    shouldTriggerTraining,
    trainingSteps: TRAINING_STEPS,
    searchKnowledge,
    raiseTicket,
    submitFeedback,
    resolveTicket,
    generateKBDraft,
    approveKB,
    rejectKB,
    setSearchResult,
    runTrainingCycle,
    resumeTraining,
    pauseTraining
  };
}
