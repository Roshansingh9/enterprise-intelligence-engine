import { useState, useCallback } from 'react';
import type { SearchResult, Ticket, SystemMetrics, KBArticle } from '@/types/support';

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

export function useSupportSystem() {
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [tickets, setTickets] = useState<Ticket[]>(mockTickets);
  const [metrics, setMetrics] = useState<SystemMetrics>(mockMetrics);

  const searchKnowledge = useCallback(async (query: string): Promise<SearchResult | null> => {
    setIsSearching(true);
    setSearchResult(null);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Simple keyword matching for demo
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
    // In real implementation, this would update learning events
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
  }, []);

  const rejectKB = useCallback(async (ticketId: string, reason: string) => {
    console.log(`KB rejected for ${ticketId}: ${reason}`);
    setTickets(prev => prev.map(t => 
      t.id === ticketId 
        ? { ...t, kbDraft: { ...t.kbDraft, status: 'rejected' as const } }
        : t
    ));
  }, []);

  return {
    isSearching,
    searchResult,
    tickets,
    metrics,
    searchKnowledge,
    raiseTicket,
    submitFeedback,
    resolveTicket,
    generateKBDraft,
    approveKB,
    rejectKB,
    setSearchResult
  };
}
