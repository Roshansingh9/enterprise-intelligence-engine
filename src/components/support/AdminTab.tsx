import { useState } from 'react';
import { 
  FileText, 
  Clock, 
  CheckCircle2, 
  XCircle, 
  Sparkles, 
  TrendingUp,
  Database,
  AlertTriangle,
  Loader2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { TrainingStatusPanel } from './TrainingStatusPanel';
import type { Ticket, SystemMetrics, KBArticle, TrainingState, TrainingStep, TrainingCheckpoint } from '@/types/support';

interface AdminTabProps {
  tickets: Ticket[];
  metrics: SystemMetrics;
  trainingState: TrainingState;
  checkpoints: TrainingCheckpoint[];
  shouldTriggerTraining: boolean;
  trainingSteps: Record<TrainingStep, { label: string; description: string }>;
  onResolveTicket: (ticketId: string, resolution: string) => Promise<void>;
  onGenerateKB: (ticketId: string) => Promise<Partial<KBArticle>>;
  onApproveKB: (ticketId: string, editedDraft?: Partial<KBArticle>) => Promise<void>;
  onRejectKB: (ticketId: string, reason: string) => Promise<void>;
  onStartTraining: () => void;
  onPauseTraining: () => void;
  onResumeTraining: () => void;
}

export function AdminTab({
  tickets,
  metrics,
  trainingState,
  checkpoints,
  shouldTriggerTraining,
  trainingSteps,
  onResolveTicket,
  onGenerateKB,
  onApproveKB,
  onRejectKB,
  onStartTraining,
  onPauseTraining,
  onResumeTraining
}: AdminTabProps) {
  const [selectedTicket, setSelectedTicket] = useState<Ticket | null>(null);
  const [resolution, setResolution] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [rejectReason, setRejectReason] = useState('');
  const [showRejectInput, setShowRejectInput] = useState(false);

  const openTickets = tickets.filter(t => t.status === 'open' || t.status === 'in_progress');
  const resolvedTickets = tickets.filter(t => t.status === 'resolved' || t.status === 'closed');

  const handleSelectTicket = (ticket: Ticket) => {
    setSelectedTicket(ticket);
    setResolution(ticket.resolution || '');
    setShowRejectInput(false);
    setRejectReason('');
  };

  const handleResolve = async () => {
    if (!selectedTicket || !resolution.trim()) return;
    await onResolveTicket(selectedTicket.id, resolution);
    setSelectedTicket(prev => prev ? { ...prev, status: 'resolved', resolution } : null);
  };

  const handleGenerateKB = async () => {
    if (!selectedTicket) return;
    setIsGenerating(true);
    await onGenerateKB(selectedTicket.id);
    setSelectedTicket(prev => {
      const updated = tickets.find(t => t.id === prev?.id);
      return updated || prev;
    });
    setIsGenerating(false);
  };

  const handleApprove = async () => {
    if (!selectedTicket) return;
    await onApproveKB(selectedTicket.id);
    setSelectedTicket(null);
  };

  const handleReject = async () => {
    if (!selectedTicket || !rejectReason.trim()) return;
    await onRejectKB(selectedTicket.id, rejectReason);
    setShowRejectInput(false);
    setRejectReason('');
  };

  const getStatusColor = (status: Ticket['status']) => {
    switch (status) {
      case 'open': return 'bg-red-100 text-red-700 border-red-200';
      case 'in_progress': return 'bg-yellow-100 text-yellow-700 border-yellow-200';
      case 'resolved': return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'closed': return 'bg-green-100 text-green-700 border-green-200';
    }
  };

  const getConfidenceStatus = (confidence: number) => {
    if (confidence >= 0.85) return { color: 'text-green-600', label: 'Auto-Approve', icon: CheckCircle2 };
    if (confidence >= 0.7) return { color: 'text-yellow-600', label: 'Review Recommended', icon: AlertTriangle };
    return { color: 'text-red-600', label: 'Human Validation Required', icon: XCircle };
  };

  // Check if training is blocking approvals
  const isApprovalBlocked = trainingState.status === 'in_progress';

  return (
    <div className="space-y-6">
      {/* Training Status Panel - Always visible at top */}
      <TrainingStatusPanel
        trainingState={trainingState}
        checkpoints={checkpoints}
        shouldTriggerTraining={shouldTriggerTraining}
        trainingSteps={trainingSteps}
        onStartTraining={onStartTraining}
        onPauseTraining={onPauseTraining}
        onResumeTraining={onResumeTraining}
      />

      {/* Approval blocked warning */}
      {isApprovalBlocked && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-center gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-600" />
          <div>
            <span className="font-medium text-amber-800">Learning cycle in progress</span>
            <p className="text-sm text-amber-700">KB approvals are temporarily paused until training completes.</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Metrics & Ticket List */}
        <div className="space-y-6">
        {/* Inline Metrics */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              System Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-primary">{metrics.kbCount.toLocaleString()}</div>
                <div className="text-xs text-muted-foreground">KB Articles</div>
              </div>
              <div className="text-center p-3 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-primary">{metrics.coveragePercent}%</div>
                <div className="text-xs text-muted-foreground">Coverage</div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Learning Progress</span>
                <span className="font-medium">{metrics.learningProgress}%</span>
              </div>
              <Progress value={metrics.learningProgress} className="h-2" />
            </div>

            <div className="text-xs text-muted-foreground flex items-center gap-1">
              <Database className="h-3 w-3" />
              Last checkpoint: {new Date(metrics.lastCheckpoint).toLocaleString()}
            </div>

            {/* Mini Accuracy Trend */}
            <div className="pt-2">
              <div className="text-xs text-muted-foreground mb-2">Accuracy Trend</div>
              <div className="flex items-end gap-1 h-12">
                {metrics.accuracyTrend.map((val, idx) => (
                  <div 
                    key={idx}
                    className="flex-1 bg-primary/20 rounded-t"
                    style={{ height: `${val}%` }}
                  />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Pending Tickets */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Pending Tickets
              <Badge variant="secondary" className="ml-auto">
                {openTickets.length}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 max-h-[400px] overflow-y-auto">
            {openTickets.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                No pending tickets
              </p>
            ) : (
              openTickets.map(ticket => (
                <div
                  key={ticket.id}
                  onClick={() => handleSelectTicket(ticket)}
                  className={`p-3 rounded-lg border cursor-pointer transition-colors hover:bg-muted/50 ${
                    selectedTicket?.id === ticket.id ? 'border-primary bg-primary/5' : ''
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <Badge variant="outline" className="text-xs">
                      {ticket.id}
                    </Badge>
                    <Badge className={`text-xs ${getStatusColor(ticket.status)}`}>
                      {ticket.status}
                    </Badge>
                  </div>
                  <p className="text-sm line-clamp-2">{ticket.issueText}</p>
                  <div className="text-xs text-muted-foreground mt-1">
                    {new Date(ticket.createdAt).toLocaleDateString()}
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>

      {/* Right Column - Ticket Detail & Workflow */}
      <div className="lg:col-span-2">
        {!selectedTicket ? (
          <Card className="h-full flex items-center justify-center">
            <CardContent className="text-center py-12">
              <FileText className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-muted-foreground">Select a Ticket</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Click on a ticket from the queue to view details and resolve
              </p>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <div className="flex items-start justify-between">
                <div>
                  <Badge variant="outline" className="mb-2">{selectedTicket.id}</Badge>
                  <CardTitle>Ticket Details</CardTitle>
                  <CardDescription className="mt-2">
                    Created: {new Date(selectedTicket.createdAt).toLocaleString()}
                  </CardDescription>
                </div>
                <Badge className={getStatusColor(selectedTicket.status)}>
                  {selectedTicket.status.toUpperCase()}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Original Issue */}
              <div>
                <h4 className="font-semibold mb-2 text-sm">Original Issue</h4>
                <div className="bg-muted/50 rounded-lg p-4 text-sm">
                  {selectedTicket.issueText}
                </div>
              </div>

              {/* AI Analysis */}
              {selectedTicket.aiAnalysis && (
                <div>
                  <h4 className="font-semibold mb-2 text-sm flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-purple-500" />
                    AI Preliminary Analysis
                  </h4>
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 text-sm text-purple-800">
                    {selectedTicket.aiAnalysis}
                  </div>
                </div>
              )}

              <Separator />

              {/* Step 2: Resolution Input */}
              {(selectedTicket.status === 'open' || selectedTicket.status === 'in_progress') && (
                <div>
                  <h4 className="font-semibold mb-2 text-sm">Step 2: Write Resolution</h4>
                  <Textarea
                    placeholder="Enter the correct resolution for this issue..."
                    value={resolution}
                    onChange={(e) => setResolution(e.target.value)}
                    className="min-h-[120px]"
                  />
                  <Button 
                    onClick={handleResolve} 
                    className="mt-3"
                    disabled={!resolution.trim()}
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    Submit Resolution
                  </Button>
                </div>
              )}

              {/* Step 3: KB Generation */}
              {selectedTicket.status === 'resolved' && !selectedTicket.kbDraft && (
                <div>
                  <h4 className="font-semibold mb-2 text-sm">Step 3: Generate KB Article</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    AI will structure the resolution into a searchable knowledge base article.
                  </p>
                  <Button onClick={handleGenerateKB} disabled={isGenerating}>
                    {isGenerating ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4" />
                        Generate KB Draft
                      </>
                    )}
                  </Button>
                </div>
              )}

              {/* Step 4: KB Review & Approval */}
              {selectedTicket.kbDraft && (
                <div className="space-y-4">
                  <h4 className="font-semibold text-sm">Step 4: Review KB Draft</h4>
                  
                  {/* Confidence Check */}
                  {selectedTicket.kbDraft.confidence && (
                    <div className={`flex items-center gap-2 p-3 rounded-lg border ${
                      selectedTicket.kbDraft.confidence >= 0.85 
                        ? 'bg-green-50 border-green-200' 
                        : selectedTicket.kbDraft.confidence >= 0.7
                          ? 'bg-yellow-50 border-yellow-200'
                          : 'bg-red-50 border-red-200'
                    }`}>
                      {(() => {
                        const status = getConfidenceStatus(selectedTicket.kbDraft.confidence);
                        return (
                          <>
                            <status.icon className={`h-5 w-5 ${status.color}`} />
                            <span className={`font-medium ${status.color}`}>
                              {Math.round(selectedTicket.kbDraft.confidence * 100)}% Confidence - {status.label}
                            </span>
                          </>
                        );
                      })()}
                    </div>
                  )}

                  {/* Draft Preview */}
                  <div className="border rounded-lg p-4 space-y-3">
                    <div>
                      <label className="text-xs text-muted-foreground">Title</label>
                      <p className="font-medium">{selectedTicket.kbDraft.title}</p>
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">Content</label>
                      <p className="text-sm text-muted-foreground">{selectedTicket.kbDraft.content}</p>
                    </div>
                    {selectedTicket.kbDraft.steps && selectedTicket.kbDraft.steps.length > 0 && (
                      <div>
                        <label className="text-xs text-muted-foreground">Steps</label>
                        <ol className="text-sm list-decimal list-inside text-muted-foreground">
                          {selectedTicket.kbDraft.steps.map((step, idx) => (
                            <li key={idx}>{step}</li>
                          ))}
                        </ol>
                      </div>
                    )}
                  </div>

                  {/* Approval Actions */}
                  {selectedTicket.kbDraft.status !== 'approved' && (
                    <div className="flex gap-3">
                      <Button onClick={handleApprove} className="flex-1">
                        <CheckCircle2 className="h-4 w-4" />
                        Approve & Publish
                      </Button>
                      {!showRejectInput ? (
                        <Button 
                          variant="outline" 
                          className="text-red-600 border-red-300"
                          onClick={() => setShowRejectInput(true)}
                        >
                          <XCircle className="h-4 w-4" />
                          Reject
                        </Button>
                      ) : (
                        <div className="flex gap-2 flex-1">
                          <Textarea
                            placeholder="Reason for rejection..."
                            value={rejectReason}
                            onChange={(e) => setRejectReason(e.target.value)}
                            className="h-10 min-h-0"
                          />
                          <Button 
                            variant="destructive" 
                            onClick={handleReject}
                            disabled={!rejectReason.trim()}
                          >
                            Confirm
                          </Button>
                        </div>
                      )}
                    </div>
                  )}

                  {selectedTicket.kbDraft.status === 'approved' && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3">
                      <CheckCircle2 className="h-5 w-5 text-green-600" />
                      <div>
                        <span className="text-green-800 font-medium">KB Article Published</span>
                        <p className="text-sm text-green-700">
                          This knowledge is now searchable and will help future queries.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
    </div>
  );
}
