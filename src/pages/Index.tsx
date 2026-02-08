import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { User, Settings } from 'lucide-react';
import { SupportAgentTab } from '@/components/support/SupportAgentTab';
import { AdminTab } from '@/components/support/AdminTab';
import { useSupportSystem } from '@/hooks/useSupportSystem';

const Index = () => {
  const {
    isSearching,
    searchResult,
    tickets,
    metrics,
    trainingState,
    checkpoints,
    shouldTriggerTraining,
    trainingSteps,
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
  } = useSupportSystem();

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground">SupportMind AI</h1>
              <p className="text-sm text-muted-foreground">Self-Learning Knowledge Engine</p>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
              System Online
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <Tabs defaultValue="support" className="space-y-6">
          <TabsList className="grid w-full max-w-md grid-cols-2">
            <TabsTrigger value="support" className="flex items-center gap-2">
              <User className="h-4 w-4" />
              🧑‍💼 Support Agent / User
            </TabsTrigger>
            <TabsTrigger value="admin" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              🛠️ Admin & Knowledge Manager
            </TabsTrigger>
          </TabsList>

          <TabsContent value="support" className="mt-6">
            <div className="max-w-3xl mx-auto">
              <SupportAgentTab
                isSearching={isSearching}
                searchResult={searchResult}
                onSearch={searchKnowledge}
                onRaiseTicket={raiseTicket}
                onFeedback={submitFeedback}
                onClearResult={() => setSearchResult(null)}
              />
            </div>
          </TabsContent>

          <TabsContent value="admin" className="mt-6">
            <AdminTab
              tickets={tickets}
              metrics={metrics}
              trainingState={trainingState}
              checkpoints={checkpoints}
              shouldTriggerTraining={shouldTriggerTraining}
              trainingSteps={trainingSteps}
              onResolveTicket={resolveTicket}
              onGenerateKB={generateKBDraft}
              onApproveKB={approveKB}
              onRejectKB={rejectKB}
              onStartTraining={runTrainingCycle}
              onPauseTraining={pauseTraining}
              onResumeTraining={resumeTraining}
            />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Index;
