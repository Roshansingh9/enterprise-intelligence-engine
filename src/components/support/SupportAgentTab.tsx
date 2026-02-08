import { useState } from 'react';
import { Search, CheckCircle2, XCircle, AlertTriangle, ExternalLink, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import type { SearchResult, Ticket } from '@/types/support';

interface SupportAgentTabProps {
  isSearching: boolean;
  searchResult: SearchResult | null;
  onSearch: (query: string) => Promise<SearchResult | null>;
  onRaiseTicket: (issueText: string) => Promise<Ticket>;
  onFeedback: (articleId: string, helpful: boolean) => void;
  onClearResult: () => void;
}

export function SupportAgentTab({
  isSearching,
  searchResult,
  onSearch,
  onRaiseTicket,
  onFeedback,
  onClearResult
}: SupportAgentTabProps) {
  const [query, setQuery] = useState('');
  const [ticketRaised, setTicketRaised] = useState<Ticket | null>(null);
  const [feedbackGiven, setFeedbackGiven] = useState<'solved' | 'not_helpful' | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setTicketRaised(null);
    setFeedbackGiven(null);
    await onSearch(query);
  };

  const handleRaiseTicket = async () => {
    const ticket = await onRaiseTicket(query);
    setTicketRaised(ticket);
  };

  const handleFeedback = (helpful: boolean) => {
    if (searchResult) {
      onFeedback(searchResult.article.id, helpful);
      setFeedbackGiven(helpful ? 'solved' : 'not_helpful');
      if (!helpful) {
        // Show option to raise ticket
      }
    }
  };

  const handleNewSearch = () => {
    setQuery('');
    setTicketRaised(null);
    setFeedbackGiven(null);
    onClearResult();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.85) return 'text-green-600 bg-green-50 border-green-200';
    if (confidence >= 0.7) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  return (
    <div className="space-y-6">
      {/* Issue Input Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Describe the Issue
          </CardTitle>
          <CardDescription>
            Enter the customer's issue or error message to find a solution
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="e.g., Customer gets error while creating profile, or 'Invalid backend reference' when processing voucher..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="min-h-[100px]"
            disabled={isSearching}
          />
          <div className="flex gap-3">
            <Button 
              onClick={handleSearch} 
              disabled={!query.trim() || isSearching}
              className="flex-1"
            >
              {isSearching ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Searching Knowledge Base...
                </>
              ) : (
                <>
                  <Search className="h-4 w-4" />
                  Find Solution
                </>
              )}
            </Button>
            {(searchResult || ticketRaised) && (
              <Button variant="outline" onClick={handleNewSearch}>
                New Search
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Search Result - KB Found */}
      {searchResult && !ticketRaised && (
        <Card className="border-l-4 border-l-primary">
          <CardHeader>
            <div className="flex items-start justify-between">
              <div>
                <Badge variant="outline" className="mb-2">
                  {searchResult.article.id}
                </Badge>
                <CardTitle className="text-xl">{searchResult.article.title}</CardTitle>
                <CardDescription className="mt-2">
                  {searchResult.article.summary}
                </CardDescription>
              </div>
              <div className={`px-3 py-1.5 rounded-md border text-sm font-medium ${getConfidenceColor(searchResult.confidence)}`}>
                {Math.round(searchResult.confidence * 100)}% confidence
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Solution Steps */}
            <div>
              <h4 className="font-semibold mb-3 flex items-center gap-2">
                🔧 Solution Steps
              </h4>
              <ol className="space-y-2">
                {searchResult.article.steps.map((step, idx) => (
                  <li key={idx} className="flex gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-sm flex items-center justify-center font-medium">
                      {idx + 1}
                    </span>
                    <span className="text-muted-foreground">
                      {step.split(/(\{\{[^}]+\}\})/).map((part, i) => 
                        part.startsWith('{{') ? (
                          <code key={i} className="px-1.5 py-0.5 bg-yellow-100 text-yellow-800 rounded text-sm font-mono">
                            {part}
                          </code>
                        ) : part
                      )}
                    </span>
                  </li>
                ))}
              </ol>
            </div>

            {/* Placeholders Notice */}
            {searchResult.article.placeholders.length > 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h5 className="font-medium text-yellow-800 mb-2">📝 Required Inputs</h5>
                <div className="flex flex-wrap gap-2">
                  {searchResult.article.placeholders.map(p => (
                    <Badge key={p} variant="outline" className="bg-yellow-100 border-yellow-300">
                      {p}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            <Separator />

            {/* Lineage */}
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                🧬 Source: <strong className="text-foreground">{searchResult.article.source.type.toUpperCase()}</strong>
              </span>
              <span className="flex items-center gap-1">
                <ExternalLink className="h-3 w-3" />
                {searchResult.article.source.id}
              </span>
              <span>Version {searchResult.article.version}</span>
              <span>Updated {new Date(searchResult.article.updatedAt).toLocaleDateString()}</span>
            </div>

            <Separator />

            {/* Feedback Section */}
            {!feedbackGiven ? (
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Did this solve the issue?</span>
                <div className="flex gap-3">
                  <Button 
                    variant="outline" 
                    className="text-green-600 border-green-300 hover:bg-green-50"
                    onClick={() => handleFeedback(true)}
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    Yes, this solved it
                  </Button>
                  <Button 
                    variant="outline"
                    className="text-red-600 border-red-300 hover:bg-red-50"
                    onClick={() => handleFeedback(false)}
                  >
                    <XCircle className="h-4 w-4" />
                    No, this didn't help
                  </Button>
                </div>
              </div>
            ) : feedbackGiven === 'solved' ? (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-600" />
                <span className="text-green-800">Feedback recorded. Thank you!</span>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 flex items-center gap-3">
                  <AlertTriangle className="h-5 w-5 text-orange-600" />
                  <span className="text-orange-800">We're sorry this didn't help. Would you like to escalate?</span>
                </div>
                <Button onClick={handleRaiseTicket} className="w-full" variant="secondary">
                  🟡 Raise Ticket for Expert Review
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* No Result Found */}
      {!isSearching && query && !searchResult && !ticketRaised && (
        <Card className="border-l-4 border-l-orange-400">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-orange-600">
              <AlertTriangle className="h-5 w-5" />
              No Verified Solution Found
            </CardTitle>
            <CardDescription>
              Our knowledge base doesn't have a verified solution for this issue yet.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={handleRaiseTicket} className="w-full" size="lg">
              🟡 Raise Ticket for Expert Review
            </Button>
            <p className="text-sm text-muted-foreground mt-3 text-center">
              A senior engineer will be notified and the resolution will be added to our knowledge base.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Ticket Raised Confirmation */}
      {ticketRaised && (
        <Card className="border-l-4 border-l-blue-400">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-blue-600">
              <CheckCircle2 className="h-5 w-5" />
              Ticket Created Successfully
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <Badge variant="outline" className="bg-blue-100">
                  {ticketRaised.id}
                </Badge>
                <Badge className="bg-blue-600">
                  {ticketRaised.status.toUpperCase()}
                </Badge>
              </div>
              <p className="text-sm text-blue-800 mt-2">
                Your issue has been escalated to a senior engineer. You will be notified once a resolution is available.
              </p>
            </div>
            <p className="text-sm text-muted-foreground">
              <strong>Issue submitted:</strong> {ticketRaised.issueText}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
