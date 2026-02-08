import { 
  RefreshCw, 
  Play, 
  Pause, 
  CheckCircle2, 
  AlertCircle, 
  Clock,
  Zap,
  TrendingUp,
  ArrowRight
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import type { TrainingState, TrainingStep, TrainingCheckpoint } from '@/types/support';

interface TrainingStatusPanelProps {
  trainingState: TrainingState;
  checkpoints: TrainingCheckpoint[];
  shouldTriggerTraining: boolean;
  trainingSteps: Record<TrainingStep, { label: string; description: string }>;
  onStartTraining: () => void;
  onPauseTraining: () => void;
  onResumeTraining: () => void;
}

export function TrainingStatusPanel({
  trainingState,
  checkpoints,
  shouldTriggerTraining,
  trainingSteps,
  onStartTraining,
  onPauseTraining,
  onResumeTraining
}: TrainingStatusPanelProps) {
  const getStatusBadge = () => {
    switch (trainingState.status) {
      case 'idle':
        return <Badge variant="outline" className="bg-muted text-muted-foreground">Idle</Badge>;
      case 'pending':
        return <Badge className="bg-amber-100 text-amber-700 border-amber-200">Ready to Train</Badge>;
      case 'in_progress':
        return <Badge className="bg-blue-100 text-blue-700 border-blue-200 animate-pulse">Training...</Badge>;
      case 'completed':
        return <Badge className="bg-green-100 text-green-700 border-green-200">Completed</Badge>;
      case 'failed':
        return <Badge className="bg-red-100 text-red-700 border-red-200">Failed</Badge>;
      case 'paused':
        return <Badge className="bg-yellow-100 text-yellow-700 border-yellow-200">Paused</Badge>;
    }
  };

  const formatTime = (seconds?: number) => {
    if (!seconds) return '--';
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  };

  const progressToThreshold = Math.min(100, (trainingState.newKBCountSinceLastTraining / trainingState.triggerThreshold) * 100);

  return (
    <Card className="border-2 border-dashed border-primary/30">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <RefreshCw className={`h-5 w-5 text-primary ${trainingState.status === 'in_progress' ? 'animate-spin' : ''}`} />
            <CardTitle className="text-lg">Continuous Learning</CardTitle>
          </div>
          {getStatusBadge()}
        </div>
        <CardDescription>
          Auto-retraining triggers at {trainingState.triggerThreshold} new approved KBs
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* KB Counter Progress */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-muted-foreground">New KBs since last training</span>
            <span className="font-medium">
              {trainingState.newKBCountSinceLastTraining} / {trainingState.triggerThreshold}
            </span>
          </div>
          <Progress 
            value={progressToThreshold} 
            className={`h-3 ${progressToThreshold >= 100 ? 'bg-amber-100' : ''}`}
          />
          {shouldTriggerTraining && trainingState.status === 'idle' && (
            <p className="text-xs text-amber-600 mt-1 flex items-center gap-1">
              <Zap className="h-3 w-3" />
              Threshold reached! Training recommended.
            </p>
          )}
        </div>

        {/* Training In Progress */}
        {trainingState.status === 'in_progress' && trainingState.currentStep && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-medium text-blue-800">
                {trainingSteps[trainingState.currentStep].label}
              </span>
              <span className="text-sm text-blue-600">
                ETA: {formatTime(trainingState.estimatedTimeRemaining)}
              </span>
            </div>
            <p className="text-sm text-blue-700">
              {trainingSteps[trainingState.currentStep].description}
            </p>
            <Progress value={trainingState.stepProgress} className="h-2" />
            <p className="text-xs text-blue-600 text-right">
              {Math.round(trainingState.stepProgress)}% complete
            </p>
          </div>
        )}

        {/* Paused State */}
        {trainingState.status === 'paused' && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-center gap-2 text-yellow-800">
              <Pause className="h-4 w-4" />
              <span className="font-medium">Training Paused</span>
            </div>
            <p className="text-sm text-yellow-700 mt-1">
              Checkpoint saved. Resume anytime without data loss.
            </p>
          </div>
        )}

        {/* Error State */}
        {trainingState.status === 'failed' && trainingState.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center gap-2 text-red-800">
              <AlertCircle className="h-4 w-4" />
              <span className="font-medium">Training Failed</span>
            </div>
            <p className="text-sm text-red-700 mt-1">{trainingState.error}</p>
          </div>
        )}

        {/* Last Training Result */}
        {trainingState.lastTrainingResult && trainingState.status !== 'in_progress' && (
          <div className="bg-muted/50 rounded-lg p-4 space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              Last Training Summary
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Accuracy:</span>
                <span className="font-medium">
                  {trainingState.lastTrainingResult.accuracyBefore.toFixed(1)}%
                </span>
                <ArrowRight className="h-3 w-3 text-green-500" />
                <span className="font-medium text-green-600">
                  {trainingState.lastTrainingResult.accuracyAfter.toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">Coverage:</span>
                <span className="font-medium">
                  {trainingState.lastTrainingResult.coverageBefore.toFixed(1)}%
                </span>
                <ArrowRight className="h-3 w-3 text-green-500" />
                <span className="font-medium text-green-600">
                  {trainingState.lastTrainingResult.coverageAfter.toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="flex items-center gap-4 text-xs text-muted-foreground pt-1">
              <span>{trainingState.lastTrainingResult.kbsProcessed} KBs processed</span>
              <span>•</span>
              <span>{trainingState.lastTrainingResult.embeddingsRefreshed} embeddings refreshed</span>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-2 pt-2">
          {(trainingState.status === 'idle' || trainingState.status === 'pending') && (
            <Button 
              onClick={onStartTraining} 
              className="flex-1"
              variant={shouldTriggerTraining ? 'default' : 'outline'}
            >
              <Play className="h-4 w-4" />
              Start Training
            </Button>
          )}
          {trainingState.status === 'in_progress' && (
            <Button onClick={onPauseTraining} variant="outline" className="flex-1">
              <Pause className="h-4 w-4" />
              Pause
            </Button>
          )}
          {trainingState.status === 'paused' && (
            <Button onClick={onResumeTraining} className="flex-1">
              <Play className="h-4 w-4" />
              Resume Training
            </Button>
          )}
          {trainingState.status === 'failed' && (
            <Button onClick={onStartTraining} className="flex-1">
              <RefreshCw className="h-4 w-4" />
              Retry
            </Button>
          )}
        </div>

        {/* Checkpoints Info */}
        {checkpoints.length > 0 && (
          <div className="text-xs text-muted-foreground flex items-center gap-1 pt-2 border-t">
            <Clock className="h-3 w-3" />
            {checkpoints.length} checkpoint(s) saved • Last: {new Date(checkpoints[checkpoints.length - 1].timestamp).toLocaleString()}
          </div>
        )}
      </CardContent>
    </Card>
  );
}