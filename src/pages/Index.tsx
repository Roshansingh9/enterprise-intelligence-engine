import { useState } from "react";
import { Download, FileCode, Database, Brain, CheckCircle2, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const pythonFiles = [
  { name: "main.py", path: "support_ai_system/main.py", size: "8.2 KB" },
  { name: "config.yaml", path: "support_ai_system/config.yaml", size: "5.1 KB" },
  { name: "requirements.txt", path: "support_ai_system/requirements.txt", size: "1.2 KB" },
  { name: "README.md", path: "support_ai_system/README.md", size: "6.8 KB" },
];

const modules = [
  { name: "Ingestion", desc: "Excel parsing & data validation", icon: Database },
  { name: "Storage", desc: "SQLite database with 12 tables", icon: Database },
  { name: "Retrieval", desc: "FAISS + BM25 hybrid search", icon: FileCode },
  { name: "Agents", desc: "6 AI agents (Extractor, Generator, QA...)", icon: Brain },
  { name: "Learning", desc: "Continuous self-learning system", icon: Brain },
  { name: "Evaluation", desc: "Hit@K, MRR, coverage metrics", icon: CheckCircle2 },
];

const Index = () => {
  const [downloading, setDownloading] = useState(false);

  const handleDownload = async () => {
    setDownloading(true);
    // In a real implementation, this would zip and download the files
    setTimeout(() => {
      setDownloading(false);
      alert("Files are available in the support_ai_system/ directory. Copy to your local machine to run with Python.");
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 bg-emerald-500/10 text-emerald-400 px-4 py-2 rounded-full text-sm mb-4">
            <Brain className="w-4 h-4" />
            Enterprise AI System
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
            SupportMind AI
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            Self-Learning Support Knowledge Engine with Local LLMs
          </p>
        </div>

        {/* Architecture Diagram */}
        <Card className="bg-slate-800/50 border-slate-700 mb-8">
          <CardHeader>
            <CardTitle className="text-white">System Architecture</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-emerald-400 text-sm overflow-x-auto">
{`Excel → Ingestion → SQLite → Embeddings → Hybrid Retrieval
                           ↓
                     AI Orchestrator (6 Agents)
                           ↓
                 Learning + Evaluation Loop
                           ↓
                    Version Store + Logs`}
            </pre>
          </CardContent>
        </Card>

        {/* Modules Grid */}
        <div className="grid md:grid-cols-3 gap-4 mb-8">
          {modules.map((mod) => (
            <Card key={mod.name} className="bg-slate-800/50 border-slate-700">
              <CardContent className="pt-6">
                <mod.icon className="w-8 h-8 text-emerald-400 mb-3" />
                <h3 className="font-semibold text-white">{mod.name}</h3>
                <p className="text-sm text-slate-400">{mod.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Files List */}
        <Card className="bg-slate-800/50 border-slate-700 mb-8">
          <CardHeader>
            <CardTitle className="text-white">Core Files</CardTitle>
            <CardDescription>Python codebase ready for local execution</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {pythonFiles.map((file) => (
                <div key={file.name} className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <FileCode className="w-5 h-5 text-emerald-400" />
                    <span className="font-mono text-sm">{file.name}</span>
                  </div>
                  <span className="text-xs text-slate-500">{file.size}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Quick Start */}
        <Card className="bg-emerald-900/20 border-emerald-700/50 mb-8">
          <CardHeader>
            <CardTitle className="text-emerald-400">Quick Start</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-sm text-slate-300 bg-slate-900 p-4 rounded-lg overflow-x-auto">
{`# Install Ollama and pull model
ollama pull llama3

# Setup Python environment
cd support_ai_system
pip install -r requirements.txt

# Initialize and run
python main.py --init
python main.py --train`}
            </pre>
          </CardContent>
        </Card>

        {/* Download Button */}
        <div className="text-center">
          <Button 
            size="lg" 
            onClick={handleDownload}
            disabled={downloading}
            className="bg-emerald-600 hover:bg-emerald-700 text-white px-8 py-6 text-lg"
          >
            <Download className="w-5 h-5 mr-2" />
            {downloading ? "Preparing..." : "View Python Codebase"}
          </Button>
          <p className="text-sm text-slate-500 mt-4">
            Files are in <code className="text-emerald-400">support_ai_system/</code> directory
          </p>
        </div>
      </div>
    </div>
  );
};

export default Index;
