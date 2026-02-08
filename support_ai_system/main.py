#!/usr/bin/env python3
"""
Support AI System - Main Entry Point
=====================================
Self-Learning Support Knowledge Engine

A fault-tolerant, resumable, self-learning AI knowledge platform
using local LLMs for enterprise customer support.

Usage:
    python main.py --init       Initialize the system
    python main.py --baseline   Run baseline evaluation
    python main.py --train      Start training pipeline
    python main.py --resume     Resume from checkpoint
    python main.py --evaluate   Run full evaluation
    python main.py --serve      Start the system server

Author: SupportMind AI Team
Version: 1.0.0
"""

import os
import sys
import click
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.orchestration.orchestrator import SystemOrchestrator
from src.orchestration.progress_tracker import ProgressTracker
from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.utils.directory_setup import ensure_directories


# =============================================================================
# CLI COMMANDS
# =============================================================================

@click.group()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config: str, verbose: bool):
    """
    SupportMind AI - Self-Learning Support Knowledge Engine
    
    Enterprise-grade AI system for customer support intelligence.
    """
    ctx.ensure_object(dict)
    
    # Load configuration
    ctx.obj['config'] = load_config(config)
    ctx.obj['verbose'] = verbose
    
    # Setup logging
    log_level = 'DEBUG' if verbose else ctx.obj['config']['system']['log_level']
    setup_logging(ctx.obj['config'], log_level)
    
    # Ensure directory structure
    ensure_directories(ctx.obj['config'])


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the system - creates database, indexes, and required structures."""
    config = ctx.obj['config']
    
    click.echo("=" * 60)
    click.echo("🚀 INITIALIZING SUPPORTMIND AI SYSTEM")
    click.echo("=" * 60)
    
    orchestrator = SystemOrchestrator(config)
    
    try:
        # Initialize all components
        click.echo("\n📁 Creating directory structure...")
        ensure_directories(config)
        
        click.echo("🗄️  Initializing database...")
        orchestrator.init_database()
        
        click.echo("📊 Loading Excel data...")
        orchestrator.ingest_data()
        
        click.echo("🔍 Building embedding indexes...")
        orchestrator.build_indexes()
        
        click.echo("📝 Setting up prompt templates...")
        orchestrator.init_prompts()
        
        click.echo("\n" + "=" * 60)
        click.echo("✅ INITIALIZATION COMPLETE")
        click.echo("=" * 60)
        click.echo("\nNext steps:")
        click.echo("  1. Run baseline: python main.py baseline")
        click.echo("  2. Start training: python main.py train")
        
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        click.echo(f"\n❌ Initialization failed: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def baseline(ctx):
    """Run baseline evaluation without training."""
    config = ctx.obj['config']
    
    click.echo("=" * 60)
    click.echo("📊 RUNNING BASELINE EVALUATION")
    click.echo("=" * 60)
    
    orchestrator = SystemOrchestrator(config)
    progress = ProgressTracker(config)
    
    try:
        orchestrator.load_state()
        
        click.echo("\n🔍 Evaluating retrieval performance...")
        retrieval_metrics = orchestrator.evaluate_retrieval()
        
        click.echo("📝 Evaluating KB quality...")
        quality_metrics = orchestrator.evaluate_kb_quality()
        
        click.echo("✅ Computing QA scores...")
        qa_metrics = orchestrator.evaluate_qa()
        
        # Display results
        click.echo("\n" + "=" * 60)
        click.echo("📈 BASELINE RESULTS")
        click.echo("=" * 60)
        
        click.echo(f"\nRetrieval Metrics:")
        click.echo(f"  Hit@1:  {retrieval_metrics.get('hit_at_1', 0):.3f}")
        click.echo(f"  Hit@3:  {retrieval_metrics.get('hit_at_3', 0):.3f}")
        click.echo(f"  MRR:    {retrieval_metrics.get('mrr', 0):.3f}")
        
        click.echo(f"\nKB Quality:")
        click.echo(f"  Avg Cosine:     {quality_metrics.get('avg_cosine', 0):.3f}")
        click.echo(f"  Structural:     {quality_metrics.get('structural_score', 0):.3f}")
        click.echo(f"  LLM Judge:      {quality_metrics.get('llm_judge_score', 0):.3f}")
        
        click.echo(f"\nQA Metrics:")
        click.echo(f"  Mean Score:     {qa_metrics.get('mean_score', 0):.3f}")
        click.echo(f"  Violations:     {qa_metrics.get('violations', 0)}")
        
        # Save baseline
        orchestrator.save_baseline_metrics({
            'retrieval': retrieval_metrics,
            'quality': quality_metrics,
            'qa': qa_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        click.echo("\n✅ Baseline saved to metrics/baseline.json")
        
    except Exception as e:
        logging.error(f"Baseline evaluation failed: {e}")
        click.echo(f"\n❌ Baseline failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--rounds', '-r', default=None, type=int, help='Number of training rounds')
@click.pass_context
def train(ctx, rounds: int):
    """Start the batch training pipeline."""
    config = ctx.obj['config']
    
    if rounds:
        config['learning']['num_rounds'] = rounds
    
    click.echo("=" * 60)
    click.echo("🧠 STARTING TRAINING PIPELINE")
    click.echo("=" * 60)
    
    orchestrator = SystemOrchestrator(config)
    progress = ProgressTracker(config)
    
    try:
        orchestrator.load_state()
        
        num_rounds = config['learning']['num_rounds']
        click.echo(f"\n📚 Training for {num_rounds} rounds")
        click.echo(f"   Train ratio: {config['learning']['train_ratio']}")
        click.echo(f"   Buffer size: {config['learning']['buffer_size']}")
        
        # Run training loop
        asyncio.run(orchestrator.run_training_pipeline(
            num_rounds=num_rounds,
            progress_callback=progress.update
        ))
        
        click.echo("\n" + "=" * 60)
        click.echo("✅ TRAINING COMPLETE")
        click.echo("=" * 60)
        
        # Show final metrics
        final_metrics = orchestrator.get_final_metrics()
        progress.print_final_report(final_metrics)
        
    except KeyboardInterrupt:
        click.echo("\n⚠️  Training interrupted - saving checkpoint...")
        orchestrator.save_checkpoint()
        click.echo("💾 Checkpoint saved. Resume with: python main.py resume")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        orchestrator.save_checkpoint()
        click.echo(f"\n❌ Training failed: {e}")
        click.echo("💾 Checkpoint saved. Resume with: python main.py resume")
        sys.exit(1)


@cli.command()
@click.pass_context
def resume(ctx):
    """Resume training from the last checkpoint."""
    config = ctx.obj['config']
    
    click.echo("=" * 60)
    click.echo("🔄 RESUMING FROM CHECKPOINT")
    click.echo("=" * 60)
    
    orchestrator = SystemOrchestrator(config)
    progress = ProgressTracker(config)
    
    try:
        # Load checkpoint
        checkpoint = orchestrator.load_checkpoint()
        
        if not checkpoint:
            click.echo("❌ No checkpoint found. Run --init first.")
            sys.exit(1)
        
        click.echo(f"\n📍 Resuming from:")
        click.echo(f"   Batch ID:        {checkpoint.get('batch_id', 'N/A')}")
        click.echo(f"   Round:           {checkpoint.get('round', 0) + 1}")
        click.echo(f"   Progress:        {checkpoint.get('progress_pct', 0):.1f}%")
        click.echo(f"   Saved at:        {checkpoint.get('timestamp', 'Unknown')}")
        
        # Resume training
        asyncio.run(orchestrator.resume_training(
            checkpoint=checkpoint,
            progress_callback=progress.update
        ))
        
        click.echo("\n" + "=" * 60)
        click.echo("✅ TRAINING RESUMED AND COMPLETED")
        click.echo("=" * 60)
        
    except Exception as e:
        logging.error(f"Resume failed: {e}")
        orchestrator.save_checkpoint()
        click.echo(f"\n❌ Resume failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--full', is_flag=True, help='Run full evaluation suite')
@click.pass_context
def evaluate(ctx, full: bool):
    """Run comprehensive evaluation."""
    config = ctx.obj['config']
    
    click.echo("=" * 60)
    click.echo("📊 RUNNING EVALUATION")
    click.echo("=" * 60)
    
    orchestrator = SystemOrchestrator(config)
    
    try:
        orchestrator.load_state()
        
        results = orchestrator.run_full_evaluation(full=full)
        
        # Display comprehensive results
        click.echo("\n" + "=" * 60)
        click.echo("📈 EVALUATION RESULTS")
        click.echo("=" * 60)
        
        click.echo("\n🔍 RETRIEVAL METRICS")
        click.echo("-" * 40)
        for k, v in results['retrieval'].items():
            click.echo(f"  {k}: {v:.4f}")
        
        click.echo("\n📝 COVERAGE METRICS")
        click.echo("-" * 40)
        click.echo(f"  Questions answered: {results['coverage']['answered']}")
        click.echo(f"  Total questions:    {results['coverage']['total']}")
        click.echo(f"  Coverage rate:      {results['coverage']['rate']:.2%}")
        
        click.echo("\n✨ KB QUALITY METRICS")
        click.echo("-" * 40)
        for k, v in results['quality'].items():
            click.echo(f"  {k}: {v:.4f}")
        
        click.echo("\n✅ QA METRICS")
        click.echo("-" * 40)
        click.echo(f"  Mean score:    {results['qa']['mean_score']:.3f}")
        click.echo(f"  Min score:     {results['qa']['min_score']:.3f}")
        click.echo(f"  Max score:     {results['qa']['max_score']:.3f}")
        click.echo(f"  Violations:    {results['qa']['violations']}")
        
        # Save results
        orchestrator.save_evaluation_results(results)
        click.echo("\n💾 Results saved to metrics/evaluation_results.json")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        click.echo(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--port', '-p', default=8080, help='Server port')
@click.option('--host', '-h', default='0.0.0.0', help='Server host')
@click.pass_context
def serve(ctx, port: int, host: str):
    """Start the system server for API access."""
    config = ctx.obj['config']
    
    click.echo("=" * 60)
    click.echo("🌐 STARTING SUPPORTMIND AI SERVER")
    click.echo("=" * 60)
    
    try:
        from src.api.server import create_app
        
        app = create_app(config)
        
        click.echo(f"\n🚀 Server running at http://{host}:{port}")
        click.echo("   Press Ctrl+C to stop")
        
        import uvicorn
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        click.echo("❌ Server dependencies not installed.")
        click.echo("   Run: pip install uvicorn fastapi")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Server failed: {e}")
        click.echo(f"\n❌ Server failed: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show current system status."""
    config = ctx.obj['config']
    
    click.echo("=" * 60)
    click.echo("📊 SYSTEM STATUS")
    click.echo("=" * 60)
    
    orchestrator = SystemOrchestrator(config)
    progress = ProgressTracker(config)
    
    try:
        status = orchestrator.get_status()
        
        click.echo(f"\n🔧 System:")
        click.echo(f"   Version:      {config['system']['version']}")
        click.echo(f"   Environment:  {config['system']['environment']}")
        
        click.echo(f"\n🗄️  Database:")
        click.echo(f"   Status:       {'✅ Connected' if status['db_connected'] else '❌ Disconnected'}")
        click.echo(f"   Tables:       {status['table_count']}")
        click.echo(f"   Records:      {status['total_records']}")
        
        click.echo(f"\n🔍 Indexes:")
        click.echo(f"   FAISS:        {'✅ Built' if status['faiss_ready'] else '❌ Not built'}")
        click.echo(f"   BM25:         {'✅ Built' if status['bm25_ready'] else '❌ Not built'}")
        
        click.echo(f"\n🧠 LLM:")
        click.echo(f"   Provider:     {config['llm']['provider']}")
        click.echo(f"   Model:        {config['llm']['ollama']['model']}")
        click.echo(f"   Status:       {'✅ Available' if status['llm_available'] else '❌ Unavailable'}")
        
        click.echo(f"\n📈 Progress:")
        click.echo(f"   Dataset:      {status['progress']['dataset_pct']:.1f}%")
        click.echo(f"   KB Coverage:  {status['progress']['kb_coverage_pct']:.1f}%")
        click.echo(f"   Phase:        {status['progress']['current_phase']}")
        
        if status.get('checkpoint'):
            click.echo(f"\n💾 Last Checkpoint:")
            click.echo(f"   Batch:        {status['checkpoint']['batch_id']}")
            click.echo(f"   Saved:        {status['checkpoint']['timestamp']}")
        
    except Exception as e:
        click.echo(f"\n❌ Status check failed: {e}")
        sys.exit(1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    cli()
