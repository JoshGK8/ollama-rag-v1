#!/usr/bin/env python3
"""Interactive CLI for the RAG system."""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from config import load_config
from rag_engine import RAGEngine


class RAGCLI:
    """Interactive CLI for the RAG system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.rag_engine = RAGEngine(self.config)
        self.console = Console() if RICH_AVAILABLE else None
        self.history: List[Dict[str, str]] = []
        
        # Setup logging
        logging.basicConfig(level=logging.WARNING)
    
    def print_rich(self, content, **kwargs):
        """Print with rich formatting if available, otherwise plain text."""
        if self.console:
            self.console.print(content, **kwargs)
        else:
            if hasattr(content, 'plain'):
                print(content.plain)
            else:
                print(str(content))
    
    def input_rich(self, prompt: str) -> str:
        """Get input with rich formatting if available."""
        if RICH_AVAILABLE:
            return Prompt.ask(prompt)
        else:
            return input(f"{prompt}: ")
    
    def display_welcome(self):
        """Display welcome message and system status."""
        company = self.config.data.company_name
        
        if self.console:
            welcome_text = f"""
# Welcome to {company} RAG System 🤖

Ask questions about {company} documentation and get instant answers!

Type 'help' for commands, 'examples' for sample questions, or 'quit' to exit.
            """
            self.print_rich(Panel(Markdown(welcome_text), title="🚀 RAG Assistant", expand=False))
        else:
            print(f"\n=== Welcome to {company} RAG System ===")
            print(f"Ask questions about {company} documentation and get instant answers!")
            print("Type 'help' for commands, 'examples' for sample questions, or 'quit' to exit.\n")
    
    def check_system_status(self) -> bool:
        """Check and display system status."""
        status = self.rag_engine.get_system_status()
        
        if not status['ollama_connected']:
            self.print_rich("[red]❌ Ollama is not running or not accessible![/red]")
            self.print_rich("Please start Ollama first: [blue]ollama serve[/blue]")
            return False
        
        if not status['embedding_model_available']:
            self.print_rich(f"[red]❌ Embedding model '{self.config.ollama.embedding_model}' not found![/red]")
            self.print_rich(f"Run: [blue]ollama pull {self.config.ollama.embedding_model}[/blue]")
            return False
        
        if not status['chat_model_available']:
            self.print_rich(f"[red]❌ Chat model '{self.config.ollama.chat_model}' not found![/red]")
            self.print_rich(f"Run: [blue]ollama pull {self.config.ollama.chat_model}[/blue]")
            return False
        
        kb_info = status.get('knowledge_base', {})
        doc_count = kb_info.get('document_count', 0)
        
        if doc_count == 0:
            self.print_rich("[yellow]⚠️  Knowledge base is empty![/yellow]")
            self.print_rich("Run: [blue]python src/process_documents.py[/blue]")
            return False
        
        if self.console:
            status_table = Table(title="System Status")
            status_table.add_column("Component", style="cyan")
            status_table.add_column("Status", style="green")
            status_table.add_column("Details")
            
            status_table.add_row("Ollama", "✅ Connected", status['config']['embedding_model'])
            status_table.add_row("Knowledge Base", "✅ Ready", f"{doc_count} documents")
            status_table.add_row("Company", "✅ Loaded", status['config']['company_name'])
            
            self.print_rich(status_table)
        else:
            print(f"✅ System ready: {doc_count} documents loaded for {status['config']['company_name']}")
        
        return True
    
    def display_help(self):
        """Display help information."""
        if self.console:
            help_text = """
## Available Commands

- **help** - Show this help message
- **status** - Show system status
- **examples** - Show example questions
- **history** - Show recent questions
- **clear** - Clear conversation history
- **quit** / **exit** - Exit the program

## Tips

- Ask specific questions about the documentation
- Use natural language - no special syntax required
- Questions are searched across all loaded documents
- Sources are shown with each answer for verification
            """
            self.print_rich(Panel(Markdown(help_text), title="📖 Help", expand=False))
        else:
            print("\n=== Available Commands ===")
            print("help      - Show this help message")
            print("status    - Show system status")
            print("examples  - Show example questions")
            print("history   - Show recent questions")
            print("clear     - Clear conversation history")
            print("quit/exit - Exit the program")
            print("\nTips: Ask specific questions about the documentation using natural language.")
    
    def display_examples(self):
        """Display example questions."""
        examples = self.rag_engine.suggest_questions()
        
        if self.console:
            self.print_rich(f"\n[bold cyan]Example questions for {self.config.data.company_name}:[/bold cyan]")
            for i, example in enumerate(examples, 1):
                self.print_rich(f"{i}. [blue]{example}[/blue]")
        else:
            print(f"\nExample questions for {self.config.data.company_name}:")
            for i, example in enumerate(examples, 1):
                print(f"{i}. {example}")
        print()
    
    def display_history(self):
        """Display conversation history."""
        if not self.history:
            self.print_rich("[yellow]No conversation history yet.[/yellow]")
            return
        
        if self.console:
            history_table = Table(title="Recent Questions")
            history_table.add_column("#", width=3)
            history_table.add_column("Question", style="cyan")
            history_table.add_column("Confidence", width=10)
            
            for i, item in enumerate(self.history[-self.config.cli.history_size:], 1):
                confidence = item.get('confidence', 0)
                conf_color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
                history_table.add_row(
                    str(i), 
                    item['question'], 
                    f"[{conf_color}]{confidence:.1%}[/{conf_color}]"
                )
            
            self.print_rich(history_table)
        else:
            print("\n=== Recent Questions ===")
            for i, item in enumerate(self.history[-self.config.cli.history_size:], 1):
                confidence = item.get('confidence', 0)
                print(f"{i}. {item['question']} (confidence: {confidence:.1%})")
    
    def display_answer(self, result: Dict[str, Any]):
        """Display the answer and sources."""
        answer = result['answer']
        sources = result.get('sources', [])
        confidence = result.get('confidence', 0)
        
        if self.console:
            # Display answer
            conf_color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
            answer_panel = Panel(
                Markdown(answer),
                title=f"[bold]Answer[/bold] [dim]({confidence:.1%} confidence)[/dim]",
                title_align="left",
                border_style=conf_color
            )
            self.print_rich(answer_panel)
            
            # Display sources if available
            if sources and self.config.cli.show_sources:
                sources_table = Table(title="📚 Sources")
                sources_table.add_column("Document", style="cyan")
                sources_table.add_column("Similarity", width=10)
                sources_table.add_column("Section")
                
                for source in sources:
                    sim_color = "green" if source['similarity'] > 0.7 else "yellow"
                    sources_table.add_row(
                        source['title'],
                        f"[{sim_color}]{source['similarity']:.1%}[/{sim_color}]",
                        source['chunk_info']
                    )
                
                self.print_rich(sources_table)
        else:
            print(f"\nAnswer (confidence: {confidence:.1%}):")
            print("=" * 50)
            print(answer)
            
            if sources and self.config.cli.show_sources:
                print(f"\nSources:")
                for i, source in enumerate(sources, 1):
                    print(f"{i}. {source['title']} ({source['similarity']:.1%} similarity)")
        
        print()
    
    def process_question(self, question: str):
        """Process a user question and display the answer."""
        self.print_rich("[dim]Searching knowledge base...[/dim]")
        
        result = self.rag_engine.query(question, include_sources=self.config.cli.show_sources)
        
        # Add to history
        self.history.append({
            'question': question,
            'answer': result['answer'],
            'confidence': result['confidence']
        })
        
        # Display result
        self.display_answer(result)
    
    def run(self):
        """Run the interactive CLI."""
        self.display_welcome()
        
        # Check system status
        if not self.check_system_status():
            return
        
        print()
        
        while True:
            try:
                question = self.input_rich("[bold green]Ask a question[/bold green]").strip()
                
                if not question:
                    continue
                
                question_lower = question.lower()
                
                # Handle commands
                if question_lower in ['quit', 'exit', 'q']:
                    self.print_rich("[bold blue]Thank you for using the RAG system! 👋[/bold blue]")
                    break
                elif question_lower == 'help':
                    self.display_help()
                elif question_lower == 'status':
                    self.check_system_status()
                elif question_lower == 'examples':
                    self.display_examples()
                elif question_lower == 'history':
                    self.display_history()
                elif question_lower == 'clear':
                    self.history.clear()
                    self.print_rich("[green]History cleared.[/green]")
                else:
                    # Process the question
                    self.process_question(question)
                
            except KeyboardInterrupt:
                self.print_rich("\n[bold blue]Goodbye! 👋[/bold blue]")
                break
            except Exception as e:
                self.print_rich(f"[red]Error: {e}[/red]")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive RAG CLI')
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        cli = RAGCLI(args.config)
        cli.run()
    except FileNotFoundError:
        print("Configuration file not found. Please copy config.example.yaml to config.yaml and customize it.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to start CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()