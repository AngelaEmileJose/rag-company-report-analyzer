#!/usr/bin/env python3
"""
RAG Question CLI - Interactive Command Line Interface
"""

import cmd
import shlex
import sys
import logging
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from rag_system import ImprovedRAGSystem, RAGConfig

# Configure logging to be less verbose for CLI
logging.basicConfig(level=logging.WARNING)
console = Console()

class RAGShell(cmd.Cmd):
    intro = 'Welcome to RAG Company Report Analyzer shell. Type help or ? to list commands.\n'
    prompt = '(rag) '

    def __init__(self):
        super().__init__()
        self.config = RAGConfig.from_env()
        self.rag = ImprovedRAGSystem(self.config)
        self.current_company = None
        self.current_topic = None

    def do_analyze(self, arg):
        """
        Analyze a document from URL or file.
        Usage: analyze <url_or_path> [company_name]
        """
        args = shlex.split(arg)
        if not args:
            console.print("[red]Error: Please provide a URL or file path[/red]")
            return

        source = args[0]
        company = args[1] if len(args) > 1 else "Unknown Company"
        
        self.current_company = company
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Processing document...", total=None)
            
            # Using asyncio.run for the async method since Cmd is synchronous
            import asyncio
            result = asyncio.run(self.rag.process_document(source))
            
        if result.success:
            console.print(Panel(f"[green]Successfully processed document[/green]\nSource: {result.source}\nChunks: {result.chunk_count}", title="Success"))
        else:
            console.print(f"[red]Error: {result.message}[/red]")

    def do_ask(self, arg):
        """
        Ask a question about the processed document.
        Usage: ask <question>
        """
        if not arg:
            console.print("[red]Error: Please ask a question[/red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Thinking...", total=None)
            answer = self.rag.ask_question(arg)

        console.print(Panel(Markdown(answer), title=f"Q: {arg}", border_style="blue"))

    def do_generate_questions(self, arg):
        """
        Generate questions about a topic.
        Usage: generate_questions <topic> [count]
        """
        args = shlex.split(arg)
        if not args:
            console.print("[red]Error: Please provide a topic[/red]")
            return
            
        topic = args[0]
        count = int(args[1]) if len(args) > 1 else 5
        self.current_topic = topic
        
        if not self.current_company:
            console.print("[yellow]Warning: Company name not set, using generic queries[/yellow]")
            company = "the company"
        else:
            company = self.current_company

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=f"Generating {count} questions about {topic}...", total=None)
            questions = self.rag.generate_questions(company, topic, count)

        if questions:
            table = Table(title=f"Questions about {topic}")
            table.add_column("No.", style="cyan", no_wrap=True)
            table.add_column("Question", style="magenta")
            
            for i, q in enumerate(questions, 1):
                table.add_row(str(i), q)
                
            console.print(table)
        else:
            console.print("[red]Failed to generate questions[/red]")

    def do_export(self, arg):
        """
        Export Q&A to PDF (must have asked questions or generated them).
        This is a placeholder for the CLI export functionality.
        """
        console.print("[yellow]Export in CLI is simplified. Please use the Web Interface for full report generation.[/yellow]")

    def do_quit(self, arg):
        """Exit the shell"""
        console.print("Goodbye!")
        return True

    def do_exit(self, arg):
        """Exit the shell"""
        return True

if __name__ == '__main__':
    try:
        RAGShell().cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
