#!/usr/bin/env python
"""
Main entry point for the ARES Voice Verification System.
"""

import os
import sys
import argparse
import logging

from ares.core.engine import ARESEngine
from ares.gui.main_window import ARESApp


def setup_logging():
    """Set up logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ARES Voice Verification System'
    )
    parser.add_argument(
        '-i', '--input', 
        help='Path to input audio file for analysis'
    )
    parser.add_argument(
        '-g', '--gui',
        action='store_true',
        help='Launch the GUI interface'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def main():
    """Main entry point for ARES."""
    args = parse_arguments()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Initialize the ARES engine
    engine = ARESEngine()
    
    if args.gui:
        # Launch the GUI application
        app = ARESApp(engine)
        app.run()
    elif args.input:
        # Process the input file and display results
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.")
            sys.exit(1)
            
        result = engine.analyze_file(args.input)
        
        # Print the results
        print("\n=== ARES Voice Verification Results ===")
        print(f"Classification: {result['classification']}")
        print(f"Confidence Score: {result['confidence_score']:.2f}%")
        print("\nEvidence Summary:")
        for key, value in result['evidence'].items():
            print(f"  - {key}: {value}")
    else:
        # No parameters provided, launch GUI by default
        app = ARESApp(engine)
        app.run()


if __name__ == "__main__":
    main() 