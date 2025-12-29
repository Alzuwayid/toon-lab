#!/usr/bin/env python3
"""
Token-Oriented Object Notation (TOON) vs JSON - AI Parsing Accuracy Test

This script tests the parsing accuracy of LLMs when processing data in JSON format
versus TOON format. It sends identical queries to Google's Gemini API in separate
requests to avoid memory contamination.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file")
    sys.exit(1)

client = genai.Client(api_key=GOOGLE_API_KEY)

# Try to list available models for debugging
try:
    print("🔍 Checking available models...")
    models = list(client.models.list())
    print(f"✅ Found {len(models)} available models")
    
    # Find gemini models that support generateContent
    gemini_models = []
    for model in models:
        if hasattr(model, 'name') and 'gemini' in model.name.lower():
            methods = getattr(model, 'supported_generation_methods', [])
            if 'generateContent' in methods or not methods:
                gemini_models.append(model.name)
    
    if gemini_models:
        print(f"📋 Available Gemini models: {', '.join(gemini_models[:5])}")
        AVAILABLE_MODEL = gemini_models[0]
    else:
        # Fallback
        AVAILABLE_MODEL = models[0].name if models else 'gemini-pro'
        print(f"⚠️  Using fallback model: {AVAILABLE_MODEL}")
except Exception as e:
    print(f"⚠️  Could not list models: {e}")
    AVAILABLE_MODEL = 'gemini-pro'


class TOONParsingTest:
    """Test AI parsing accuracy between JSON and TOON formats"""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = Path(json_file_path)
        self.toon_file_path = None
        self.json_content = None
        self.toon_content = None
        
        # Validate JSON file
        if not self.json_file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        if not self.json_file_path.suffix == '.json':
            raise ValueError("File must have .json extension")
    
    def convert_to_toon(self) -> Path:
        """Convert JSON file to TOON format using the TOON CLI"""
        print(f"\n📝 Converting {self.json_file_path.name} to TOON format...")
        
        # Generate output path
        output_name = self.json_file_path.stem + "_output.toon"
        self.toon_file_path = self.json_file_path.parent / output_name
        
        try:
            # Run the TOON CLI converter
            result = subprocess.run(
                ['npx', '@toon-format/cli', str(self.json_file_path), '-o', str(self.toon_file_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✅ Successfully converted to: {self.toon_file_path.name}")
            return self.toon_file_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting to TOON format:")
            print(f"   {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ Error: 'npx' command not found. Make sure Node.js and npm are installed.")
            sys.exit(1)
    
    def load_contents(self):
        """Load both JSON and TOON file contents"""
        print("\n📖 Loading file contents...")
        
        # Load JSON
        with open(self.json_file_path, 'r') as f:
            self.json_content = f.read()
        
        # Load TOON
        with open(self.toon_file_path, 'r') as f:
            self.toon_content = f.read()
        
        print(f"   JSON size: {len(self.json_content)} characters")
        print(f"   TOON size: {len(self.toon_content)} characters")
    
    def query_llm(self, data_content: str, data_format: str, question: str) -> Tuple[str, float]:
        """
        Query the LLM with data and a question
        
        Returns: (response_text, response_time)
        """
        prompt = f"""You are a data analyst. I will provide you with data in {data_format} format.

DATA:
{data_content}

TASK: {question}

Please analyze the data and provide a precise answer. Be specific and include all relevant details."""

        try:
            start_time = time.time()
            response = client.models.generate_content(
                model=AVAILABLE_MODEL,
                contents=prompt
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            return response.text.strip(), response_time
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    
    def run_test(self, question: str) -> Dict[str, Any]:
        """
        Run the parsing test with a specific question
        
        Returns a dictionary with test results
        """
        print(f"\n{'='*80}")
        print(f"🔬 TESTING QUESTION: {question}")
        print(f"{'='*80}")
        
        # Test with JSON format
        print("\n[1/2] Querying with JSON format...")
        json_response, json_time = self.query_llm(self.json_content, "JSON", question)
        print(f"   ⏱️  Response time: {json_time:.2f}s")
        
        # Wait a bit to ensure separate requests
        time.sleep(2)
        
        # Test with TOON format
        print("\n[2/2] Querying with TOON format...")
        toon_response, toon_time = self.query_llm(self.toon_content, "TOON", question)
        print(f"   ⏱️  Response time: {toon_time:.2f}s")
        
        return {
            'question': question,
            'json_response': json_response,
            'json_time': json_time,
            'toon_response': toon_response,
            'toon_time': toon_time
        }
    
    def display_results(self, results: Dict[str, Any]):
        """Display test results in a readable format"""
        print(f"\n{'='*80}")
        print("📊 TEST RESULTS")
        print(f"{'='*80}")
        
        print(f"\n❓ QUESTION:")
        print(f"   {results['question']}")
        
        print(f"\n📄 JSON RESPONSE (⏱️  {results['json_time']:.2f}s):")
        print("   " + "-" * 76)
        for line in results['json_response'].split('\n'):
            print(f"   {line}")
        
        print(f"\n📝 TOON RESPONSE (⏱️  {results['toon_time']:.2f}s):")
        print("   " + "-" * 76)
        for line in results['toon_response'].split('\n'):
            print(f"   {line}")
        
        print(f"\n{'='*80}")
        print("⚖️  COMPARISON")
        print(f"{'='*80}")
        
        # Check if responses match (basic comparison)
        if results['json_response'].lower() == results['toon_response'].lower():
            print("✅ Responses are IDENTICAL")
        else:
            print("⚠️  Responses DIFFER")
            print("\n   Manual review recommended to determine accuracy.")
        
        # Performance comparison
        time_diff = results['json_time'] - results['toon_time']
        if abs(time_diff) < 0.5:
            print(f"⏱️  Performance: SIMILAR (~{abs(time_diff):.2f}s difference)")
        elif time_diff > 0:
            print(f"⏱️  Performance: TOON was FASTER by {time_diff:.2f}s")
        else:
            print(f"⏱️  Performance: JSON was FASTER by {abs(time_diff):.2f}s")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, results: Dict[str, Any], output_file: str = "test_results.json"):
        """Save test results to a JSON file"""
        output_path = self.json_file_path.parent / output_file
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def print_banner():
    """Print welcome banner"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   TOON vs JSON - AI Parsing Accuracy Test                                ║
║   Testing LLM parsing efficiency with different data formats             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)


def get_sample_questions():
    """Return sample questions for testing"""
    return [
        "Extract all entries where age is greater than 24 and list their names.",
        "Find the department with the most active projects and list the team leads.",
        "What is the total count of active projects across all departments?",
        "List all unique job titles in the dataset.",
        "Find the person with the highest age and provide all their details."
    ]


def main():
    print_banner()
    
    # Check for command line argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_json_file> [question]")
        print("\nExample:")
        print("   python script.py data.json")
        print('   python script.py data.json "Find all users with age > 24"')
        print("\nIf no question is provided, you'll be prompted to enter one.")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # Get question from arguments or prompt user
    if len(sys.argv) >= 3:
        question = ' '.join(sys.argv[2:])
    else:
        print("\n📋 Sample Questions:")
        for i, q in enumerate(get_sample_questions(), 1):
            print(f"   {i}. {q}")
        
        print("\n" + "="*80)
        question = input("\n🎯 Enter your question for the LLM: ").strip()
        
        if not question:
            print("❌ No question provided. Exiting.")
            sys.exit(1)
    
    # Initialize test
    try:
        test = TOONParsingTest(json_file)
        
        # Convert to TOON
        test.convert_to_toon()
        
        # Load contents
        test.load_contents()
        
        # Run test
        results = test.run_test(question)
        
        # Display results
        test.display_results(results)
        
        # Save results
        test.save_results(results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

