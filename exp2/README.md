# AI Parsing Accuracy Test: TOON vs JSON

This experiment tests AI parsing accuracy and efficiency between JSON and TOON (Token-Oriented Object Notation) formats using Google's Gemini LLM.

## Setup

1. **Create `.env` file:**
   ```bash
   echo 'GOOGLE_API_KEY=your_key_here' > .env
   ```

2. **Install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip3 install -r requirements.txt
   ```

3. **Ensure Node.js is installed** (for TOON CLI converter)

## Usage

```bash
python3 script.py <json_file> [question]
```

**Examples:**
```bash
# Interactive mode (prompts for question)
python3 script.py data.json

# With specific question
python3 script.py data.json "Find all users with age > 24"

# Complex nested query
python3 script.py data.json "Find the department with the most active projects and list team leads"
```

## How It Works

1. Takes a JSON file as input
2. Converts to TOON format using `@toon-format/cli`
3. Sends **separate requests** to Gemini (with delay to prevent memory contamination)
4. Compares responses and performance
5. Saves results to `test_results.json`

## Output

The script displays:
- ✅ Response accuracy comparison
- ⏱️ Performance metrics (response times)
- 📊 Side-by-side response analysis
- 💾 Saved JSON results for further analysis
