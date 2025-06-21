# Legal Information Retrieval System - Spell Correction

A legal information retrieval system that compares Standard Levenshtein and Weighted Edit Distance algorithms for spell correction of legal terms.

## Quick Start

### Prerequisites
- Python 3.6 or higher
- Required packages (install using `pip install -r requirements.txt`)

### Basic Usage
```bash
python IRL_Group_65_Assignment_01_PS07.py
```

## Usage Instructions

### Interactive Mode
Run the application and follow the on-screen prompts:

1. **Start the application**: Execute the Python script to enter interactive mode
2. **Enter legal terms**: Type any legal term you want to check for spelling
3. **View corrections**: The system will display suggestions using both algorithms
4. **Continue testing**: Enter more terms or type 'quit' to exit

### Input Options
- **Single words**: Enter individual legal terms (e.g., "contract", "litigation")
- **Exit command**: Type "quit", "exit", or press Ctrl+C to stop the application

### Output Information
For each input term, the system provides:
- **Original term**: The word you entered
- **Standard Levenshtein suggestions**: Top corrections using standard edit distance
- **Weighted Edit Distance suggestions**: Top corrections using optimized weights
- **Distance scores**: Numerical similarity scores for each suggestion
- **Algorithm comparison**: Side-by-side comparison of both methods

### Example Session
```
Enter a legal term to check (or 'quit' to exit): contarct
Original: contarct

Standard Levenshtein Distance Results:
1. contract (distance: 1)
2. contact (distance: 2)

Weighted Edit Distance Results:
1. contract (distance: 0.8)
2. contact (distance: 1.6)

Enter a legal term to check (or 'quit' to exit): litiagtion
Original: litiagtion

Standard Levenshtein Distance Results:
1. litigation (distance: 1)
2. mitigation (distance: 2)

Weighted Edit Distance Results:
1. litigation (distance: 0.9)
2. mitigation (distance: 1.8)
```

### Sample Legal Terms for Testing
The system works well with common legal terminology:
- Constitutional terms: "constitution", "amendment", "rights"
- Contract law: "contract", "agreement", "breach", "consideration"
- Criminal law: "criminal", "prosecution", "defendant", "verdict"
- Civil procedure: "plaintiff", "litigation", "discovery", "jurisdiction"
- Property law: "property", "ownership", "easement", "mortgage"
- Evidence: "evidence", "testimony", "witness", "hearsay"

### Performance Features
- **Fast processing**: Real-time spell checking with immediate results
- **Comprehensive database**: Covers extensive legal terminology
- **Dual algorithms**: Compare standard and weighted distance methods
- **Error handling**: Graceful handling of invalid inputs and interruptions

### Batch Testing
For testing multiple terms systematically, consider using the Jupyter notebook version (`IRL_Group_65_Assignment_01_PS07.ipynb`) which provides additional analysis and batch processing capabilities.

### Troubleshooting
- **Import errors**: Ensure all required packages are installed
- **Performance issues**: For large-scale testing, use the notebook version
- **Unexpected exits**: Check Python version compatibility (3.6+)

### File Structure
- `IRL_Group_65_Assignment_01_PS07.py` - Main Python application
- `IRL_Group_65_Assignment_01_PS07.ipynb` - Jupyter notebook version with analysis
- `legal_terms.txt` - Legal terminology database
- `requirements.txt` - Required Python packages
- `sample_documents/` - Sample legal documents for reference

### Notes
- The system focuses on legal terminology and may not perform optimally on general vocabulary
- Weighted Edit Distance typically provides more accurate results for legal terms
- The application maintains a persistent session until explicitly terminated
- All corrections are based on the legal terms database included with the system
