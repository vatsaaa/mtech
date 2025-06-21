#!/usr/bin/env python3
"""
Legal Information Retrieval System - Spell Checker Application
Comparative Analysis: Standard vs. Weighted Edit Distance for Legal Term Correction

This application implements a comprehensive comparison between Standard Levenshtein 
Edit Distance and Weighted Edit Distance algorithms for spell correction of legal 
terms in legal information retrieval systems like Westlaw and LexisNexis.
"""

import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class LegalTermDictionary:
    """
    Manages the legal term dictionary for spell correction in legal domain.
    
    This class handles loading, storing, and managing legal terms used for
    spell correction in legal information retrieval systems.
    """
    
    def __init__(self, filepath: str = "legal_terms.txt"):
        """Initialize the legal term dictionary."""
        self.filepath = filepath
        self.terms = self._load_legal_terms()
        self.term_frequency = Counter()
        
    def _load_legal_terms(self) -> Set[str]:
        """Load legal terms from file or use default comprehensive set."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                terms = set(line.strip().lower() for line in f if line.strip())
            print(f"📚 Legal Dictionary initialized with {len(terms)} terms from {self.filepath}")
            return terms
        except FileNotFoundError:
            print(f"⚠️ {self.filepath} not found. Using comprehensive default legal terms.")
            return self._get_default_legal_terms()
    
    def _get_default_legal_terms(self) -> Set[str]:
        """Comprehensive set of 100+ legal terms across various domains."""
        return {
            # Core legal terms
            'plaintiff', 'defendant', 'jurisdiction', 'jurisprudence', 'habeas', 'corpus',
            'affidavit', 'subpoena', 'testimony', 'indictment', 'tort', 'contract',
            'negligence', 'liability', 'litigation', 'brief', 'motion', 'statute',
            'precedent', 'appeal', 'injunction', 'deposition', 'verdict', 'sentence',
            'plea', 'probate', 'hearsay', 'damages', 'contempt', 'bail', 'writ',
            'equity', 'trust', 'trustee', 'executor', 'guardian', 'fiduciary',
            
            # Criminal law terms
            'perjury', 'misdemeanor', 'felony', 'prosecution', 'defense', 'accused',
            'accomplice', 'allegation', 'charge', 'evidence', 'discovery', 'burden',
            'proof', 'restitution', 'arraignment', 'witness', 'jury', 'judge',
            
            # Contract and property law
            'breach', 'consideration', 'offer', 'acceptance', 'capacity', 'duress',
            'fraud', 'coercion', 'parol', 'ambiguity', 'condition', 'novation',
            'assignment', 'indemnity', 'surety', 'mortgage', 'foreclosure', 'lease',
            'tenant', 'landlord', 'easement', 'title', 'possession', 'trespass',
            'nuisance', 'remedy', 'settlement',
            
            # Procedural terms
            'arbitration', 'mediation', 'clause', 'covenant', 'statutory',
            'constitutional', 'binding', 'estoppel', 'lien', 'summons', 'complaint',
            'petition', 'hearing', 'rebuttal', 'cross', 'examination',
            
            # Advanced legal concepts
            'certiorari', 'mandamus', 'amicus', 'curiae', 'res', 'judicata',
            'collateral', 'proximate', 'causation', 'contributory', 'comparative',
            'vicarious', 'respondeat', 'superior', 'force', 'majeure', 'ultra',
            'vires', 'venue', 'forum', 'limitations', 'laches', 'waiver',
            'ratification', 'rescission', 'reformation', 'specific', 'performance',
            'liquidated', 'punitive', 'exemplary', 'nominal', 'incidental',
            'consequential', 'mitigation', 'foreseeability',
            
            # Legal professionals
            'attorney', 'counsel', 'solicitor', 'barrister', 'advocate',
            'prosecutor', 'magistrate', 'bailiff', 'clerk', 'stenographer'
        }
    
    def get_terms(self) -> Set[str]:
        """Get all legal terms."""
        return self.terms
    
    def get_term_count(self) -> int:
        """Get total number of terms."""
        return len(self.terms)


class EditDistanceCalculator:
    """
    Implements both Standard Levenshtein and Weighted Edit Distance algorithms.
    
    This class provides the core functionality for comparing spell correction
    algorithms in the legal domain with detailed operation tracking.
    """
    
    def __init__(self):
        """Initialize with legal domain optimized weights."""
        # Custom weights optimized for legal term corrections
        self.legal_weights = {
            'insertion': 1.0,        # Standard insertion cost
            'deletion': 1.2,         # Slightly higher deletion penalty
            'substitution': 1.5,     # Higher substitution penalty
            'vowel_confusion': 0.8,  # Lower penalty for vowel errors (a/e, i/y)
            'common_legal': 0.5      # Much lower for common legal errors
        }
        
    def standard_levenshtein(self, s1: str, s2: str) -> Tuple[int, List[str]]:
        """
        Calculate Standard Levenshtein distance with operation tracking.
        
        Args:
            s1: Source string (misspelled word)
            s2: Target string (correct legal term)
            
        Returns:
            Tuple of (edit distance, list of operations)
        """
        m, n = len(s1), len(s2)
        
        # DP table for distances
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # Operations tracking
        ops = [[[] for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
            if i > 0:
                ops[i][0] = ops[i-1][0] + [f"Delete '{s1[i-1]}'"]
        
        for j in range(n + 1):
            dp[0][j] = j
            if j > 0:
                ops[0][j] = ops[0][j-1] + [f"Insert '{s2[j-1]}'"]
        
        # Fill DP table with operation tracking
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    ops[i][j] = ops[i-1][j-1]
                else:
                    # Calculate costs for each operation
                    delete_cost = dp[i-1][j] + 1
                    insert_cost = dp[i][j-1] + 1
                    substitute_cost = dp[i-1][j-1] + 1
                    
                    min_cost = min(delete_cost, insert_cost, substitute_cost)
                    dp[i][j] = min_cost
                    
                    # Track which operation was chosen
                    if min_cost == substitute_cost:
                        ops[i][j] = ops[i-1][j-1] + [f"Substitute '{s1[i-1]}' → '{s2[j-1]}'"]
                    elif min_cost == delete_cost:
                        ops[i][j] = ops[i-1][j] + [f"Delete '{s1[i-1]}'"]
                    else:
                        ops[i][j] = ops[i][j-1] + [f"Insert '{s2[j-1]}'"]
        
        return dp[m][n], ops[m][n]
    
    def weighted_edit_distance(self, s1: str, s2: str, weights: Dict[str, float] = None) -> Tuple[float, List[str]]:
        """
        Calculate Weighted Edit Distance with custom operation costs.
        
        Args:
            s1: Source string (misspelled word)
            s2: Target string (correct legal term)
            weights: Custom weights for operations
            
        Returns:
            Tuple of (weighted distance, list of operations with costs)
        """
        if weights is None:
            weights = self.legal_weights
        
        m, n = len(s1), len(s2)
        
        # DP table for weighted distances
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        # Operations tracking with costs
        ops = [[[] for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize base cases with weighted costs
        for i in range(m + 1):
            dp[i][0] = i * weights.get('deletion', 1.0)
            if i > 0:
                del_cost = weights.get('deletion', 1.0)
                ops[i][0] = ops[i-1][0] + [f"Delete '{s1[i-1]}' (cost: {del_cost})"]
        
        for j in range(n + 1):
            dp[0][j] = j * weights.get('insertion', 1.0)
            if j > 0:
                ins_cost = weights.get('insertion', 1.0)
                ops[0][j] = ops[0][j-1] + [f"Insert '{s2[j-1]}' (cost: {ins_cost})"]
        
        # Fill DP table with weighted costs
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    ops[i][j] = ops[i-1][j-1]
                else:
                    # Calculate weighted costs
                    sub_cost = self._get_substitution_cost(s1[i-1], s2[j-1], weights)
                    del_cost = weights.get('deletion', 1.0)
                    ins_cost = weights.get('insertion', 1.0)
                    
                    delete_total = dp[i-1][j] + del_cost
                    insert_total = dp[i][j-1] + ins_cost
                    substitute_total = dp[i-1][j-1] + sub_cost
                    
                    min_cost = min(delete_total, insert_total, substitute_total)
                    dp[i][j] = min_cost
                    
                    # Track operation with cost
                    if min_cost == substitute_total:
                        ops[i][j] = ops[i-1][j-1] + [f"Substitute '{s1[i-1]}' → '{s2[j-1]}' (cost: {sub_cost:.1f})"]
                    elif min_cost == delete_total:
                        ops[i][j] = ops[i-1][j] + [f"Delete '{s1[i-1]}' (cost: {del_cost})"]
                    else:
                        ops[i][j] = ops[i][j-1] + [f"Insert '{s2[j-1]}' (cost: {ins_cost})"]
        
        return dp[m][n], ops[m][n]
    
    def _get_substitution_cost(self, c1: str, c2: str, weights: Dict[str, float]) -> float:
        """Calculate context-aware substitution cost for legal domain."""
        base_cost = weights.get('substitution', 1.0)
        
        # Vowel confusion penalty (common in legal terms)
        vowels = set('aeiou')
        if c1 in vowels and c2 in vowels and c1 != c2:
            return base_cost * weights.get('vowel_confusion', 0.8)
        
        # Common legal character confusions
        legal_confusions = [
            ('c', 'k'), ('s', 'c'), ('i', 'y'), ('ph', 'f'), ('ae', 'e')
        ]
        
        for pair in legal_confusions:
            if (c1, c2) == pair or (c2, c1) == pair:
                return base_cost * weights.get('common_legal', 0.5)
        
        return base_cost


class LegalSpellChecker:
    """
    Main spell checker class that combines legal dictionary with edit distance algorithms
    for legal document spell correction.
    """
    
    def __init__(self, legal_dict: LegalTermDictionary):
        self.legal_dict = legal_dict
        self.calculator = EditDistanceCalculator()
        self.correction_history = []
    
    def is_correct_spelling(self, word: str) -> bool:
        """
        Check if a word is correctly spelled (exists in the legal dictionary).
        
        Args:
            word: The word to check
            
        Returns:
            bool: True if the word exists in the dictionary, False otherwise
        """
        return word.lower() in self.legal_dict.get_terms()
    
    def correct_word(self, word: str, algorithm: str = 'both', max_distance: int = 3) -> Dict[str, Any]:
        """
        Correct a misspelled word using specified algorithm(s).
        
        Args:
            word: The word to correct
            algorithm: 'standard', 'weighted', or 'both'
            max_distance: Maximum edit distance to consider
            
        Returns:
            Dictionary containing correction results
        """
        word = word.lower().strip()
        
        # Check if word is already correct
        if self.is_correct_spelling(word):
            return {
                'input_word': word,
                'is_correct': True,
                'correction': word,
                'distance': 0,
                'confidence': 100.0,
                'algorithm': algorithm
            }
        
        # Get candidates from dictionary
        candidates = []
        for term in self.legal_dict.get_terms():
            if algorithm in ['standard', 'both']:
                std_dist, std_ops = self.calculator.standard_levenshtein(word, term)
                if std_dist <= max_distance:
                    candidates.append((term, std_dist, 'standard'))
            
            if algorithm in ['weighted', 'both']:
                weighted_dist, weighted_ops = self.calculator.weighted_edit_distance(word, term)
                if weighted_dist <= max_distance:
                    candidates.append((term, weighted_dist, 'weighted'))
        
        if not candidates:
            return {
                'input_word': word,
                'is_correct': False,
                'correction': '',
                'distance': float('inf'),
                'confidence': 0.0,
                'algorithm': algorithm
            }
        
        # Find best candidate
        if algorithm == 'standard':
            best_candidate = min([c for c in candidates if c[2] == 'standard'], key=lambda x: x[1])
        elif algorithm == 'weighted':
            best_candidate = min([c for c in candidates if c[2] == 'weighted'], key=lambda x: x[1])
        else:  # both
            best_candidate = min(candidates, key=lambda x: x[1])
        
        # Calculate confidence (inverse of normalized distance)
        max_len = max(len(word), len(best_candidate[0]))
        confidence = max(0, (1 - best_candidate[1] / max_len)) * 100
        
        return {
            'input_word': word,
            'is_correct': False,
            'correction': best_candidate[0],
            'distance': best_candidate[1],
            'confidence': confidence,
            'algorithm': best_candidate[2]
        }
    
    def get_top_suggestions(self, word: str, algorithm: str = 'weighted', top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N suggestions for a misspelled word.
        
        Args:
            word: The misspelled word
            algorithm: 'standard' or 'weighted'
            top_n: Number of suggestions to return
            
        Returns:
            List of (term, distance) tuples sorted by distance
        """
        word = word.lower().strip()
        suggestions = []
        
        for term in self.legal_dict.get_terms():
            if algorithm == 'standard':
                distance, _ = self.calculator.standard_levenshtein(word, term)
            else:
                distance, _ = self.calculator.weighted_edit_distance(word, term)
            
            suggestions.append((term, distance))
        
        # Sort by distance and return top N
        suggestions.sort(key=lambda x: x[1])
        return suggestions[:top_n]
    
    def analyze_correction(self, word: str, max_distance: int = 3) -> Dict[str, Any]:
        """
        Perform comprehensive analysis comparing both algorithms.
        
        Args:
            word: The word to analyze
            max_distance: Maximum edit distance to consider
            
        Returns:
            Detailed analysis dictionary
        """
        word = word.lower().strip()
        
        # Check if already correct
        if self.is_correct_spelling(word):
            return {
                'input_word': word,
                'is_correct': True,
                'message': 'Word is already correctly spelled'
            }
        
        # Get candidates for both algorithms
        std_candidates = []
        weighted_candidates = []
        
        for term in self.legal_dict.get_terms():
            # Standard algorithm
            std_dist, std_ops = self.calculator.standard_levenshtein(word, term)
            if std_dist <= max_distance:
                std_candidates.append((term, std_dist, std_ops))
            
            # Weighted algorithm
            weighted_dist, weighted_ops = self.calculator.weighted_edit_distance(word, term)
            if weighted_dist <= max_distance:
                weighted_candidates.append((term, weighted_dist, weighted_ops))
        
        # Sort candidates
        std_candidates.sort(key=lambda x: x[1])
        weighted_candidates.sort(key=lambda x: x[1])
        
        # Get best results
        std_result = {
            'term': std_candidates[0][0] if std_candidates else '',
            'distance': std_candidates[0][1] if std_candidates else float('inf'),
            'operations': std_candidates[0][2] if std_candidates else []
        }
        
        weighted_result = {
            'term': weighted_candidates[0][0] if weighted_candidates else '',
            'distance': weighted_candidates[0][1] if weighted_candidates else float('inf'),
            'operations': weighted_candidates[0][2] if weighted_candidates else []
        }
        
        # Compare results
        same_suggestion = std_result['term'] == weighted_result['term']
        
        result = {
            'input_word': word,
            'is_correct': False,
            'standard_result': std_result,
            'weighted_result': weighted_result,
            'std_candidates': std_candidates[:5],
            'weighted_candidates': weighted_candidates[:5],
            'analysis': {
                'same_suggestion': same_suggestion,
                'standard_distance': std_result['distance'],
                'weighted_distance': weighted_result['distance'],
                'operations_std': len(std_result['operations']),
                'operations_weighted': len(weighted_result['operations']),
                'improvement': 'weighted' if weighted_result['distance'] < std_result['distance'] else 'standard' if std_result['distance'] < weighted_result['distance'] else 'equal'
            }
        }
        
        self.correction_history.append(result)
        return result
    
    def display_analysis(self, result: Dict[str, Any]) -> None:
        """Display comprehensive analysis of correction results."""
        print(f"\n{'='*80}")
        print(f"🔍 SPELL CORRECTION ANALYSIS: '{result['input_word'].upper()}'")
        print(f"{'='*80}")
        
        if result['is_correct']:
            print("✅ Word is already correct in legal dictionary!")
            return
        
        # Standard Algorithm Results
        print(f"\n📊 STANDARD LEVENSHTEIN EDIT DISTANCE:")
        print(f"{'─'*50}")
        std_result = result['standard_result']
        if std_result['term']:
            print(f"✓ Best Match: {std_result['term']}")
            print(f"✓ Distance: {std_result['distance']}")
            print(f"✓ Operations: {len(std_result['operations'])}")
            if std_result['operations']:
                print("✓ Operation Details:")
                for i, op in enumerate(std_result['operations'], 1):
                    print(f"    {i}. {op}")
        else:
            print("❌ No suitable correction found")
        
        # Weighted Algorithm Results  
        print(f"\n⚖️  WEIGHTED EDIT DISTANCE:")
        print(f"{'─'*50}")
        weighted_result = result['weighted_result']
        if weighted_result['term']:
            print(f"✓ Best Match: {weighted_result['term']}")
            print(f"✓ Distance: {weighted_result['distance']:.2f}")
            print(f"✓ Operations: {len(weighted_result['operations'])}")
            if weighted_result['operations']:
                print("✓ Operation Details:")
                for i, op in enumerate(weighted_result['operations'], 1):
                    print(f"    {i}. {op}")
        else:
            print("❌ No suitable correction found")
        
        # Comparative Analysis
        print(f"\n🔍 COMPARATIVE ANALYSIS:")
        print(f"{'─'*50}")
        analysis = result['analysis']
        
        if analysis['same_suggestion']:
            print("✅ Both algorithms suggest the SAME correction")
            print(f"   Agreed Correction: {std_result['term']}")
        else:
            print("⚠️  Algorithms suggest DIFFERENT corrections:")
            print(f"   Standard: {std_result['term']}")
            print(f"   Weighted: {weighted_result['term']}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Standard Distance: {analysis['standard_distance']}")
        print(f"   Weighted Distance: {analysis['weighted_distance']:.2f}")
        print(f"   Standard Operations: {analysis['operations_std']}")
        print(f"   Weighted Operations: {analysis['operations_weighted']}")
        
        # Determine winner
        if analysis['improvement'] == 'weighted':
            print("🏆 Weighted algorithm found a lower-cost solution")
        elif analysis['improvement'] == 'standard':
            print("🏆 Standard algorithm found a lower-cost solution")
        else:
            print("🤝 Both algorithms achieved the same cost")
        
        # Top candidates
        print(f"\n🏆 TOP CANDIDATES:")
        print(f"{'─'*30}")
        print("Standard Algorithm:")
        for i, (term, dist, _) in enumerate(result['std_candidates'][:3], 1):
            print(f"  {i}. {term:20} (distance: {dist})")
        
        print("\nWeighted Algorithm:")
        for i, (term, dist, _) in enumerate(result['weighted_candidates'][:3], 1):
            print(f"  {i}. {term:20} (distance: {dist:.2f})")


class LegalSpellCheckerApp:
    """
    Main application class that provides command-line interface for the legal spell checker.
    """
    
    def __init__(self, dict_file: Optional[str] = None):
        """Initialize the application."""
        self.legal_dict = LegalTermDictionary(dict_file or "legal_terms.txt")
        self.spell_checker = LegalSpellChecker(self.legal_dict)
        
        # Predefined test cases
        self.test_cases = [
            ("plentiff", "plaintiff"),          # Character substitution error
            ("jurispudence", "jurisprudence"),  # Character deletion
            ("subpena", "subpoena"),            # Missing character
            ("affedavit", "affidavit"),         # Character substitution
            ("neglegence", "negligence"),       # Character rearrangement
            ("contarct", "contract"),           # Character transposition
            ("testimon", "testimony"),          # Character deletion at end
            ("presedent", "precedent")          # Common s/c confusion
        ]
    
    def run_batch_test(self) -> None:
        """Run batch testing on predefined legal term misspellings."""
        print("🧪 COMPREHENSIVE LEGAL SPELL CORRECTION TESTING")
        print("="*60)
        print(f"Testing {len(self.test_cases)} real-world legal term misspellings...")
        print("Using legal domain optimized weights")

        # Track performance metrics
        results = []
        standard_correct = 0
        weighted_correct = 0
        total_tests = len(self.test_cases)

        for i, (misspelled, expected) in enumerate(self.test_cases, 1):
            print(f"\n{'─'*60}")
            print(f"TEST CASE {i}/{total_tests}: '{misspelled}' → expected: '{expected}'")
            
            # Get comprehensive analysis result
            result = self.spell_checker.analyze_correction(misspelled)
            results.append((result, expected))
            
            # Display detailed analysis
            self.spell_checker.display_analysis(result)
            
            # Track accuracy
            if result['standard_result']['term'] == expected:
                standard_correct += 1
                print(f"✅ Standard algorithm: CORRECT")
            else:
                print(f"❌ Standard algorithm: Got '{result['standard_result']['term']}', expected '{expected}'")
            
            if result['weighted_result']['term'] == expected:
                weighted_correct += 1
                print(f"✅ Weighted algorithm: CORRECT")
            else:
                print(f"❌ Weighted algorithm: Got '{result['weighted_result']['term']}', expected '{expected}'")

        # Summary
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Test Cases: {total_tests}")
        print(f"Standard Algorithm Accuracy: {standard_correct}/{total_tests} ({(standard_correct/total_tests)*100:.1f}%)")
        print(f"Weighted Algorithm Accuracy: {weighted_correct}/{total_tests} ({(weighted_correct/total_tests)*100:.1f}%)")

        improvement = ((weighted_correct - standard_correct) / total_tests) * 100
        if improvement > 0:
            print(f"✅ Weighted algorithm shows {improvement:.1f}% improvement over standard")
        elif improvement < 0:
            print(f"⚠️  Standard algorithm performs {abs(improvement):.1f}% better")
        else:
            print("🤝 Both algorithms perform equally")
        
        # Detailed analysis
        self._detailed_performance_analysis(results)
    
    def _detailed_performance_analysis(self, results: List[Tuple[Dict[str, Any], str]]) -> None:
        """Analyze performance differences between algorithms."""
        print("\n🔬 DETAILED ALGORITHM PERFORMANCE ANALYSIS")
        print("="*60)

        # Analyze algorithm agreement and differences
        same_corrections = 0
        different_corrections = 0
        weighted_better = 0
        standard_better = 0
        cost_improvements = []

        print("\n📊 Individual Case Analysis:")
        print(f"{'Misspelled':15} {'Standard':15} {'Weighted':15} {'Agreement':12} {'Better'}")
        print("-" * 75)

        for result, expected in results:
            misspelled = result['input_word']
            std_term = result['standard_result']['term']
            weighted_term = result['weighted_result']['term']
            std_dist = result['standard_result']['distance']
            weighted_dist = result['weighted_result']['distance']
            
            # Check agreement
            agrees = "✅ Yes" if std_term == weighted_term else "❌ No"
            if std_term == weighted_term:
                same_corrections += 1
            else:
                different_corrections += 1
            
            # Determine which is better
            if weighted_dist < std_dist:
                better = "Weighted"
                weighted_better += 1
                cost_improvements.append((std_dist - weighted_dist) / std_dist * 100)
            elif std_dist < weighted_dist:
                better = "Standard"
                standard_better += 1
            else:
                better = "Equal"
            
            print(f"{misspelled:15} {std_term[:14]:15} {weighted_term[:14]:15} {agrees:12} {better}")

        print(f"\n📈 Summary Statistics:")
        print(f"Agreement Rate: {same_corrections}/{len(results)} ({(same_corrections/len(results)*100):.1f}%)")
        print(f"Cases where Weighted performed better: {weighted_better}")
        print(f"Cases where Standard performed better: {standard_better}")

        if cost_improvements:
            avg_improvement = sum(cost_improvements) / len(cost_improvements)
            print(f"Average cost improvement (weighted): {avg_improvement:.1f}%")

        print(f"\n💡 Key Insights:")
        print("• Weighted edit distance advantages:")
        print("  - Better handling of vowel confusions (a/e, i/y)")
        print("  - Lower penalties for common legal character patterns")
        print("  - Domain-specific optimization for legal terminology")
        print("• Standard Levenshtein advantages:")
        print("  - Consistent, predictable behavior across all domains")
        print("  - Simple implementation without domain knowledge")
        print("  - Equal treatment of all character operations")
    
    def interactive_mode(self) -> None:
        """Run interactive spell checking mode."""
        print("🎯 INTERACTIVE LEGAL SPELL CHECKER")
        print("="*40)
        print(f"📖 Dictionary: {self.legal_dict.get_term_count()} legal terms available")
        
        def show_help():
            """Display help information."""
            print("\n📚 HELP - Legal Spell Checker")
            print("=" * 40)
            print("🎯 Purpose: Compare Standard vs Weighted Edit Distance")
            print(f"📖 Dictionary: {self.legal_dict.get_term_count()} legal terms available")
            print("\n🔧 Commands:")
            print("  • 'help' - Show this help")
            print("  • 'samples' - Show sample legal terms")
            print("  • 'quit' or 'exit' - Exit the loop")
            print("Example misspellings to try: 'plentiff', 'jurispudence', 'atorney', 'contarct'")
            print("=" * 40)

        def show_samples():
            """Show sample legal terms from dictionary."""
            print("\n📖 SAMPLE LEGAL TERMS:")
            sample_terms = sorted(list(self.legal_dict.get_terms()))[:20]
            for i, term in enumerate(sample_terms, 1):
                print(f"  {i:2d}. {term}")
            print(f"   ... and {self.legal_dict.get_term_count() - 20} more terms")

        # Interactive loop
        try:
            while True:
                print("\n" + "-" * 40)
                user_input = input("🔍 Enter word to check (or command): ").strip()
                
                if not user_input:
                    print("⚠️  Please enter a word to check")
                    continue
                    
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Exiting interactive mode. Thanks for testing!")
                    break
                    
                elif user_input.lower() == 'help':
                    show_help()
                    continue
                    
                elif user_input.lower() == 'samples':
                    show_samples()
                    continue
                
                # Process the word
                print(f"\n🔍 ANALYZING: '{user_input}'")
                print("=" * 30)
                
                # Get correction result
                result = self.spell_checker.analyze_correction(user_input)
                
                if result['is_correct']:
                    print("✅ Word is already correct in legal dictionary!")
                else:
                    # Show quick comparison
                    std_result = result['standard_result']
                    weighted_result = result['weighted_result']
                    std_term = std_result['term']
                    weighted_term = weighted_result['term']
                    std_dist = std_result['distance']
                    weighted_dist = weighted_result['distance']
                    
                    print(f"📊 QUICK RESULTS:")
                    print(f"   Standard: {user_input} → {std_term} (distance: {std_dist})")
                    print(f"   Weighted: {user_input} → {weighted_term} (distance: {weighted_dist:.2f})")
                    
                    if std_term == weighted_term:
                        print("   🤝 Both algorithms agree!")
                    else:
                        print("   ⚠️  Different corrections suggested")
                    
                    # Ask for detailed analysis
                    detail = input("\n🔍 Show detailed analysis? (y/n): ").strip().lower()
                    if detail in ['y', 'yes', '1']:
                        print("\n" + "=" * 60)
                        self.spell_checker.display_analysis(result)
                
                # Ask to continue
                continue_choice = input("\n➡️  Test another word? (y/n): ").strip().lower()
                if continue_choice in ['n', 'no', '0']:
                    print("👋 Thanks for testing the Legal Spell Checker!")
                    break

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Exiting interactive mode...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Interactive mode ended unexpectedly.")

        print(f"\n✅ Interactive testing completed!")
    
    def single_word_check(self, word: str, detailed: bool = False) -> None:
        """Check a single word for spelling correction."""
        print(f"🔍 CHECKING: '{word}'")
        print("="*30)
        
        result = self.spell_checker.analyze_correction(word)
        
        if result['is_correct']:
            print("✅ Word is already correct in legal dictionary!")
        else:
            if detailed:
                self.spell_checker.display_analysis(result)
            else:
                std_result = result['standard_result']
                weighted_result = result['weighted_result']
                print(f"Standard: {word} → {std_result['term']} (distance: {std_result['distance']})")
                print(f"Weighted: {word} → {weighted_result['term']} (distance: {weighted_result['distance']:.2f})")
    
    def export_results(self, filename: str = "spell_check_results.json") -> None:
        """Export correction history to JSON file."""
        if not self.spell_checker.correction_history:
            print("⚠️ No correction history to export. Run some tests first.")
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.spell_checker.correction_history, f, indent=2, default=str)
            print(f"✅ Results exported to {filename}")
        except Exception as e:
            print(f"❌ Error exporting results: {e}")


def main():
    """Main function to run the Legal Spell Checker application."""
    parser = argparse.ArgumentParser(
        description="Legal Information Retrieval System - Spell Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python legal_spell_checker.py --interactive
  python legal_spell_checker.py --batch-test
  python legal_spell_checker.py --word "plentiff" --detailed
  python legal_spell_checker.py --word "jurispudence"
        """
    )
    
    parser.add_argument(
        '--dict-file', '-d',
        type=str,
        help='Path to legal terms dictionary file (default: legal_terms.txt)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--batch-test', '-b',
        action='store_true',
        help='Run batch testing on predefined misspellings'
    )
    
    parser.add_argument(
        '--word', '-w',
        type=str,
        help='Check spelling of a single word'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed analysis (use with --word)'
    )
    
    parser.add_argument(
        '--export', '-e',
        type=str,
        help='Export results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Initialize application
    print("🏛️ Legal Information Retrieval System - Spell Checker")
    print("=" * 60)
    print("Comparative Analysis: Standard vs. Weighted Edit Distance")
    print("=" * 60)
    
    try:
        app = LegalSpellCheckerApp(args.dict_file)
        
        if args.batch_test:
            app.run_batch_test()
        elif args.interactive:
            app.interactive_mode()
        elif args.word:
            app.single_word_check(args.word, args.detailed)
        else:
            # Default: show help and run interactive mode
            parser.print_help()
            print("\n" + "="*60)
            print("Starting interactive mode...")
            app.interactive_mode()
        
        if args.export:
            app.export_results(args.export)
    
    except Exception as e:
        print(f"❌ Application error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
