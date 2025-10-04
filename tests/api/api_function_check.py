#!/usr/bin/env python3
"""
API Function Check - Comprehensive Review of All ML API Methods

This script checks all public methods in each API class, validates signatures,
and documents any issues or inconsistencies found.

Usage:
    python tests/api/api_function_check.py
"""

import sys
import time
import traceback
import inspect
from pathlib import Path
from typing import Dict, List, Any, Tuple, get_type_hints

# Add ML directory to Python path
ml_dir = Path(__file__).parent.parent.parent / "ML"
sys.path.insert(0, str(ml_dir))

# Import all APIs
try:
    from ML.src.api.training_api import TrainingAPI
    from ML.src.api.prediction_api import PredictionAPI
    from ML.src.api.explanation_api import ExplanationAPI
    from ML.src.api.user_api import ExoplanetMLAPI
except ImportError:
    # Alternative import for when running from tests directory
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ML.src.api.training_api import TrainingAPI
    from ML.src.api.prediction_api import PredictionAPI
    from ML.src.api.explanation_api import ExplanationAPI
    from ML.src.api.user_api import ExoplanetMLAPI

class APIFunctionChecker:
    """Comprehensive checker for all API functions"""
    
    def __init__(self):
        self.apis = {
            'TrainingAPI': TrainingAPI,
            'PredictionAPI': PredictionAPI,
            'ExplanationAPI': ExplanationAPI,
            'ExoplanetMLAPI': ExoplanetMLAPI
        }
        self.issues = []
        
    def check_all_apis(self) -> Dict[str, Any]:
        """Check all API classes and their methods"""
        results = {}
        
        for api_name, api_class in self.apis.items():
            print(f"\n{'='*60}")
            print(f"Checking {api_name}")
            print(f"{'='*60}")
            
            api_results = self.check_api_class(api_name, api_class)
            results[api_name] = api_results
            
        return results
    
    def check_api_class(self, api_name: str, api_class) -> Dict[str, Any]:
        """Check a single API class"""
        results = {
            'class_name': api_name,
            'methods': {},
            'issues': [],
            'stats': {
                'total_methods': 0,
                'public_methods': 0,
                'private_methods': 0,
                'documented_methods': 0,
                'type_annotated_methods': 0
            }
        }
        
        # Get all methods
        methods = inspect.getmembers(api_class, predicate=inspect.isfunction)
        
        for method_name, method_func in methods:
            if method_name.startswith('__'):
                continue  # Skip dunder methods
                
            results['stats']['total_methods'] += 1
            
            if method_name.startswith('_'):
                results['stats']['private_methods'] += 1
            else:
                results['stats']['public_methods'] += 1
            
            method_info = self.check_method(api_name, method_name, method_func)
            results['methods'][method_name] = method_info
            
            # Update stats
            if method_info['has_docstring']:
                results['stats']['documented_methods'] += 1
            if method_info['has_type_annotations']:
                results['stats']['type_annotated_methods'] += 1
                
            # Collect issues
            if method_info['issues']:
                results['issues'].extend(method_info['issues'])
        
        self.print_api_summary(api_name, results)
        return results
    
    def check_method(self, api_name: str, method_name: str, method_func) -> Dict[str, Any]:
        """Check a single method"""
        method_info = {
            'name': method_name,
            'is_public': not method_name.startswith('_'),
            'signature': str(inspect.signature(method_func)),
            'has_docstring': bool(method_func.__doc__),
            'docstring': method_func.__doc__ or "",
            'has_type_annotations': False,
            'parameters': [],
            'return_annotation': None,
            'issues': []
        }
        
        # Check signature and annotations
        sig = inspect.signature(method_func)
        
        # Check parameters
        for param_name, param in sig.parameters.items():
            param_info = {
                'name': param_name,
                'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                'kind': str(param.kind)
            }
            method_info['parameters'].append(param_info)
        
        # Check return annotation
        if sig.return_annotation != inspect.Signature.empty:
            method_info['return_annotation'] = str(sig.return_annotation)
            method_info['has_type_annotations'] = True
        
        # Check if any parameters have type annotations
        if any(p.annotation != inspect.Parameter.empty for p in sig.parameters.values()):
            method_info['has_type_annotations'] = True
        
        # Validate method based on API standards
        self.validate_method(api_name, method_info)
        
        return method_info
    
    def validate_method(self, api_name: str, method_info: Dict[str, Any]):
        """Validate method against API standards"""
        method_name = method_info['name']
        
        # Check public methods have docstrings
        if method_info['is_public'] and not method_info['has_docstring']:
            issue = f"{api_name}.{method_name}: Public method missing docstring"
            method_info['issues'].append(issue)
        
        # Check return type annotation for public methods
        if method_info['is_public'] and not method_info['return_annotation']:
            issue = f"{api_name}.{method_name}: Public method missing return type annotation"
            method_info['issues'].append(issue)
        
        # Check specific parameter patterns
        params = method_info['parameters']
        
        # Skip 'self' parameter
        non_self_params = [p for p in params if p['name'] != 'self']
        
        # Check for session_id parameter in TrainingAPI methods
        if api_name == 'TrainingAPI' and method_info['is_public']:
            if method_name not in ['__init__'] and 'session_id' not in [p['name'] for p in non_self_params]:
                if method_name != '_update_training_progress':  # Private method exception
                    issue = f"{api_name}.{method_name}: TrainingAPI public method should have session_id parameter"
                    method_info['issues'].append(issue)
        
        # Check for model_id parameter in PredictionAPI and ExplanationAPI methods
        if api_name in ['PredictionAPI', 'ExplanationAPI'] and method_info['is_public']:
            if method_name not in ['__init__', 'get_loaded_models'] and 'model_id' not in [p['name'] for p in non_self_params]:
                issue = f"{api_name}.{method_name}: {api_name} method should typically have model_id parameter"
                method_info['issues'].append(issue)
        
        # Check that API methods return Dict[str, Any]
        if method_info['is_public'] and method_name != '__init__':
            if method_info['return_annotation'] and 'Dict[str, Any]' not in method_info['return_annotation']:
                issue = f"{api_name}.{method_name}: API method should return Dict[str, Any] for consistency"
                method_info['issues'].append(issue)
    
    def print_api_summary(self, api_name: str, results: Dict[str, Any]):
        """Print summary for an API class"""
        stats = results['stats']
        issues = results['issues']
        
        print(f"\n{api_name} Summary:")
        print(f"  Total methods: {stats['total_methods']}")
        print(f"  Public methods: {stats['public_methods']}")
        print(f"  Private methods: {stats['private_methods']}")
        print(f"  Documented methods: {stats['documented_methods']}")
        print(f"  Type annotated methods: {stats['type_annotated_methods']}")
        
        if issues:
            print(f"\n  Issues found ({len(issues)}):")
            for issue in issues:
                print(f"    ⚠️  {issue}")
        else:
            print(f"\n  ✅ No issues found!")
        
        # Print method details for public methods
        public_methods = [(name, info) for name, info in results['methods'].items() 
                         if info['is_public'] and name != '__init__']
        
        if public_methods:
            print(f"\n  Public Methods ({len(public_methods)}):")
            for method_name, method_info in public_methods:
                status = "✅" if not method_info['issues'] else "⚠️ "
                print(f"    {status} {method_name}{method_info['signature']}")
                if method_info['issues']:
                    for issue in method_info['issues']:
                        print(f"        - {issue}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report"""
        report = []
        report.append("# API Function Check Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_issues = sum(len(api_results['issues']) for api_results in results.values())
        report.append(f"## Summary")
        report.append(f"- Total APIs checked: {len(results)}")
        report.append(f"- Total issues found: {total_issues}")
        report.append("")
        
        for api_name, api_results in results.items():
            report.append(f"## {api_name}")
            stats = api_results['stats']
            report.append(f"- Total methods: {stats['total_methods']}")
            report.append(f"- Public methods: {stats['public_methods']}")
            report.append(f"- Issues: {len(api_results['issues'])}")
            
            if api_results['issues']:
                report.append(f"\n### Issues")
                for issue in api_results['issues']:
                    report.append(f"- {issue}")
            
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function to run API checks"""
    print("Starting API Function Check...")
    
    checker = APIFunctionChecker()
    
    try:
        results = checker.check_all_apis()
        
        # Generate and save report
        report_content = checker.generate_report(results)
        
        report_file = Path(__file__).parent / "api_check_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"\n{'='*60}")
        print(f"API Check Complete!")
        print(f"Report saved to: {report_file}")
        print(f"{'='*60}")
        
        # Print overall summary
        total_methods = sum(api['stats']['total_methods'] for api in results.values())
        total_issues = sum(len(api['issues']) for api in results.values())
        
        print(f"\nOverall Summary:")
        print(f"  APIs checked: {len(results)}")
        print(f"  Total methods: {total_methods}")
        print(f"  Total issues: {total_issues}")
        
        if total_issues > 0:
            print(f"\n⚠️  {total_issues} issues found. See report for details.")
        else:
            print(f"\n✅ All APIs look good!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during API check: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()