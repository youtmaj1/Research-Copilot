#!/bin/bash

# Research Copilot - Automated Cleanup Script
# This script safely removes development/testing artifacts
# Run from the project root directory

set -e  # Exit on error

echo "🧹 Research Copilot - Project Cleanup Script"
echo "=============================================="
echo ""
echo "This script will remove development artifacts and organize test files."
echo "A backup will NOT be created - ensure you have git commits first!"
echo ""
read -p "Continue? (yes/no): " confirmation
if [ "$confirmation" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "📋 Phase 1: Creating proper test directory structure..."
mkdir -p tests/performance
mkdir -p examples
mkdir -p tests/integration

echo "📋 Phase 2: Moving demo and example files..."
[ -f demo_module2.py ] && mv demo_module2.py tests/demo_module2.py && echo "  ✓ Moved demo_module2.py"
[ -f interactive_demo.py ] && mv interactive_demo.py examples/interactive_demo.py && echo "  ✓ Moved interactive_demo.py"
[ -f research_demo.py ] && mv research_demo.py examples/research_demo.py && echo "  ✓ Moved research_demo.py"
[ -f simple_demo.py ] && mv simple_demo.py examples/simple_demo.py && echo "  ✓ Moved simple_demo.py"

echo "📋 Phase 3: Moving performance and integration tests..."
[ -f load_test.py ] && mv load_test.py tests/performance/load_test.py && echo "  ✓ Moved load_test.py"
[ -f advanced_paper_testing.py ] && mv advanced_paper_testing.py tests/advanced_paper_testing.py && echo "  ✓ Moved advanced_paper_testing.py"
[ -f new_papers_test.py ] && mv new_papers_test.py tests/new_papers_test.py && echo "  ✓ Moved new_papers_test.py"
[ -f research_query_test.py ] && mv research_query_test.py tests/research_query_test.py && echo "  ✓ Moved research_query_test.py"
[ -f test_module2_comprehensive.py ] && mv test_module2_comprehensive.py tests/test_module2_comprehensive.py && echo "  ✓ Moved test_module2_comprehensive.py"
[ -f test_module2_fixed.py ] && mv test_module2_fixed.py tests/test_module2_fixed.py && echo "  ✓ Moved test_module2_fixed.py"
[ -f test_real_world.py ] && mv test_real_world.py tests/test_real_world.py && echo "  ✓ Moved test_real_world.py"
[ -f direct_research_test.py ] && mv direct_research_test.py tests/direct_research_test.py && echo "  ✓ Moved direct_research_test.py"
[ -f simplified_end_to_end_test.py ] && mv simplified_end_to_end_test.py tests/end_to_end_simplified.py && echo "  ✓ Moved simplified_end_to_end_test.py"
[ -f real_world_end_to_end_test.py ] && mv real_world_end_to_end_test.py tests/integration/end_to_end.py && echo "  ✓ Moved real_world_end_to_end_test.py"

echo "📋 Phase 4: Deleting validation files..."
rm -f validate_module.py && echo "  ✓ Deleted validate_module.py"
rm -f validate_module2.py && echo "  ✓ Deleted validate_module2.py"
rm -f validate_module3.py && echo "  ✓ Deleted validate_module3.py"
rm -f validate_module4.py && echo "  ✓ Deleted validate_module4.py"
rm -f complete_integration_test.py && echo "  ✓ Deleted complete_integration_test.py"
rm -f complete_system_validation.py && echo "  ✓ Deleted complete_system_validation.py"
rm -f comprehensive_module_validation.py && echo "  ✓ Deleted comprehensive_module_validation.py"
rm -f comprehensive_validation.py && echo "  ✓ Deleted comprehensive_validation.py"
rm -f end_to_end_system_test.py && echo "  ✓ Deleted end_to_end_system_test.py"
rm -f enterprise_performance_test.py && echo "  ✓ Deleted enterprise_performance_test.py"
rm -f enterprise_system_summary.py && echo "  ✓ Deleted enterprise_system_summary.py"
rm -f final_assessment.py && echo "  ✓ Deleted final_assessment.py"
rm -f final_validation.py && echo "  ✓ Deleted final_validation.py"
rm -f final_validation_report.py && echo "  ✓ Deleted final_validation_report.py"
rm -f functional_validation.py && echo "  ✓ Deleted functional_validation.py"
rm -f intelligent_godel_testing.py && echo "  ✓ Deleted intelligent_godel_testing.py"
rm -f live_validation.py && echo "  ✓ Deleted live_validation.py"
rm -f production_roadmap.py && echo "  ✓ Deleted production_roadmap.py"
rm -f production_validation.py && echo "  ✓ Deleted production_validation.py"
rm -f progressive_validator.py && echo "  ✓ Deleted progressive_validator.py"
rm -f real_world_test.py && echo "  ✓ Deleted real_world_test.py"
rm -f current_system_analysis.py && echo "  ✓ Deleted current_system_analysis.py"
rm -f integration.py && echo "  ✓ Deleted integration.py"
rm -f quick_system_status.py && echo "  ✓ Deleted quick_system_status.py"
rm -f quick_validate.py && echo "  ✓ Deleted quick_validate.py"

echo "📋 Phase 5: Deleting report and documentation files..."
rm -f COMPREHENSIVE_VALIDATION_REPORT.md && echo "  ✓ Deleted COMPREHENSIVE_VALIDATION_REPORT.md"
rm -f END_TO_END_TEST_REPORT.md && echo "  ✓ Deleted END_TO_END_TEST_REPORT.md"
rm -f MODULE2_VALIDATION_REPORT.md && echo "  ✓ Deleted MODULE2_VALIDATION_REPORT.md"
rm -f MODULE4_SUMMARY.md && echo "  ✓ Deleted MODULE4_SUMMARY.md"
rm -f FINAL_ENTERPRISE_READINESS_ASSESSMENT.md && echo "  ✓ Deleted FINAL_ENTERPRISE_READINESS_ASSESSMENT.md"
rm -f FINAL_STATUS.md && echo "  ✓ Deleted FINAL_STATUS.md"
rm -f NEXT_STEPS.md && echo "  ✓ Deleted NEXT_STEPS.md"
rm -f ResearchCopilotProject.txt && echo "  ✓ Deleted ResearchCopilotProject.txt"

echo "📋 Phase 6: Deleting JSON report files..."
rm -f *.json 2>/dev/null || true
echo "  ✓ Deleted all .json report files"

echo "📋 Phase 7: Deleting log files..."
rm -f *.log 2>/dev/null || true
echo "  ✓ Deleted all .log files"

echo "📋 Phase 8: Deleting test databases..."
rm -f test.db 2>/dev/null || true
rm -f test_*.db 2>/dev/null || true
rm -f papers.db 2>/dev/null || true
rm -f knowledge_graph.db 2>/dev/null || true
echo "  ✓ Deleted test database files"

echo "📋 Phase 9: Cleaning Python cache..."
rm -rf __pycache__ 2>/dev/null || true
rm -rf .pytest_cache 2>/dev/null || true
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Cleaned Python cache"

echo "📋 Phase 10: Removing macOS system files..."
rm -f .DS_Store 2>/dev/null || true
find . -name .DS_Store -delete 2>/dev/null || true
echo "  ✓ Removed macOS system files"

echo ""
echo "📋 Phase 11: Creating .gitignore..."
cat > .gitignore << 'EOF'
# Virtual environments
.venv/
venv/
env/
ENV/
.env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Environment variables
.env
.env.local
.env.production

# Databases
*.db
*.sqlite
*.sqlite3

# Logs
*.log
logs/

# Cache and temporary files
.cache/
*.tmp
tmp/

# Testing
test_results/
test_output.json

# Large files (if applicable)
# data/raw/papers/*.pdf

# API keys and secrets
secrets.json
credentials.json
.aws/
EOF
echo "  ✓ Created .gitignore"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Summary:"
echo "  ✓ Moved demo files to examples/"
echo "  ✓ Moved tests to tests/"
echo "  ✓ Deleted validation files"
echo "  ✓ Deleted report files"
echo "  ✓ Deleted log files"
echo "  ✓ Cleaned cache directories"
echo "  ✓ Created .gitignore"
echo ""
echo "🎯 Next steps:"
echo "  1. Review the CLEANUP_AND_GITHUB_GUIDE.md for more details"
echo "  2. Consolidate documentation into README.md"
echo "  3. Run: git status"
echo "  4. Run: pytest tests/"
echo "  5. Commit changes: git commit -m 'Clean up development artifacts'"
echo ""
