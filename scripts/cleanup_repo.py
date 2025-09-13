#!/usr/bin/env python3
"""
Repository cleanup and organization script.

This script performs comprehensive cleanup and organization tasks including:
- Code formatting and linting
- File organization and structure validation
- Documentation updates
- Dependency management
- Security scanning
- Performance optimization suggestions
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
import hashlib

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class RepoCleanupManager:
    """Manager class for repository cleanup and organization."""

    def __init__(self, repo_path: str):
        """
        Initialize cleanup manager.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)
        self.issues = []
        self.fixes_applied = []

    def run_full_cleanup(self) -> Dict[str, Any]:
        """
        Run comprehensive repository cleanup.

        Returns:
            Cleanup report with issues found and fixes applied
        """
        print("🔧 Starting comprehensive repository cleanup...")

        # Run all cleanup tasks
        tasks = [
            ("Code Formatting", self.check_code_formatting),
            ("File Organization", self.check_file_organization),
            ("Documentation", self.check_documentation),
            ("Dependencies", self.check_dependencies),
            ("Security", self.check_security_issues),
            ("Performance", self.check_performance_issues),
            ("Git Health", self.check_git_health),
            ("File Permissions", self.check_file_permissions),
        ]

        results = {}
        total_issues = 0
        total_fixes = 0

        for task_name, task_func in tasks:
            print(f"\n📋 Running {task_name} check...")
            try:
                task_results = task_func()
                results[task_name] = task_results
                total_issues += task_results.get("issues_found", 0)
                total_fixes += task_results.get("fixes_applied", 0)
                print(f"   ✓ {task_name}: {task_results.get('issues_found', 0)} issues, {task_results.get('fixes_applied', 0)} fixes")
            except Exception as e:
                print(f"   ✗ {task_name} failed: {e}")
                results[task_name] = {"error": str(e)}

        # Generate cleanup report
        report = {
            "cleanup_timestamp": datetime.now().isoformat(),
            "repository_path": str(self.repo_path),
            "total_issues_found": total_issues,
            "total_fixes_applied": total_fixes,
            "tasks_results": results,
            "recommendations": self.generate_recommendations(results)
        }

        # Save report
        report_path = self.repo_path / "cleanup_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Cleanup complete! Report saved to {report_path}")
        print(f"   Total issues found: {total_issues}")
        print(f"   Total fixes applied: {total_fixes}")

        return report

    def check_code_formatting(self) -> Dict[str, Any]:
        """Check and fix code formatting issues."""
        issues_found = 0
        fixes_applied = 0

        # Python files to check
        python_files = list(self.repo_path.rglob("*.py"))

        # Check for basic formatting issues
        for py_file in python_files:
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()

            # Check for trailing whitespace
            lines = content.split('\n')
            has_trailing_whitespace = any(line.rstrip() != line for line in lines)
            if has_trailing_whitespace:
                issues_found += 1
                # Fix trailing whitespace
                fixed_content = '\n'.join(line.rstrip() for line in lines)
                py_file.write_text(fixed_content)
                fixes_applied += 1

            # Check for long lines (>100 characters)
            long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 100]
            if long_lines:
                issues_found += len(long_lines)
                # Note: We don't auto-fix long lines as they may need manual review

            # Check for missing final newline
            if content and not content.endswith('\n'):
                issues_found += 1
                py_file.write_text(content + '\n')
                fixes_applied += 1

        # Check if black is available and run formatting
        try:
            subprocess.run([sys.executable, "-m", "black", "--check", "--diff", "."],
                         cwd=self.repo_path, capture_output=True)
            print("   Black formatting check completed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("   Black not available or formatting issues found")

        return {
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "python_files_checked": len(python_files)
        }

    def check_file_organization(self) -> Dict[str, Any]:
        """Check and organize file structure."""
        issues_found = 0
        fixes_applied = 0

        # Check for required directories
        required_dirs = [
            "src/openinvestments",
            "data",
            "models",
            "logs",
            "reports",
            "tests",
            "docs"
        ]

        for dir_path in required_dirs:
            full_path = self.repo_path / dir_path
            if not full_path.exists():
                issues_found += 1
                full_path.mkdir(parents=True, exist_ok=True)
                fixes_applied += 1
                print(f"   Created directory: {dir_path}")

        # Check for misplaced files
        root_files = list(self.repo_path.glob("*"))
        misplaced_files = []

        for file_path in root_files:
            if file_path.is_file() and file_path.name not in [
                "main.py", "setup.py", "requirements.txt", "README.md",
                "Dockerfile", "docker-compose.yml", ".gitignore", "LICENSE"
            ] and not file_path.name.startswith('.'):
                misplaced_files.append(file_path)

        if misplaced_files:
            issues_found += len(misplaced_files)
            # Create scripts directory and move files
            scripts_dir = self.repo_path / "scripts"
            scripts_dir.mkdir(exist_ok=True)

            for file_path in misplaced_files:
                if file_path.name.endswith('.py'):
                    new_path = scripts_dir / file_path.name
                    shutil.move(file_path, new_path)
                    fixes_applied += 1
                    print(f"   Moved {file_path.name} to scripts/")

        # Check for empty directories
        empty_dirs = []
        for dir_path in self.repo_path.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                empty_dirs.append(dir_path)

        if empty_dirs:
            issues_found += len(empty_dirs)
            for empty_dir in empty_dirs:
                if "venv" not in str(empty_dir) and "__pycache__" not in str(empty_dir):
                    empty_dir.rmdir()
                    fixes_applied += 1

        return {
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "directories_created": len(required_dirs),
            "files_moved": len(misplaced_files) if 'misplaced_files' in locals() else 0
        }

    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness and quality."""
        issues_found = 0
        fixes_applied = 0

        # Check for README files
        readme_files = list(self.repo_path.glob("README*")) + list(self.repo_path.glob("readme*"))
        if not readme_files:
            issues_found += 1
            # Create basic README if missing
            readme_content = f"""# {self.repo_path.name}

OpenInvestments Quantitative Risk Platform

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
python main.py --help
```

## License

MIT License
"""
            readme_path = self.repo_path / "README.md"
            readme_path.write_text(readme_content)
            fixes_applied += 1

        # Check Python files for docstrings
        python_files = list(self.repo_path.rglob("*.py"))
        files_without_docstrings = []

        for py_file in python_files:
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()

            # Check if file has module docstring
            if not re.search(r'""".*?"""', content, re.DOTALL):
                files_without_docstrings.append(py_file)
                issues_found += 1

        # Check for API documentation
        api_docs = self.repo_path / "docs" / "api"
        if not api_docs.exists():
            api_docs.mkdir(parents=True, exist_ok=True)
            fixes_applied += 1

        return {
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "readme_files": len(readme_files),
            "files_without_docstrings": len(files_without_docstrings),
            "documentation_dirs_created": 1 if fixes_applied > 0 else 0
        }

    def check_dependencies(self) -> Dict[str, Any]:
        """Check dependency management and requirements."""
        issues_found = 0
        fixes_applied = 0

        # Check for requirements.txt
        requirements_file = self.repo_path / "requirements.txt"
        if not requirements_file.exists():
            issues_found += 1
            # Create basic requirements file
            basic_requirements = """numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
fastapi>=0.85.0
uvicorn>=0.18.0
click>=8.0.0
rich>=12.0.0
"""
            requirements_file.write_text(basic_requirements)
            fixes_applied += 1

        # Check for setup.py/pyproject.toml
        setup_files = ["setup.py", "pyproject.toml"]
        has_setup = any((self.repo_path / f).exists() for f in setup_files)
        if not has_setup:
            issues_found += 1

        # Check for version conflicts in requirements
        if requirements_file.exists():
            with open(requirements_file) as f:
                requirements = f.read()

            # Check for unpinned versions
            unpinned_deps = re.findall(r'^\w+>=[\d.]+$', requirements, re.MULTILINE)
            if unpinned_deps:
                issues_found += len(unpinned_deps)

        return {
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "requirements_file_exists": requirements_file.exists(),
            "setup_file_exists": has_setup,
            "unpinned_dependencies": len(unpinned_deps) if 'unpinned_deps' in locals() else 0
        }

    def check_security_issues(self) -> Dict[str, Any]:
        """Check for security vulnerabilities and issues."""
        issues_found = 0
        fixes_applied = 0

        # Check for sensitive files that shouldn't be committed
        sensitive_files = [
            ".env",
            "config.env",
            "secrets.json",
            "credentials.json",
            ".aws/credentials",
            "id_rsa",
            "id_rsa.pub"
        ]

        for sensitive_file in sensitive_files:
            file_path = self.repo_path / sensitive_file
            if file_path.exists():
                issues_found += 1
                print(f"   ⚠️  Found sensitive file: {sensitive_file}")

        # Check .gitignore file
        gitignore_path = self.repo_path / ".gitignore"
        if not gitignore_path.exists():
            issues_found += 1
            # Create basic .gitignore
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data and models
data/
models/
logs/
reports/

# Secrets
.env
*.key
*.pem
secrets.json
"""
            gitignore_path.write_text(gitignore_content)
            fixes_applied += 1

        # Check for hardcoded secrets in Python files
        python_files = list(self.repo_path.rglob("*.py"))
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]

        for py_file in python_files:
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            for pattern in secret_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    issues_found += len(matches)
                    print(f"   ⚠️  Potential hardcoded secrets in {py_file.name}")

        return {
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "sensitive_files_found": sum(1 for f in sensitive_files if (self.repo_path / f).exists()),
            "gitignore_created": fixes_applied > 0
        }

    def check_performance_issues(self) -> Dict[str, Any]:
        """Check for potential performance issues."""
        issues_found = 0
        fixes_applied = 0

        # Check for large files
        large_files = []
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                large_files.append((file_path, file_path.stat().st_size))
                issues_found += 1

        if large_files:
            print("   ⚠️  Large files found:")
            for file_path, size in large_files:
                print(f"      {file_path.name}: {size / (1024*1024):.1f} MB")

        # Check for inefficient imports
        python_files = list(self.repo_path.rglob("*.py"))
        star_imports = []

        for py_file in python_files:
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            if re.search(r'from\s+\w+\s+import\s+\*', content):
                star_imports.append(py_file)
                issues_found += 1

        # Check for memory-intensive operations
        memory_intensive_patterns = [
            r'pd\.read_csv.*chunksize',
            r'pd\.read_excel',
            r'plt\.figure.*dpi=[3-9]\d+'
        ]

        memory_issues = []
        for py_file in python_files:
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            for pattern in memory_intensive_patterns:
                if re.search(pattern, content):
                    memory_issues.append((py_file, pattern))
                    issues_found += 1

        return {
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "large_files": len(large_files),
            "star_imports": len(star_imports),
            "memory_intensive_operations": len(memory_issues)
        }

    def check_git_health(self) -> Dict[str, Any]:
        """Check Git repository health."""
        issues_found = 0
        fixes_applied = 0

        # Check if it's a git repository
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            issues_found += 1
            print("   ⚠️  Not a Git repository")
            return {"issues_found": issues_found, "fixes_applied": fixes_applied, "is_git_repo": False}

        try:
            # Check Git status
            result = subprocess.run(["git", "status", "--porcelain"],
                                  cwd=self.repo_path, capture_output=True, text=True)

            if result.returncode == 0:
                uncommitted_files = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                if uncommitted_files > 0:
                    issues_found += uncommitted_files
                    print(f"   ⚠️  {uncommitted_files} uncommitted files")

            # Check for large files in Git history
            result = subprocess.run(["git", "ls-files"],
                                  cwd=self.repo_path, capture_output=True, text=True)

            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                large_tracked_files = []

                for file_path in files:
                    if file_path:
                        full_path = self.repo_path / file_path
                        if full_path.exists() and full_path.stat().st_size > 50 * 1024 * 1024:  # 50MB
                            large_tracked_files.append(file_path)
                            issues_found += 1

                if large_tracked_files:
                    print("   ⚠️  Large files tracked by Git:")
                    for file_path in large_tracked_files:
                        size = (self.repo_path / file_path).stat().st_size
                        print(f"      {file_path}: {size / (1024*1024):.1f} MB")

        except FileNotFoundError:
            print("   ⚠️  Git not available")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Git command failed: {e}")

        return {
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "is_git_repo": True,
            "uncommitted_files": uncommitted_files if 'uncommitted_files' in locals() else 0,
            "large_tracked_files": len(large_tracked_files) if 'large_tracked_files' in locals() else 0
        }

    def check_file_permissions(self) -> Dict[str, Any]:
        """Check and fix file permissions."""
        issues_found = 0
        fixes_applied = 0

        # Check Python files for executable permissions
        python_files = list(self.repo_path.rglob("*.py"))

        for py_file in python_files:
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            # Check if file is executable
            if os.access(py_file, os.X_OK):
                issues_found += 1
                # Remove executable permission
                current_permissions = os.stat(py_file).st_mode
                new_permissions = current_permissions & ~0o111  # Remove execute permissions
                os.chmod(py_file, new_permissions)
                fixes_applied += 1

        # Check for files with incorrect permissions
        all_files = list(self.repo_path.rglob("*"))
        permission_issues = []

        for file_path in all_files:
            if file_path.is_file():
                stat_info = file_path.stat()
                # Check for world-writable files
                if stat_info.st_mode & 0o002:
                    permission_issues.append(file_path)
                    issues_found += 1

        if permission_issues:
            print(f"   ⚠️  Found {len(permission_issues)} files with world-writable permissions")

        return {
            "issues_found": issues_found,
            "fixes_applied": fixes_applied,
            "executable_python_files": sum(1 for f in python_files if os.access(f, os.X_OK)),
            "world_writable_files": len(permission_issues)
        }

    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate cleanup recommendations based on results."""
        recommendations = []

        # Code formatting recommendations
        if results.get("Code Formatting", {}).get("issues_found", 0) > 0:
            recommendations.append("Run code formatting tools (black, isort) to standardize code style")
            recommendations.append("Configure pre-commit hooks for automatic formatting")

        # Documentation recommendations
        doc_results = results.get("Documentation", {})
        if doc_results.get("files_without_docstrings", 0) > 0:
            recommendations.append("Add docstrings to Python files for better documentation")
            recommendations.append("Consider using Sphinx for comprehensive API documentation")

        # Security recommendations
        security_results = results.get("Security", {})
        if security_results.get("sensitive_files_found", 0) > 0:
            recommendations.append("Move sensitive files to .env or secure storage")
            recommendations.append("Review .gitignore to ensure sensitive files are not committed")

        # Performance recommendations
        perf_results = results.get("Performance", {})
        if perf_results.get("large_files", 0) > 0:
            recommendations.append("Consider using Git LFS for large files or remove unnecessary large files")
        if perf_results.get("memory_intensive_operations", 0) > 0:
            recommendations.append("Review memory-intensive operations and consider optimizations")

        # Git recommendations
        git_results = results.get("Git Health", {})
        if not git_results.get("is_git_repo", False):
            recommendations.append("Initialize Git repository for version control")
        if git_results.get("uncommitted_files", 0) > 0:
            recommendations.append("Commit or stage important changes to Git")

        # General recommendations
        recommendations.extend([
            "Set up CI/CD pipeline with automated testing and linting",
            "Configure code quality tools (mypy, pylint, flake8)",
            "Create comprehensive test suite with good coverage",
            "Document API endpoints and usage patterns",
            "Set up monitoring and alerting for production deployment"
        ])

        return recommendations


def main():
    """Main cleanup function."""
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = "."

    print("🧹 OpenInvestments Repository Cleanup Tool")
    print("=" * 50)

    cleanup_manager = RepoCleanupManager(repo_path)
    report = cleanup_manager.run_full_cleanup()

    print("\n📋 Recommendations:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"   {i}. {rec}")

    print(f"\n✅ Cleanup complete! Check cleanup_report.json for detailed results.")


if __name__ == "__main__":
    main()
