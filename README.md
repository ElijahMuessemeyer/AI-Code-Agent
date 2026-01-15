# AI Code Agent

An autonomous AI agent that performs code review, generates code, detects bugs, and integrates with development workflows.

## Features

- 🔍 **Automated Code Review**: Analyze PRs and provide intelligent feedback
- 🛠️ **Code Generation**: Generate code from natural language requirements
- 🐛 **Bug Detection**: Identify and suggest fixes for code issues
- 🧪 **Test Generation**: Create comprehensive test suites automatically
- 🔗 **GitHub/GitLab Integration**: Seamless workflow integration
- 📊 **Quality Metrics**: Track code quality trends over time

## Quick Start

### Prerequisites

- Python 3.10+
- Redis (for caching and background tasks)
- PostgreSQL (for data storage)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ai-code-agent
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Run the application:
```bash
uvicorn src.api.main:app --reload
```

## Architecture

The agent consists of specialized components:

- **Code Reviewer Agent**: Analyzes code quality and provides feedback
- **Code Generator Agent**: Creates code from requirements
- **Bug Detector Agent**: Identifies potential issues and vulnerabilities
- **Test Generator Agent**: Creates comprehensive test suites

## Development Status

🚧 **Currently in Phase 1**: Core Analysis Engine
- [x] Project structure setup
- [x] Dependency management
- [ ] Static code analysis framework
- [ ] Basic LLM integration

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete roadmap.

## Contributing

This is a portfolio project demonstrating AI agent development skills. See [PROGRESS_NOTES.md](PROGRESS_NOTES.md) for development updates.

## License

MIT License - see LICENSE file for details.