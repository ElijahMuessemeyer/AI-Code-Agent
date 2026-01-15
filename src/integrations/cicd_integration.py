"""
CI/CD pipeline integration for automated code analysis in build processes.
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from src.agents.agent_coordinator import AgentCoordinator, WorkflowType


@dataclass
class PipelineConfig:
    """Configuration for CI/CD pipeline integration."""
    platform: str  # github_actions, gitlab_ci, jenkins, azure_devops
    project_name: str
    branch_patterns: List[str]
    analysis_triggers: List[str]  # pr, push, scheduled
    quality_gates: Dict[str, Any]
    notification_settings: Dict[str, Any]


@dataclass
class PipelineResult:
    """Result of CI/CD pipeline analysis."""
    pipeline_id: str
    project_name: str
    branch: str
    trigger: str
    workflow_results: Dict[str, Any]
    quality_gates_passed: bool
    recommendations: List[str]
    artifacts_generated: List[str]
    execution_time_ms: float


class CICDIntegration:
    """Handles CI/CD pipeline integration across multiple platforms."""
    
    def __init__(self):
        self.coordinator = AgentCoordinator()
        self.supported_platforms = [
            "github_actions",
            "gitlab_ci", 
            "jenkins",
            "azure_devops",
            "circleci"
        ]
    
    def generate_github_actions_workflow(self, config: PipelineConfig) -> str:
        """Generate GitHub Actions workflow file."""
        workflow = {
            "name": "AI Code Agent Analysis",
            "on": self._get_github_triggers(config.analysis_triggers, config.branch_patterns),
            "jobs": {
                "ai-analysis": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "3.10"
                            }
                        },
                        {
                            "name": "Install AI Code Agent",
                            "run": "pip install ai-code-agent"
                        },
                        {
                            "name": "Run comprehensive analysis",
                            "env": {
                                "OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}",
                                "ANTHROPIC_API_KEY": "${{ secrets.ANTHROPIC_API_KEY }}"
                            },
                            "run": "ai-code-agent analyze . --workflow comprehensive_analysis --output analysis_report.md"
                        },
                        {
                            "name": "Check quality gates",
                            "run": "ai-code-agent check-quality --config .ai-code-agent.yml"
                        },
                        {
                            "name": "Upload analysis report",
                            "uses": "actions/upload-artifact@v3",
                            "with": {
                                "name": "ai-analysis-report",
                                "path": "analysis_report.md"
                            }
                        },
                        {
                            "name": "Comment on PR", 
                            "if": "github.event_name == 'pull_request'",
                            "uses": "actions/github-script@v6",
                            "with": {
                                "script": self._get_pr_comment_script()
                            }
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False)
    
    def generate_gitlab_ci_config(self, config: PipelineConfig) -> str:
        """Generate GitLab CI configuration."""
        gitlab_config = {
            "stages": ["analysis", "quality-gate", "report"],
            "variables": {
                "PIP_CACHE_DIR": "$CI_PROJECT_DIR/.cache/pip"
            },
            "cache": {
                "paths": [".cache/pip", "venv/"]
            },
            "before_script": [
                "python -m venv venv",
                "source venv/bin/activate",
                "pip install ai-code-agent"
            ],
            "ai-analysis": {
                "stage": "analysis",
                "script": [
                    "source venv/bin/activate",
                    "ai-code-agent analyze . --workflow comprehensive_analysis --output analysis_report.md"
                ],
                "artifacts": {
                    "reports": {
                        "junit": "analysis_report.xml"
                    },
                    "paths": ["analysis_report.md"],
                    "expire_in": "1 week"
                },
                "only": config.branch_patterns
            },
            "quality-gate": {
                "stage": "quality-gate",
                "script": [
                    "source venv/bin/activate",
                    "ai-code-agent check-quality --config .ai-code-agent.yml --fail-on-issues"
                ],
                "dependencies": ["ai-analysis"],
                "allow_failure": False,
                "only": config.branch_patterns
            },
            "generate-report": {
                "stage": "report",
                "script": [
                    "source venv/bin/activate",
                    "ai-code-agent report --format html --output public/analysis.html"
                ],
                "artifacts": {
                    "paths": ["public/"],
                    "expire_in": "30 days"
                },
                "dependencies": ["ai-analysis"],
                "only": config.branch_patterns
            }
        }
        
        return yaml.dump(gitlab_config, default_flow_style=False)
    
    def generate_jenkins_pipeline(self, config: PipelineConfig) -> str:
        """Generate Jenkins pipeline script."""
        pipeline_script = f'''
pipeline {{
    agent any
    
    environment {{
        OPENAI_API_KEY = credentials('openai-api-key')
        ANTHROPIC_API_KEY = credentials('anthropic-api-key')
    }}
    
    triggers {{
        pollSCM('H/5 * * * *')
    }}
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}
        
        stage('Setup') {{
            steps {{
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install ai-code-agent
                '''
            }}
        }}
        
        stage('AI Analysis') {{
            steps {{
                sh '''
                    . venv/bin/activate
                    ai-code-agent analyze . --workflow comprehensive_analysis --output analysis_report.md
                '''
            }}
            post {{
                always {{
                    archiveArtifacts artifacts: 'analysis_report.md', fingerprint: true
                }}
            }}
        }}
        
        stage('Quality Gate') {{
            steps {{
                sh '''
                    . venv/bin/activate
                    ai-code-agent check-quality --config .ai-code-agent.yml
                '''
            }}
        }}
        
        stage('Generate Reports') {{
            steps {{
                sh '''
                    . venv/bin/activate
                    ai-code-agent report --format html --output analysis.html
                '''
            }}
            post {{
                always {{
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: '.',
                        reportFiles: 'analysis.html',
                        reportName: 'AI Code Analysis Report'
                    ])
                }}
            }}
        }}
    }}
    
    post {{
        always {{
            cleanWs()
        }}
        failure {{
            emailext (
                subject: "AI Code Analysis Failed: ${{env.JOB_NAME}} - ${{env.BUILD_NUMBER}}",
                body: "The AI code analysis pipeline has failed. Please check the console output.",
                recipientProviders: [developers()]
            )
        }}
    }}
}}
'''
        return pipeline_script.strip()
    
    def generate_azure_devops_pipeline(self, config: PipelineConfig) -> str:
        """Generate Azure DevOps pipeline YAML."""
        azure_config = {
            "trigger": config.branch_patterns,
            "pr": config.branch_patterns,
            "pool": {
                "vmImage": "ubuntu-latest"
            },
            "variables": {
                "pythonVersion": "3.10"
            },
            "stages": [
                {
                    "stage": "Analysis",
                    "displayName": "AI Code Analysis",
                    "jobs": [
                        {
                            "job": "AIAnalysis",
                            "displayName": "Run AI Analysis",
                            "steps": [
                                {
                                    "task": "UsePythonVersion@0",
                                    "inputs": {
                                        "versionSpec": "$(pythonVersion)"
                                    },
                                    "displayName": "Use Python $(pythonVersion)"
                                },
                                {
                                    "script": "pip install ai-code-agent",
                                    "displayName": "Install AI Code Agent"
                                },
                                {
                                    "script": "ai-code-agent analyze . --workflow comprehensive_analysis --output $(Agent.TempDirectory)/analysis_report.md",
                                    "displayName": "Run Analysis",
                                    "env": {
                                        "OPENAI_API_KEY": "$(OPENAI_API_KEY)",
                                        "ANTHROPIC_API_KEY": "$(ANTHROPIC_API_KEY)"
                                    }
                                },
                                {
                                    "task": "PublishPipelineArtifact@1",
                                    "inputs": {
                                        "targetPath": "$(Agent.TempDirectory)/analysis_report.md",
                                        "artifact": "analysis-report",
                                        "publishLocation": "pipeline"
                                    },
                                    "displayName": "Publish Analysis Report"
                                }
                            ]
                        }
                    ]
                },
                {
                    "stage": "QualityGate", 
                    "displayName": "Quality Gate",
                    "dependsOn": "Analysis",
                    "jobs": [
                        {
                            "job": "QualityCheck",
                            "displayName": "Quality Gate Check",
                            "steps": [
                                {
                                    "script": "ai-code-agent check-quality --config .ai-code-agent.yml --fail-on-issues",
                                    "displayName": "Check Quality Gates"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        return yaml.dump(azure_config, default_flow_style=False)
    
    def generate_quality_config(self, config: PipelineConfig) -> str:
        """Generate quality gates configuration file."""
        quality_config = {
            "version": "1.0",
            "quality_gates": {
                "overall_score": {
                    "min_score": config.quality_gates.get("min_score", 7.0),
                    "fail_below": True
                },
                "bugs": {
                    "max_critical": config.quality_gates.get("max_critical_bugs", 0),
                    "max_high": config.quality_gates.get("max_high_bugs", 2),
                    "max_total": config.quality_gates.get("max_total_bugs", 10)
                },
                "security": {
                    "max_critical": config.quality_gates.get("max_security_critical", 0),
                    "max_high": config.quality_gates.get("max_security_high", 1)
                },
                "test_coverage": {
                    "min_coverage": config.quality_gates.get("min_test_coverage", 0.8)
                },
                "maintainability": {
                    "min_score": config.quality_gates.get("min_maintainability", 6.0)
                }
            },
            "analysis": {
                "include_patterns": ["**/*.py", "**/*.js", "**/*.ts", "**/*.java"],
                "exclude_patterns": ["**/node_modules/**", "**/venv/**", "**/.git/**"],
                "max_files": config.quality_gates.get("max_files", 50)
            },
            "notifications": config.notification_settings,
            "reporting": {
                "formats": ["markdown", "html", "json"],
                "include_suggestions": True,
                "include_examples": True
            }
        }
        
        return yaml.dump(quality_config, default_flow_style=False)
    
    async def run_pipeline_analysis(self, 
                                  source_dir: str, 
                                  config: PipelineConfig,
                                  pipeline_context: Dict[str, Any]) -> PipelineResult:
        """Run analysis as part of CI/CD pipeline."""
        import time
        start_time = time.time()
        
        pipeline_id = pipeline_context.get("pipeline_id", "local")
        branch = pipeline_context.get("branch", "main")
        trigger = pipeline_context.get("trigger", "manual")
        
        # Run comprehensive analysis
        workflow_result = await self.coordinator.analyze_codebase(source_dir)
        
        # Check quality gates
        quality_gates_passed = self._check_quality_gates(workflow_result, config.quality_gates)
        
        # Generate recommendations
        recommendations = self._generate_pipeline_recommendations(
            workflow_result, quality_gates_passed, config
        )
        
        # Generate artifacts
        artifacts = await self._generate_pipeline_artifacts(
            workflow_result, source_dir, config
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            pipeline_id=pipeline_id,
            project_name=config.project_name,
            branch=branch,
            trigger=trigger,
            workflow_results=workflow_result.__dict__,
            quality_gates_passed=quality_gates_passed,
            recommendations=recommendations,
            artifacts_generated=artifacts,
            execution_time_ms=execution_time
        )
    
    def _get_github_triggers(self, triggers: List[str], branches: List[str]) -> Dict[str, Any]:
        """Generate GitHub Actions trigger configuration."""
        github_triggers = {}
        
        if "push" in triggers:
            github_triggers["push"] = {"branches": branches}
        
        if "pr" in triggers:
            github_triggers["pull_request"] = {"branches": branches}
        
        if "scheduled" in triggers:
            github_triggers["schedule"] = [{"cron": "0 2 * * *"}]  # Daily at 2 AM
        
        return github_triggers
    
    def _get_pr_comment_script(self) -> str:
        """Generate GitHub Actions script for PR comments."""
        return '''
const fs = require('fs');
const path = './analysis_report.md';

if (fs.existsSync(path)) {
  const report = fs.readFileSync(path, 'utf8');
  
  github.rest.issues.createComment({
    issue_number: context.issue.number,
    owner: context.repo.owner,
    repo: context.repo.repo,
    body: '## 🤖 AI Code Agent Analysis\\n\\n' + report
  });
}
'''
    
    def _check_quality_gates(self, workflow_result, quality_gates: Dict[str, Any]) -> bool:
        """Check if analysis results pass quality gates."""
        if not hasattr(workflow_result, 'task_results'):
            return True
        
        # Extract metrics from workflow results
        review_results = []
        bug_results = []
        
        for task_result in workflow_result.task_results.values():
            if hasattr(task_result, 'ai_feedback'):
                review_results.append(task_result)
            elif hasattr(task_result, 'bugs_detected'):
                bug_results.append(task_result)
        
        # Check overall score
        if review_results:
            avg_score = sum(r.ai_feedback.overall_score for r in review_results) / len(review_results)
            min_score = quality_gates.get("min_score", 7.0)
            if avg_score < min_score:
                return False
        
        # Check bug counts
        if bug_results:
            total_bugs = sum(len(r.bugs_detected) for r in bug_results)
            critical_bugs = sum(
                sum(1 for bug in r.bugs_detected if bug.severity.value == "critical")
                for r in bug_results
            )
            
            if critical_bugs > quality_gates.get("max_critical_bugs", 0):
                return False
            
            if total_bugs > quality_gates.get("max_total_bugs", 10):
                return False
        
        return True
    
    def _generate_pipeline_recommendations(self, 
                                         workflow_result,
                                         quality_gates_passed: bool,
                                         config: PipelineConfig) -> List[str]:
        """Generate recommendations for pipeline results."""
        recommendations = []
        
        if not quality_gates_passed:
            recommendations.append("❌ Quality gates failed - address critical issues before merging")
        else:
            recommendations.append("✅ All quality gates passed")
        
        # Add specific recommendations based on results
        if hasattr(workflow_result, 'recommendations'):
            recommendations.extend(workflow_result.recommendations[:3])
        
        # Add CI/CD specific recommendations
        recommendations.append("Consider adding pre-commit hooks for faster feedback")
        recommendations.append("Set up branch protection rules requiring analysis checks")
        
        return recommendations
    
    async def _generate_pipeline_artifacts(self, 
                                         workflow_result,
                                         source_dir: str,
                                         config: PipelineConfig) -> List[str]:
        """Generate artifacts for pipeline."""
        artifacts = []
        
        # Generate markdown report
        if hasattr(workflow_result, 'summary'):
            report_content = f"""# AI Code Analysis Report

## Summary
{workflow_result.summary}

## Results
- Success Rate: {workflow_result.success_rate:.1%}
- Execution Time: {workflow_result.execution_time_ms:.0f}ms
- Files Processed: {len(workflow_result.target_files)}

## Recommendations
"""
            for rec in workflow_result.recommendations:
                report_content += f"- {rec}\n"
            
            report_path = os.path.join(source_dir, "analysis_report.md")
            with open(report_path, 'w') as f:
                f.write(report_content)
            artifacts.append("analysis_report.md")
        
        # Generate JSON summary for programmatic consumption
        summary_data = {
            "timestamp": time.time(),
            "project": config.project_name,
            "success_rate": getattr(workflow_result, 'success_rate', 0.0),
            "execution_time_ms": getattr(workflow_result, 'execution_time_ms', 0.0),
            "files_processed": len(getattr(workflow_result, 'target_files', [])),
            "recommendations": getattr(workflow_result, 'recommendations', [])
        }
        
        summary_path = os.path.join(source_dir, "analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        artifacts.append("analysis_summary.json")
        
        return artifacts
    
    def create_setup_guide(self, platform: str, config: PipelineConfig) -> str:
        """Create setup guide for specific CI/CD platform."""
        if platform == "github_actions":
            return self._create_github_setup_guide(config)
        elif platform == "gitlab_ci":
            return self._create_gitlab_setup_guide(config)
        elif platform == "jenkins":
            return self._create_jenkins_setup_guide(config)
        elif platform == "azure_devops":
            return self._create_azure_setup_guide(config)
        else:
            return "Platform not supported"
    
    def _create_github_setup_guide(self, config: PipelineConfig) -> str:
        """Create GitHub Actions setup guide."""
        return f"""# GitHub Actions Setup Guide

## 1. Create Workflow File
Create `.github/workflows/ai-code-agent.yml` with the generated configuration.

## 2. Set Repository Secrets
Go to Repository Settings > Secrets and add:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key (optional)

## 3. Create Quality Configuration
Create `.ai-code-agent.yml` in your repository root with quality gates.

## 4. Enable Required Checks (Optional)
1. Go to Repository Settings > Branches
2. Add branch protection rule for main/master
3. Require "AI Code Agent Analysis" check to pass

## 5. Test the Setup
Create a pull request to trigger the analysis workflow.

## Configuration Files Generated:
- Workflow: `.github/workflows/ai-code-agent.yml`
- Quality Config: `.ai-code-agent.yml`

The workflow will run on: {', '.join(config.analysis_triggers)}
For branches: {', '.join(config.branch_patterns)}
"""
    
    def _create_gitlab_setup_guide(self, config: PipelineConfig) -> str:
        """Create GitLab CI setup guide."""
        return f"""# GitLab CI Setup Guide

## 1. Create Pipeline File
Create `.gitlab-ci.yml` in your repository root with the generated configuration.

## 2. Set CI/CD Variables
Go to Project Settings > CI/CD > Variables and add:
- `OPENAI_API_KEY`: Your OpenAI API key (masked)
- `ANTHROPIC_API_KEY`: Your Anthropic API key (masked, optional)

## 3. Create Quality Configuration
Create `.ai-code-agent.yml` in your repository root.

## 4. Enable Merge Request Pipelines
1. Go to Project Settings > CI/CD > General pipelines
2. Enable "Pipelines for merge requests"

## 5. Test the Setup
Create a merge request to trigger the pipeline.

## Configuration Files Generated:
- Pipeline: `.gitlab-ci.yml`
- Quality Config: `.ai-code-agent.yml`

The pipeline will run for branches: {', '.join(config.branch_patterns)}
"""
    
    def _create_jenkins_setup_guide(self, config: PipelineConfig) -> str:
        """Create Jenkins setup guide."""
        return f"""# Jenkins Setup Guide

## 1. Install Required Plugins
- Pipeline Plugin
- Git Plugin
- HTML Publisher Plugin
- Email Extension Plugin

## 2. Create Pipeline Job
1. New Item > Pipeline
2. Copy the generated Jenkinsfile content
3. Configure SCM polling if needed

## 3. Set Credentials
1. Manage Jenkins > Credentials
2. Add secret text credentials:
   - ID: `openai-api-key`, Value: Your OpenAI API key
   - ID: `anthropic-api-key`, Value: Your Anthropic API key

## 4. Configure Notifications
Set up email notifications in job configuration.

## 5. Test the Setup
Trigger a build manually or push to the repository.

Project: {config.project_name}
Branches: {', '.join(config.branch_patterns)}
"""
    
    def _create_azure_setup_guide(self, config: PipelineConfig) -> str:
        """Create Azure DevOps setup guide."""
        return f"""# Azure DevOps Setup Guide

## 1. Create Pipeline File
Create `azure-pipelines.yml` in your repository root.

## 2. Set Pipeline Variables
1. Go to Pipelines > Your Pipeline > Edit
2. Click Variables and add:
   - `OPENAI_API_KEY`: Your OpenAI API key (secret)
   - `ANTHROPIC_API_KEY`: Your Anthropic API key (secret)

## 3. Create Quality Configuration
Create `.ai-code-agent.yml` in your repository root.

## 4. Enable Branch Policies (Optional)
1. Go to Repos > Branches
2. Set policies for main branch
3. Add build validation requiring the pipeline

## 5. Test the Setup
Create a pull request to trigger the pipeline.

## Configuration Generated:
- Pipeline: `azure-pipelines.yml`
- Quality Config: `.ai-code-agent.yml`

Project: {config.project_name}
Triggers: {', '.join(config.analysis_triggers)}
"""