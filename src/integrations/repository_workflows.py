"""
Repository analysis workflows for automated code quality management.
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import schedule
import time

from src.agents.agent_coordinator import AgentCoordinator, WorkflowType
from src.integrations.github_integration import GitHubIntegration
from src.integrations.gitlab_integration import GitLabIntegration


@dataclass
class RepositoryConfig:
    """Configuration for repository monitoring."""
    name: str
    platform: str  # github, gitlab
    url: str
    branch: str = "main"
    analysis_schedule: str = "daily"  # daily, weekly, on_change
    workflow_types: List[WorkflowType] = field(default_factory=lambda: [WorkflowType.COMPREHENSIVE_ANALYSIS])
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)
    auto_fix_enabled: bool = False


@dataclass 
class AnalysisSchedule:
    """Scheduled analysis task."""
    repository: RepositoryConfig
    next_run: datetime
    last_run: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    results: Optional[Dict[str, Any]] = None


class RepositoryWorkflowManager:
    """Manages automated workflows for multiple repositories."""
    
    def __init__(self):
        self.coordinator = AgentCoordinator()
        self.github_integration: Optional[GitHubIntegration] = None
        self.gitlab_integration: Optional[GitLabIntegration] = None
        
        self.repositories: Dict[str, RepositoryConfig] = {}
        self.schedules: Dict[str, AnalysisSchedule] = {}
        self.running = False
    
    def add_repository(self, config: RepositoryConfig) -> bool:
        """Add a repository to monitoring."""
        try:
            # Validate repository access
            if config.platform == "github":
                if not self.github_integration:
                    self.github_integration = GitHubIntegration()
                # Test access
                repo = self.github_integration.github.get_repo(config.name)
                config.url = repo.clone_url
                
            elif config.platform == "gitlab":
                if not self.gitlab_integration:
                    self.gitlab_integration = GitLabIntegration()
                # Test access
                project = self.gitlab_integration.gitlab.projects.get(config.name)
                config.url = project.http_url_to_repo
            
            else:
                raise ValueError(f"Unsupported platform: {config.platform}")
            
            # Add to monitoring
            self.repositories[config.name] = config
            
            # Create schedule
            next_run = self._calculate_next_run(config.analysis_schedule)
            self.schedules[config.name] = AnalysisSchedule(
                repository=config,
                next_run=next_run
            )
            
            print(f"Added repository {config.name} to monitoring")
            return True
            
        except Exception as e:
            print(f"Failed to add repository {config.name}: {e}")
            return False
    
    def remove_repository(self, repo_name: str) -> bool:
        """Remove a repository from monitoring."""
        if repo_name in self.repositories:
            del self.repositories[repo_name]
            if repo_name in self.schedules:
                del self.schedules[repo_name]
            print(f"Removed repository {repo_name} from monitoring")
            return True
        return False
    
    async def analyze_repository_now(self, repo_name: str) -> Dict[str, Any]:
        """Trigger immediate analysis of a repository."""
        if repo_name not in self.repositories:
            return {"error": "Repository not found"}
        
        config = self.repositories[repo_name]
        
        try:
            # Update schedule status
            if repo_name in self.schedules:
                self.schedules[repo_name].status = "running"
            
            # Clone or update repository
            repo_path = await self._prepare_repository(config)
            
            if not repo_path:
                return {"error": "Failed to prepare repository"}
            
            # Run analysis workflows
            results = {}
            for workflow_type in config.workflow_types:
                print(f"Running {workflow_type.value} for {repo_name}")
                
                result = await self.coordinator.execute_workflow(
                    workflow_type,
                    self._get_target_files(repo_path)
                )
                
                results[workflow_type.value] = {
                    "success_rate": result.success_rate,
                    "execution_time_ms": result.execution_time_ms,
                    "summary": result.summary,
                    "recommendations": result.recommendations,
                    "files_analyzed": len(result.target_files)
                }
            
            # Check quality thresholds
            quality_alerts = self._check_quality_thresholds(results, config)
            
            # Generate report
            report = await self._generate_repository_report(config, results, quality_alerts)
            
            # Send notifications
            await self._send_notifications(config, results, quality_alerts)
            
            # Update schedule
            if repo_name in self.schedules:
                schedule = self.schedules[repo_name]
                schedule.status = "completed"
                schedule.last_run = datetime.now()
                schedule.next_run = self._calculate_next_run(config.analysis_schedule)
                schedule.results = results
            
            return {
                "repository": repo_name,
                "status": "completed",
                "workflows_executed": len(config.workflow_types),
                "results": results,
                "quality_alerts": quality_alerts,
                "report_generated": bool(report)
            }
            
        except Exception as e:
            # Update schedule status
            if repo_name in self.schedules:
                self.schedules[repo_name].status = "failed"
            
            return {
                "repository": repo_name,
                "status": "failed",
                "error": str(e)
            }
    
    async def start_scheduler(self):
        """Start the automated scheduler."""
        self.running = True
        print("Repository workflow scheduler started")
        
        while self.running:
            try:
                await self._check_scheduled_analyses()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    def stop_scheduler(self):
        """Stop the automated scheduler."""
        self.running = False
        print("Repository workflow scheduler stopped")
    
    async def _check_scheduled_analyses(self):
        """Check for scheduled analyses that need to run."""
        now = datetime.now()
        
        for repo_name, schedule in self.schedules.items():
            if (schedule.status == "pending" and 
                schedule.next_run <= now):
                
                print(f"Running scheduled analysis for {repo_name}")
                
                # Run analysis in background
                asyncio.create_task(self.analyze_repository_now(repo_name))
    
    async def _prepare_repository(self, config: RepositoryConfig) -> Optional[str]:
        """Clone or update repository for analysis."""
        import tempfile
        import subprocess
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix=f"repo_{config.name.replace('/', '_')}_")
            
            # Clone repository
            clone_cmd = [
                "git", "clone", 
                "--depth", "1",
                "--branch", config.branch,
                config.url,
                temp_dir
            ]
            
            result = subprocess.run(
                clone_cmd, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return temp_dir
            else:
                print(f"Git clone failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Repository preparation failed: {e}")
            return None
    
    def _get_target_files(self, repo_path: str) -> List[str]:
        """Get list of files to analyze in repository."""
        target_files = []
        repo_dir = Path(repo_path)
        
        # Supported file extensions
        extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}
        
        # Directories to exclude
        exclude_dirs = {'.git', 'node_modules', 'venv', '__pycache__', '.pytest_cache'}
        
        for file_path in repo_dir.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in extensions and
                not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs)):
                target_files.append(str(file_path))
        
        # Limit files for performance
        return target_files[:50]
    
    def _check_quality_thresholds(self, 
                                results: Dict[str, Any], 
                                config: RepositoryConfig) -> List[Dict[str, Any]]:
        """Check if results meet quality thresholds."""
        alerts = []
        
        for workflow_name, result in results.items():
            success_rate = result.get("success_rate", 0.0)
            
            # Check success rate threshold
            min_success_rate = config.quality_thresholds.get("min_success_rate", 0.8)
            if success_rate < min_success_rate:
                alerts.append({
                    "type": "quality_threshold",
                    "level": "warning",
                    "workflow": workflow_name,
                    "metric": "success_rate",
                    "value": success_rate,
                    "threshold": min_success_rate,
                    "message": f"Success rate {success_rate:.1%} below threshold {min_success_rate:.1%}"
                })
            
            # Check execution time threshold
            max_execution_time = config.quality_thresholds.get("max_execution_time_ms", 60000)
            execution_time = result.get("execution_time_ms", 0)
            if execution_time > max_execution_time:
                alerts.append({
                    "type": "performance",
                    "level": "info",
                    "workflow": workflow_name,
                    "metric": "execution_time",
                    "value": execution_time,
                    "threshold": max_execution_time,
                    "message": f"Analysis took {execution_time:.0f}ms (threshold: {max_execution_time:.0f}ms)"
                })
        
        return alerts
    
    async def _generate_repository_report(self, 
                                        config: RepositoryConfig,
                                        results: Dict[str, Any],
                                        alerts: List[Dict[str, Any]]) -> Optional[str]:
        """Generate comprehensive repository analysis report."""
        try:
            report_content = f"""# Repository Analysis Report: {config.name}

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Platform:** {config.platform}
**Branch:** {config.branch}

## Summary
"""
            
            # Add workflow results
            for workflow_name, result in results.items():
                report_content += f"""
### {workflow_name.replace('_', ' ').title()}
- **Success Rate:** {result['success_rate']:.1%}
- **Execution Time:** {result['execution_time_ms']:.0f}ms
- **Files Analyzed:** {result['files_analyzed']}
- **Summary:** {result['summary']}

**Recommendations:**
"""
                for rec in result.get('recommendations', []):
                    report_content += f"- {rec}\n"
            
            # Add quality alerts
            if alerts:
                report_content += "\n## Quality Alerts\n"
                for alert in alerts:
                    level_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
                    emoji = level_emoji.get(alert['level'], "")
                    report_content += f"- {emoji} **{alert['type'].title()}:** {alert['message']}\n"
            
            # Save report
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            report_filename = f"{config.name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            report_path = reports_dir / report_filename
            
            report_path.write_text(report_content)
            
            return str(report_path)
            
        except Exception as e:
            print(f"Report generation failed: {e}")
            return None
    
    async def _send_notifications(self, 
                                config: RepositoryConfig,
                                results: Dict[str, Any],
                                alerts: List[Dict[str, Any]]):
        """Send notifications about analysis results."""
        # This is a placeholder for notification implementation
        # In a real system, you would integrate with:
        # - Slack/Teams webhooks
        # - Email services
        # - GitHub/GitLab issue creation
        # - Custom webhook endpoints
        
        if not config.notification_channels:
            return
        
        # Create notification message
        message = f"Repository analysis completed for {config.name}"
        
        if alerts:
            critical_alerts = [a for a in alerts if a['level'] in ['error', 'warning']]
            if critical_alerts:
                message += f" with {len(critical_alerts)} quality alerts"
        
        print(f"Notification: {message}")
        
        # TODO: Implement actual notification sending
        # For each channel in config.notification_channels:
        #   - Parse channel type (slack, email, webhook, etc.)
        #   - Format message appropriately
        #   - Send notification
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next scheduled run time."""
        now = datetime.now()
        
        if schedule == "daily":
            return now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif schedule == "weekly":
            days_ahead = 6 - now.weekday()  # Sunday
            if days_ahead <= 0:
                days_ahead += 7
            return (now + timedelta(days=days_ahead)).replace(hour=2, minute=0, second=0, microsecond=0)
        elif schedule == "hourly":
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:  # on_change - set far future, will be triggered by webhooks
            return now + timedelta(days=365)
    
    def get_repository_status(self) -> Dict[str, Any]:
        """Get status of all monitored repositories."""
        status = {
            "total_repositories": len(self.repositories),
            "scheduler_running": self.running,
            "repositories": []
        }
        
        for repo_name, config in self.repositories.items():
            schedule = self.schedules.get(repo_name)
            
            repo_status = {
                "name": repo_name,
                "platform": config.platform,
                "branch": config.branch,
                "schedule": config.analysis_schedule,
                "status": schedule.status if schedule else "unknown",
                "last_run": schedule.last_run.isoformat() if schedule and schedule.last_run else None,
                "next_run": schedule.next_run.isoformat() if schedule else None,
                "workflow_types": [wf.value for wf in config.workflow_types]
            }
            
            status["repositories"].append(repo_status)
        
        return status
    
    def save_configuration(self, config_file: str = "repository_config.json"):
        """Save repository monitoring configuration."""
        config_data = {
            "repositories": []
        }
        
        for repo_name, repo_config in self.repositories.items():
            config_data["repositories"].append({
                "name": repo_config.name,
                "platform": repo_config.platform,
                "url": repo_config.url,
                "branch": repo_config.branch,
                "analysis_schedule": repo_config.analysis_schedule,
                "workflow_types": [wf.value for wf in repo_config.workflow_types],
                "quality_thresholds": repo_config.quality_thresholds,
                "notification_channels": repo_config.notification_channels,
                "auto_fix_enabled": repo_config.auto_fix_enabled
            })
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Configuration saved to {config_file}")
    
    def load_configuration(self, config_file: str = "repository_config.json"):
        """Load repository monitoring configuration."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for repo_data in config_data.get("repositories", []):
                workflow_types = [WorkflowType(wf) for wf in repo_data.get("workflow_types", ["comprehensive_analysis"])]
                
                config = RepositoryConfig(
                    name=repo_data["name"],
                    platform=repo_data["platform"],
                    url=repo_data["url"],
                    branch=repo_data.get("branch", "main"),
                    analysis_schedule=repo_data.get("analysis_schedule", "daily"),
                    workflow_types=workflow_types,
                    quality_thresholds=repo_data.get("quality_thresholds", {}),
                    notification_channels=repo_data.get("notification_channels", []),
                    auto_fix_enabled=repo_data.get("auto_fix_enabled", False)
                )
                
                self.add_repository(config)
            
            print(f"Loaded {len(self.repositories)} repositories from {config_file}")
            
        except FileNotFoundError:
            print(f"Configuration file {config_file} not found")
        except Exception as e:
            print(f"Failed to load configuration: {e}")