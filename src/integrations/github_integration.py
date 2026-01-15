"""
GitHub integration for automated code analysis and PR interactions.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac

try:
    from github import Github, GithubException
    from github.Repository import Repository
    from github.PullRequest import PullRequest
    from github.Commit import Commit
except ImportError:
    print("PyGithub not installed. Run: pip install PyGithub")
    Github = None

from src.agents.agent_coordinator import AgentCoordinator, WorkflowType
from src.agents.code_reviewer import ReviewResult
from src.agents.bug_detector import BugDetectionResult


@dataclass
class PRAnalysisResult:
    """Result of analyzing a pull request."""
    pr_number: int
    repository: str
    changed_files: List[str]
    review_results: List[ReviewResult]
    bug_results: List[BugDetectionResult]
    overall_score: float
    recommendation: str
    analysis_time_ms: float


@dataclass
class WebhookEvent:
    """Represents a webhook event from GitHub."""
    event_type: str
    repository: str
    pr_number: Optional[int]
    action: str
    payload: Dict[str, Any]
    timestamp: datetime


class GitHubIntegration:
    """Handles GitHub API interactions and webhook processing."""
    
    def __init__(self, token: Optional[str] = None, webhook_secret: Optional[str] = None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.webhook_secret = webhook_secret or os.getenv('GITHUB_WEBHOOK_SECRET')
        
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable.")
        
        if Github is None:
            raise ImportError("PyGithub is required. Install with: pip install PyGithub")
        
        self.github = Github(self.token)
        self.coordinator = AgentCoordinator()
        
        # Rate limiting
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = datetime.now()
    
    def verify_webhook_signature(self, payload_body: bytes, signature_header: str) -> bool:
        """Verify GitHub webhook signature."""
        if not self.webhook_secret:
            return True  # Skip verification if no secret is set
        
        if not signature_header:
            return False
        
        hash_object = hmac.new(
            self.webhook_secret.encode('utf-8'),
            msg=payload_body,
            digestmod=hashlib.sha256
        )
        expected_signature = "sha256=" + hash_object.hexdigest()
        
        return hmac.compare_digest(expected_signature, signature_header)
    
    def parse_webhook_event(self, payload: Dict[str, Any], event_type: str) -> WebhookEvent:
        """Parse a GitHub webhook event."""
        repository = payload.get('repository', {}).get('full_name', '')
        action = payload.get('action', '')
        pr_number = None
        
        if 'pull_request' in payload:
            pr_number = payload['pull_request']['number']
        elif 'number' in payload:
            pr_number = payload['number']
        
        return WebhookEvent(
            event_type=event_type,
            repository=repository,
            pr_number=pr_number,
            action=action,
            payload=payload,
            timestamp=datetime.now()
        )
    
    async def handle_webhook_event(self, event: WebhookEvent) -> Optional[Any]:
        """Handle different types of webhook events."""
        try:
            if event.event_type == 'pull_request':
                return await self._handle_pull_request_event(event)
            elif event.event_type == 'push':
                return await self._handle_push_event(event)
            elif event.event_type == 'issues':
                return await self._handle_issue_event(event)
            else:
                print(f"Unhandled event type: {event.event_type}")
                return None
                
        except Exception as e:
            print(f"Error handling webhook event: {e}")
            return None
    
    async def _handle_pull_request_event(self, event: WebhookEvent) -> Optional[PRAnalysisResult]:
        """Handle pull request events."""
        if event.action not in ['opened', 'synchronize', 'reopened']:
            return None
        
        try:
            # Get repository and PR
            repo = self.github.get_repo(event.repository)
            pr = repo.get_pull(event.pr_number)
            
            # Analyze the PR
            result = await self.analyze_pull_request(repo, pr)
            
            # Post analysis comment
            await self._post_pr_analysis_comment(pr, result)
            
            return result
            
        except GithubException as e:
            print(f"GitHub API error: {e}")
            return None
    
    async def _handle_push_event(self, event: WebhookEvent) -> Optional[Dict[str, Any]]:
        """Handle push events."""
        try:
            repo = self.github.get_repo(event.repository)
            commits = event.payload.get('commits', [])
            
            # Analyze recent commits
            analysis_results = []
            for commit_data in commits[-3:]:  # Analyze last 3 commits
                commit_sha = commit_data['id']
                commit = repo.get_commit(commit_sha)
                
                # Get changed files
                changed_files = [f.filename for f in commit.files if f.filename.endswith(('.py', '.js', '.ts', '.java'))]
                
                if changed_files:
                    # Run quick analysis on changed files
                    result = await self._analyze_commit_files(repo, commit, changed_files)
                    analysis_results.append(result)
            
            return {'commit_analyses': analysis_results}
            
        except GithubException as e:
            print(f"GitHub API error: {e}")
            return None
    
    async def _handle_issue_event(self, event: WebhookEvent) -> Optional[Dict[str, Any]]:
        """Handle issue events."""
        if event.action != 'opened':
            return None
        
        try:
            repo = self.github.get_repo(event.repository)
            issue = repo.get_issue(event.payload['issue']['number'])
            
            # Check if issue is requesting code generation
            if any(keyword in issue.title.lower() for keyword in ['generate', 'create', 'implement']):
                # Auto-generate code based on issue description
                suggestion = await self._generate_code_from_issue(issue)
                
                if suggestion:
                    await self._post_issue_code_suggestion(issue, suggestion)
                    return {'code_suggestion': suggestion}
            
            return None
            
        except GithubException as e:
            print(f"GitHub API error: {e}")
            return None
    
    async def analyze_pull_request(self, repo: Repository, pr: PullRequest) -> PRAnalysisResult:
        """Perform comprehensive analysis of a pull request."""
        import time
        start_time = time.time()
        
        # Get changed files
        changed_files = []
        for file in pr.get_files():
            if file.filename.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs')):
                changed_files.append(file.filename)
        
        if not changed_files:
            return PRAnalysisResult(
                pr_number=pr.number,
                repository=repo.full_name,
                changed_files=[],
                review_results=[],
                bug_results=[],
                overall_score=10.0,
                recommendation="No code changes to analyze.",
                analysis_time_ms=0.0
            )
        
        # Download and analyze changed files
        review_results = []
        bug_results = []
        
        for file_path in changed_files[:10]:  # Limit to 10 files for performance
            try:
                # Get file content from PR
                file_content = self._get_file_content_from_pr(repo, pr, file_path)
                if file_content:
                    # Create temporary file for analysis
                    temp_file_path = f"/tmp/pr_analysis_{pr.number}_{file_path.replace('/', '_')}"
                    with open(temp_file_path, 'w', encoding='utf-8') as f:
                        f.write(file_content)
                    
                    # Run analysis
                    review_result = await self.coordinator.code_reviewer.review_file(temp_file_path)
                    if review_result:
                        review_result.file_path = file_path  # Update to PR file path
                        review_results.append(review_result)
                    
                    bug_result = await self.coordinator.bug_detector.detect_bugs(temp_file_path)
                    if bug_result:
                        bug_result.file_path = file_path  # Update to PR file path
                        bug_results.append(bug_result)
                    
                    # Cleanup
                    os.unlink(temp_file_path)
                    
            except Exception as e:
                print(f"Error analyzing file {file_path}: {e}")
                continue
        
        # Calculate overall score and recommendation
        overall_score, recommendation = self._calculate_pr_assessment(review_results, bug_results)
        
        analysis_time = (time.time() - start_time) * 1000
        
        return PRAnalysisResult(
            pr_number=pr.number,
            repository=repo.full_name,
            changed_files=changed_files,
            review_results=review_results,
            bug_results=bug_results,
            overall_score=overall_score,
            recommendation=recommendation,
            analysis_time_ms=analysis_time
        )
    
    def _get_file_content_from_pr(self, repo: Repository, pr: PullRequest, file_path: str) -> Optional[str]:
        """Get file content from PR head commit."""
        try:
            # Get the file from the PR head commit
            file_obj = repo.get_contents(file_path, ref=pr.head.sha)
            if hasattr(file_obj, 'decoded_content'):
                return file_obj.decoded_content.decode('utf-8')
            return None
        except:
            return None
    
    def _calculate_pr_assessment(self, 
                                review_results: List[ReviewResult], 
                                bug_results: List[BugDetectionResult]) -> Tuple[float, str]:
        """Calculate overall PR assessment."""
        if not review_results and not bug_results:
            return 8.0, "✅ No significant issues found."
        
        # Calculate average review score
        avg_review_score = 8.0
        if review_results:
            scores = [r.ai_feedback.overall_score for r in review_results]
            avg_review_score = sum(scores) / len(scores)
        
        # Count critical issues
        critical_bugs = 0
        total_bugs = 0
        for bug_result in bug_results:
            total_bugs += len(bug_result.bugs_detected)
            critical_bugs += sum(1 for bug in bug_result.bugs_detected 
                               if bug.severity.value in ['critical', 'high'])
        
        # Determine overall score
        overall_score = avg_review_score
        
        if critical_bugs > 0:
            overall_score = min(overall_score, 5.0)
        elif total_bugs > 5:
            overall_score = min(overall_score, 6.5)
        
        # Generate recommendation
        if overall_score >= 8.0:
            recommendation = "✅ **Looks good!** Code quality is excellent. Ready to merge."
        elif overall_score >= 7.0:
            recommendation = "⚠️ **Minor issues found.** Consider addressing feedback before merging."
        elif overall_score >= 5.0:
            recommendation = "🔴 **Significant issues detected.** Please review and fix before merging."
        else:
            recommendation = "🚨 **Critical issues found!** Do not merge until all critical issues are resolved."
        
        return overall_score, recommendation
    
    async def _post_pr_analysis_comment(self, pr: PullRequest, result: PRAnalysisResult):
        """Post analysis results as a PR comment."""
        try:
            comment_body = self._generate_pr_comment_body(result)
            
            # Check if we already posted a comment (to avoid spam)
            existing_comments = list(pr.get_issue_comments())
            bot_comments = [c for c in existing_comments if "🤖 AI Code Agent Analysis" in c.body]
            
            if bot_comments:
                # Update existing comment
                bot_comments[-1].edit(comment_body)
            else:
                # Create new comment
                pr.create_issue_comment(comment_body)
                
        except GithubException as e:
            print(f"Error posting PR comment: {e}")
    
    def _generate_pr_comment_body(self, result: PRAnalysisResult) -> str:
        """Generate PR comment body with analysis results."""
        comment = f"""🤖 **AI Code Agent Analysis**

## Overall Assessment
**Score:** {result.overall_score:.1f}/10
{result.recommendation}

**Files Analyzed:** {len(result.changed_files)}
**Analysis Time:** {result.analysis_time_ms:.0f}ms

"""
        
        # Add review summary
        if result.review_results:
            total_issues = sum(len(r.ai_feedback.issues) for r in result.review_results)
            security_concerns = sum(len(r.ai_feedback.security_concerns) for r in result.review_results)
            
            comment += f"""## Code Review Summary
- **Total Issues:** {total_issues}
- **Security Concerns:** {security_concerns}
- **Files Reviewed:** {len(result.review_results)}

"""
        
        # Add bug detection summary
        if result.bug_results:
            total_bugs = sum(len(r.bugs_detected) for r in result.bug_results)
            critical_bugs = sum(
                sum(1 for bug in r.bugs_detected if bug.severity.value in ['critical', 'high'])
                for r in result.bug_results
            )
            
            comment += f"""## Bug Detection Summary
- **Total Bugs Found:** {total_bugs}
- **Critical/High Severity:** {critical_bugs}

"""
        
        # Add detailed findings for critical issues
        critical_findings = []
        
        for review_result in result.review_results:
            for issue in review_result.ai_feedback.issues:
                if issue.get('severity') in ['high', 'critical']:
                    critical_findings.append(f"**{review_result.file_path}**: {issue.get('description', 'Unknown issue')}")
        
        for bug_result in result.bug_results:
            for bug in bug_result.bugs_detected:
                if bug.severity.value in ['critical', 'high']:
                    critical_findings.append(f"**{bug_result.file_path}:L{bug.line_number}**: {bug.description}")
        
        if critical_findings:
            comment += "## ⚠️ Critical Issues\n"
            for finding in critical_findings[:5]:  # Limit to 5 for readability
                comment += f"- {finding}\n"
            
            if len(critical_findings) > 5:
                comment += f"\n*... and {len(critical_findings) - 5} more issues*\n"
        
        comment += f"""
---
*Generated by AI Code Agent v1.0 | [Report an issue](https://github.com/your-repo/ai-code-agent/issues)*
"""
        
        return comment
    
    async def _analyze_commit_files(self, repo: Repository, commit: Commit, changed_files: List[str]) -> Dict[str, Any]:
        """Analyze files changed in a commit."""
        analysis_results = []
        
        for file_path in changed_files[:5]:  # Limit for performance
            try:
                file_content = repo.get_contents(file_path, ref=commit.sha).decoded_content.decode('utf-8')
                
                # Create temporary file
                temp_file_path = f"/tmp/commit_analysis_{commit.sha}_{file_path.replace('/', '_')}"
                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                
                # Quick bug scan
                bug_result = await self.coordinator.bug_detector.detect_bugs(temp_file_path)
                
                if bug_result and bug_result.bugs_detected:
                    analysis_results.append({
                        'file': file_path,
                        'bugs_found': len(bug_result.bugs_detected),
                        'critical_bugs': sum(1 for bug in bug_result.bugs_detected 
                                           if bug.severity.value == 'critical')
                    })
                
                # Cleanup
                os.unlink(temp_file_path)
                
            except Exception as e:
                print(f"Error analyzing commit file {file_path}: {e}")
                continue
        
        return {
            'commit_sha': commit.sha,
            'files_analyzed': analysis_results,
            'total_files': len(changed_files)
        }
    
    async def _generate_code_from_issue(self, issue) -> Optional[str]:
        """Generate code suggestions based on issue description."""
        try:
            from src.agents.code_generator import CodeRequirement, CodeType, CodeQuality
            
            # Simple heuristics to determine what to generate
            title = issue.title.lower()
            body = issue.body.lower() if issue.body else ""
            
            # Determine code type and language
            if 'function' in title or 'method' in title:
                code_type = CodeType.FUNCTION
            elif 'class' in title:
                code_type = CodeType.CLASS
            elif 'api' in title or 'endpoint' in title:
                code_type = CodeType.API_ENDPOINT
            else:
                code_type = CodeType.FUNCTION
            
            # Determine language (default to Python)
            language = "python"
            if 'javascript' in body or 'js' in body:
                language = "javascript"
            elif 'java' in body and 'javascript' not in body:
                language = "java"
            
            # Create requirement
            requirement = CodeRequirement(
                description=f"{issue.title}\n\n{issue.body or ''}",
                language=language,
                code_type=code_type,
                quality_level=CodeQuality.PRODUCTION
            )
            
            # Generate code
            result = await self.coordinator.code_generator.generate_code(requirement)
            
            if result.confidence > 0.7:
                return result.generated_code.code
            
            return None
            
        except Exception as e:
            print(f"Error generating code from issue: {e}")
            return None
    
    async def _post_issue_code_suggestion(self, issue, code_suggestion: str):
        """Post code suggestion as issue comment."""
        try:
            comment_body = f"""🤖 **AI Code Suggestion**

Based on your issue description, here's a code suggestion:

```python
{code_suggestion}
```

*This code was automatically generated by AI Code Agent. Please review and test before using.*
"""
            
            issue.create_comment(comment_body)
            
        except GithubException as e:
            print(f"Error posting issue comment: {e}")
    
    async def setup_repository_webhooks(self, repo_name: str, webhook_url: str) -> bool:
        """Set up webhooks for a repository."""
        try:
            repo = self.github.get_repo(repo_name)
            
            # Check if webhook already exists
            existing_hooks = repo.get_hooks()
            for hook in existing_hooks:
                if hook.config.get('url') == webhook_url:
                    print(f"Webhook already exists for {repo_name}")
                    return True
            
            # Create new webhook
            config = {
                'url': webhook_url,
                'content_type': 'json',
                'insecure_ssl': '0'
            }
            
            if self.webhook_secret:
                config['secret'] = self.webhook_secret
            
            repo.create_hook(
                name='web',
                config=config,
                events=['pull_request', 'push', 'issues'],
                active=True
            )
            
            print(f"Webhook created successfully for {repo_name}")
            return True
            
        except GithubException as e:
            print(f"Error setting up webhook: {e}")
            return False
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information."""
        try:
            rate_limit = self.github.get_rate_limit()
            return {
                'core': {
                    'limit': rate_limit.core.limit,
                    'remaining': rate_limit.core.remaining,
                    'reset': rate_limit.core.reset.isoformat()
                },
                'search': {
                    'limit': rate_limit.search.limit,
                    'remaining': rate_limit.search.remaining,
                    'reset': rate_limit.search.reset.isoformat()
                }
            }
        except GithubException:
            return {'error': 'Unable to fetch rate limit info'}
    
    async def analyze_repository(self, repo_name: str, max_files: int = 20) -> Dict[str, Any]:
        """Perform comprehensive analysis of a repository."""
        try:
            repo = self.github.get_repo(repo_name)
            
            # Get all Python/JS files (limit for performance)
            contents = repo.get_contents("")
            analyzable_files = []
            
            def collect_files(contents, max_files):
                files = []
                for content_file in contents:
                    if len(files) >= max_files:
                        break
                    
                    if content_file.type == "dir":
                        try:
                            subcontents = repo.get_contents(content_file.path)
                            files.extend(collect_files(subcontents, max_files - len(files)))
                        except:
                            continue
                    elif content_file.name.endswith(('.py', '.js', '.ts', '.java')):
                        files.append(content_file.path)
                
                return files
            
            analyzable_files = collect_files(contents, max_files)
            
            # Run comprehensive analysis
            result = await self.coordinator.execute_workflow(
                WorkflowType.COMPREHENSIVE_ANALYSIS,
                analyzable_files[:10]  # Further limit for demo
            )
            
            return {
                'repository': repo_name,
                'files_analyzed': len(analyzable_files),
                'workflow_result': {
                    'success_rate': result.success_rate,
                    'execution_time_ms': result.execution_time_ms,
                    'summary': result.summary,
                    'recommendations': result.recommendations
                }
            }
            
        except GithubException as e:
            return {'error': f'GitHub API error: {e}'}
        except Exception as e:
            return {'error': f'Analysis error: {e}'}