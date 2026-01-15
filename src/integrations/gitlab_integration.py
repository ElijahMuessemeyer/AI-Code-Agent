"""
GitLab integration for automated code analysis and MR interactions.
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
    import gitlab
    from gitlab.v4.objects import Project, ProjectMergeRequest, ProjectCommit
except ImportError:
    print("python-gitlab not installed. Run: pip install python-gitlab")
    gitlab = None

from src.agents.agent_coordinator import AgentCoordinator, WorkflowType
from src.agents.code_reviewer import ReviewResult
from src.agents.bug_detector import BugDetectionResult


@dataclass
class MRAnalysisResult:
    """Result of analyzing a merge request."""
    mr_iid: int
    project_id: int
    changed_files: List[str]
    review_results: List[ReviewResult]
    bug_results: List[BugDetectionResult]
    overall_score: float
    recommendation: str
    analysis_time_ms: float


class GitLabIntegration:
    """Handles GitLab API interactions and webhook processing."""
    
    def __init__(self, token: Optional[str] = None, webhook_secret: Optional[str] = None, url: Optional[str] = None):
        self.token = token or os.getenv('GITLAB_TOKEN')
        self.webhook_secret = webhook_secret or os.getenv('GITLAB_WEBHOOK_SECRET')
        self.url = url or os.getenv('GITLAB_URL', 'https://gitlab.com')
        
        if not self.token:
            raise ValueError("GitLab token is required. Set GITLAB_TOKEN environment variable.")
        
        if gitlab is None:
            raise ImportError("python-gitlab is required. Install with: pip install python-gitlab")
        
        self.gitlab = gitlab.Gitlab(self.url, private_token=self.token)
        self.coordinator = AgentCoordinator()
    
    def verify_webhook_signature(self, payload_body: str, signature_header: str) -> bool:
        """Verify GitLab webhook signature."""
        if not self.webhook_secret:
            return True  # Skip verification if no secret is set
        
        if not signature_header:
            return False
        
        # GitLab uses X-Gitlab-Token header for webhook verification
        return hmac.compare_digest(self.webhook_secret, signature_header)
    
    def parse_webhook_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a GitLab webhook event."""
        event_type = payload.get('object_kind', '')
        
        return {
            'event_type': event_type,
            'project_id': payload.get('project', {}).get('id'),
            'project_name': payload.get('project', {}).get('path_with_namespace'),
            'payload': payload,
            'timestamp': datetime.now()
        }
    
    async def handle_webhook_event(self, event_data: Dict[str, Any]) -> Optional[Any]:
        """Handle different types of GitLab webhook events."""
        try:
            event_type = event_data['event_type']
            
            if event_type == 'merge_request':
                return await self._handle_merge_request_event(event_data)
            elif event_type == 'push':
                return await self._handle_push_event(event_data)
            elif event_type == 'issue':
                return await self._handle_issue_event(event_data)
            else:
                print(f"Unhandled GitLab event type: {event_type}")
                return None
                
        except Exception as e:
            print(f"Error handling GitLab webhook event: {e}")
            return None
    
    async def _handle_merge_request_event(self, event_data: Dict[str, Any]) -> Optional[MRAnalysisResult]:
        """Handle merge request events."""
        payload = event_data['payload']
        
        # Only handle opened, updated MRs
        if payload.get('object_attributes', {}).get('action') not in ['open', 'update', 'reopen']:
            return None
        
        try:
            project_id = payload['project']['id']
            mr_iid = payload['object_attributes']['iid']
            
            # Get project and MR
            project = self.gitlab.projects.get(project_id)
            mr = project.mergerequests.get(mr_iid)
            
            # Analyze the MR
            result = await self.analyze_merge_request(project, mr)
            
            # Post analysis comment
            await self._post_mr_analysis_note(project, mr, result)
            
            return result
            
        except gitlab.GitlabError as e:
            print(f"GitLab API error: {e}")
            return None
    
    async def _handle_push_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle push events."""
        try:
            payload = event_data['payload']
            project_id = payload['project_id']
            commits = payload.get('commits', [])
            
            project = self.gitlab.projects.get(project_id)
            
            # Analyze recent commits
            analysis_results = []
            for commit_data in commits[-3:]:  # Analyze last 3 commits
                commit_id = commit_data['id']
                commit = project.commits.get(commit_id)
                
                # Get commit changes
                commit_diff = commit.diff()
                changed_files = []
                
                for diff in commit_diff:
                    if diff['new_path'].endswith(('.py', '.js', '.ts', '.java')):
                        changed_files.append(diff['new_path'])
                
                if changed_files:
                    result = await self._analyze_commit_files(project, commit, changed_files)
                    analysis_results.append(result)
            
            return {'commit_analyses': analysis_results}
            
        except gitlab.GitlabError as e:
            print(f"GitLab API error: {e}")
            return None
    
    async def _handle_issue_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle issue events."""
        payload = event_data['payload']
        
        if payload.get('object_attributes', {}).get('action') != 'open':
            return None
        
        try:
            project_id = payload['project']['id']
            issue_iid = payload['object_attributes']['iid']
            
            project = self.gitlab.projects.get(project_id)
            issue = project.issues.get(issue_iid)
            
            # Check if issue is requesting code generation
            title = issue.title.lower()
            if any(keyword in title for keyword in ['generate', 'create', 'implement']):
                suggestion = await self._generate_code_from_issue(issue)
                
                if suggestion:
                    await self._post_issue_code_suggestion(project, issue, suggestion)
                    return {'code_suggestion': suggestion}
            
            return None
            
        except gitlab.GitlabError as e:
            print(f"GitLab API error: {e}")
            return None
    
    async def analyze_merge_request(self, project: Project, mr: ProjectMergeRequest) -> MRAnalysisResult:
        """Perform comprehensive analysis of a merge request."""
        import time
        start_time = time.time()
        
        # Get MR changes
        changes = mr.changes()
        changed_files = []
        
        for change in changes.get('changes', []):
            file_path = change.get('new_path') or change.get('old_path')
            if file_path and file_path.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs')):
                changed_files.append(file_path)
        
        if not changed_files:
            return MRAnalysisResult(
                mr_iid=mr.iid,
                project_id=project.id,
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
                # Get file content from MR
                file_content = self._get_file_content_from_mr(project, mr, file_path)
                if file_content:
                    # Create temporary file for analysis
                    temp_file_path = f"/tmp/mr_analysis_{mr.iid}_{file_path.replace('/', '_')}"
                    with open(temp_file_path, 'w', encoding='utf-8') as f:
                        f.write(file_content)
                    
                    # Run analysis
                    review_result = await self.coordinator.code_reviewer.review_file(temp_file_path)
                    if review_result:
                        review_result.file_path = file_path
                        review_results.append(review_result)
                    
                    bug_result = await self.coordinator.bug_detector.detect_bugs(temp_file_path)
                    if bug_result:
                        bug_result.file_path = file_path
                        bug_results.append(bug_result)
                    
                    # Cleanup
                    os.unlink(temp_file_path)
                    
            except Exception as e:
                print(f"Error analyzing file {file_path}: {e}")
                continue
        
        # Calculate overall score and recommendation
        overall_score, recommendation = self._calculate_mr_assessment(review_results, bug_results)
        
        analysis_time = (time.time() - start_time) * 1000
        
        return MRAnalysisResult(
            mr_iid=mr.iid,
            project_id=project.id,
            changed_files=changed_files,
            review_results=review_results,
            bug_results=bug_results,
            overall_score=overall_score,
            recommendation=recommendation,
            analysis_time_ms=analysis_time
        )
    
    def _get_file_content_from_mr(self, project: Project, mr: ProjectMergeRequest, file_path: str) -> Optional[str]:
        """Get file content from MR source branch."""
        try:
            file_obj = project.files.get(file_path, ref=mr.source_branch)
            return file_obj.decode().decode('utf-8')
        except:
            return None
    
    def _calculate_mr_assessment(self, 
                                review_results: List[ReviewResult], 
                                bug_results: List[BugDetectionResult]) -> Tuple[float, str]:
        """Calculate overall MR assessment."""
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
    
    async def _post_mr_analysis_note(self, project: Project, mr: ProjectMergeRequest, result: MRAnalysisResult):
        """Post analysis results as an MR note."""
        try:
            note_body = self._generate_mr_note_body(result)
            
            # Check if we already posted a note (to avoid spam)
            existing_notes = mr.notes.list()
            bot_notes = [n for n in existing_notes if "🤖 AI Code Agent Analysis" in n.body]
            
            if bot_notes:
                # Update existing note
                bot_notes[-1].body = note_body
                bot_notes[-1].save()
            else:
                # Create new note
                mr.notes.create({'body': note_body})
                
        except gitlab.GitlabError as e:
            print(f"Error posting MR note: {e}")
    
    def _generate_mr_note_body(self, result: MRAnalysisResult) -> str:
        """Generate MR note body with analysis results."""
        note = f"""🤖 **AI Code Agent Analysis**

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
            
            note += f"""## Code Review Summary
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
            
            note += f"""## Bug Detection Summary
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
            note += "## ⚠️ Critical Issues\n"
            for finding in critical_findings[:5]:  # Limit to 5 for readability
                note += f"- {finding}\n"
            
            if len(critical_findings) > 5:
                note += f"\n*... and {len(critical_findings) - 5} more issues*\n"
        
        note += f"""
---
*Generated by AI Code Agent v1.0*
"""
        
        return note
    
    async def _analyze_commit_files(self, project: Project, commit: ProjectCommit, changed_files: List[str]) -> Dict[str, Any]:
        """Analyze files changed in a commit."""
        analysis_results = []
        
        for file_path in changed_files[:5]:  # Limit for performance
            try:
                file_obj = project.files.get(file_path, ref=commit.id)
                file_content = file_obj.decode().decode('utf-8')
                
                # Create temporary file
                temp_file_path = f"/tmp/commit_analysis_{commit.short_id}_{file_path.replace('/', '_')}"
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
            'commit_id': commit.id,
            'files_analyzed': analysis_results,
            'total_files': len(changed_files)
        }
    
    async def _generate_code_from_issue(self, issue) -> Optional[str]:
        """Generate code suggestions based on issue description."""
        try:
            from src.agents.code_generator import CodeRequirement, CodeType, CodeQuality
            
            # Simple heuristics to determine what to generate
            title = issue.title.lower()
            description = issue.description.lower() if issue.description else ""
            
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
            if 'javascript' in description or 'js' in description:
                language = "javascript"
            elif 'java' in description and 'javascript' not in description:
                language = "java"
            
            # Create requirement
            requirement = CodeRequirement(
                description=f"{issue.title}\n\n{issue.description or ''}",
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
    
    async def _post_issue_code_suggestion(self, project: Project, issue, code_suggestion: str):
        """Post code suggestion as issue note."""
        try:
            note_body = f"""🤖 **AI Code Suggestion**

Based on your issue description, here's a code suggestion:

```python
{code_suggestion}
```

*This code was automatically generated by AI Code Agent. Please review and test before using.*
"""
            
            issue.notes.create({'body': note_body})
            
        except gitlab.GitlabError as e:
            print(f"Error posting issue note: {e}")
    
    async def setup_project_webhooks(self, project_id: int, webhook_url: str) -> bool:
        """Set up webhooks for a GitLab project."""
        try:
            project = self.gitlab.projects.get(project_id)
            
            # Check if webhook already exists
            existing_hooks = project.hooks.list()
            for hook in existing_hooks:
                if hook.url == webhook_url:
                    print(f"Webhook already exists for project {project_id}")
                    return True
            
            # Create new webhook
            hook_data = {
                'url': webhook_url,
                'merge_requests_events': True,
                'push_events': True,
                'issues_events': True,
                'note_events': True,
                'enable_ssl_verification': True
            }
            
            if self.webhook_secret:
                hook_data['token'] = self.webhook_secret
            
            project.hooks.create(hook_data)
            
            print(f"Webhook created successfully for project {project_id}")
            return True
            
        except gitlab.GitlabError as e:
            print(f"Error setting up GitLab webhook: {e}")
            return False
    
    async def analyze_project(self, project_id: int, max_files: int = 20) -> Dict[str, Any]:
        """Perform comprehensive analysis of a GitLab project."""
        try:
            project = self.gitlab.projects.get(project_id)
            
            # Get repository tree
            tree = project.repository_tree(recursive=True, per_page=100)
            
            # Filter for analyzable files
            analyzable_files = []
            for item in tree:
                if (item['type'] == 'blob' and 
                    item['name'].endswith(('.py', '.js', '.ts', '.java')) and
                    len(analyzable_files) < max_files):
                    analyzable_files.append(item['path'])
            
            if not analyzable_files:
                return {
                    'project_id': project_id,
                    'error': 'No analyzable files found'
                }
            
            # Run comprehensive analysis
            result = await self.coordinator.execute_workflow(
                WorkflowType.COMPREHENSIVE_ANALYSIS,
                analyzable_files[:10]  # Limit for demo
            )
            
            return {
                'project_id': project_id,
                'project_name': project.path_with_namespace,
                'files_analyzed': len(analyzable_files),
                'workflow_result': {
                    'success_rate': result.success_rate,
                    'execution_time_ms': result.execution_time_ms,
                    'summary': result.summary,
                    'recommendations': result.recommendations
                }
            }
            
        except gitlab.GitlabError as e:
            return {'error': f'GitLab API error: {e}'}
        except Exception as e:
            return {'error': f'Analysis error: {e}'}
    
    def get_project_info(self, project_id: int) -> Dict[str, Any]:
        """Get basic project information."""
        try:
            project = self.gitlab.projects.get(project_id)
            return {
                'id': project.id,
                'name': project.name,
                'path': project.path,
                'namespace': project.namespace['name'],
                'url': project.web_url,
                'default_branch': project.default_branch,
                'languages': getattr(project, 'languages', {})
            }
        except gitlab.GitlabError as e:
            return {'error': f'GitLab API error: {e}'}