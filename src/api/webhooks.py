"""
Webhook endpoints for GitHub and GitLab integration.
"""

from fastapi import APIRouter, HTTPException, Header, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import json
import asyncio
from datetime import datetime

from src.integrations.github_integration import GitHubIntegration
from src.integrations.gitlab_integration import GitLabIntegration


router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Global integration instances
github_integration: Optional[GitHubIntegration] = None
gitlab_integration: Optional[GitLabIntegration] = None


def get_github_integration() -> GitHubIntegration:
    """Get or create GitHub integration instance."""
    global github_integration
    if github_integration is None:
        try:
            github_integration = GitHubIntegration()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GitHub integration error: {e}")
    return github_integration


def get_gitlab_integration() -> GitLabIntegration:
    """Get or create GitLab integration instance."""
    global gitlab_integration
    if gitlab_integration is None:
        try:
            gitlab_integration = GitLabIntegration()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GitLab integration error: {e}")
    return gitlab_integration


@router.post("/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: Optional[str] = Header(None),
    x_hub_signature_256: Optional[str] = Header(None)
):
    """Handle GitHub webhook events."""
    try:
        # Get request body
        payload_bytes = await request.body()
        payload = await request.json()
        
        # Get GitHub integration
        gh_integration = get_github_integration()
        
        # Verify signature
        if not gh_integration.verify_webhook_signature(payload_bytes, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse event
        event = gh_integration.parse_webhook_event(payload, x_github_event)
        
        # Log event
        print(f"GitHub webhook: {event.event_type} for {event.repository}")
        
        # Handle event in background
        background_tasks.add_task(
            _handle_github_event_background,
            gh_integration,
            event
        )
        
        return {
            "status": "received",
            "event_type": event.event_type,
            "repository": event.repository,
            "pr_number": event.pr_number,
            "timestamp": event.timestamp.isoformat()
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook processing error: {e}")


@router.post("/gitlab")
async def gitlab_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_gitlab_event: Optional[str] = Header(None),
    x_gitlab_token: Optional[str] = Header(None)
):
    """Handle GitLab webhook events."""
    try:
        # Get request body
        payload_body = await request.body()
        payload = await request.json()
        
        # Get GitLab integration
        gl_integration = get_gitlab_integration()
        
        # Verify token
        if not gl_integration.verify_webhook_signature(payload_body.decode(), x_gitlab_token):
            raise HTTPException(status_code=401, detail="Invalid webhook token")
        
        # Parse event
        event_data = gl_integration.parse_webhook_event(payload)
        
        # Log event
        print(f"GitLab webhook: {event_data['event_type']} for project {event_data['project_id']}")
        
        # Handle event in background
        background_tasks.add_task(
            _handle_gitlab_event_background,
            gl_integration,
            event_data
        )
        
        return {
            "status": "received",
            "event_type": event_data['event_type'],
            "project_id": event_data['project_id'],
            "project_name": event_data['project_name'],
            "timestamp": event_data['timestamp'].isoformat()
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook processing error: {e}")


async def _handle_github_event_background(integration: GitHubIntegration, event):
    """Handle GitHub event in background task."""
    try:
        result = await integration.handle_webhook_event(event)
        
        if result:
            print(f"GitHub event processed successfully: {type(result).__name__}")
            
            # Log analysis results
            if hasattr(result, 'overall_score'):
                print(f"  - Overall score: {result.overall_score:.1f}/10")
                print(f"  - Files analyzed: {len(result.changed_files)}")
                
        else:
            print("GitHub event processed (no analysis performed)")
            
    except Exception as e:
        print(f"Error processing GitHub event: {e}")


async def _handle_gitlab_event_background(integration: GitLabIntegration, event_data):
    """Handle GitLab event in background task."""
    try:
        result = await integration.handle_webhook_event(event_data)
        
        if result:
            print(f"GitLab event processed successfully: {type(result).__name__}")
            
            # Log analysis results
            if hasattr(result, 'overall_score'):
                print(f"  - Overall score: {result.overall_score:.1f}/10")
                print(f"  - Files analyzed: {len(result.changed_files)}")
                
        else:
            print("GitLab event processed (no analysis performed)")
            
    except Exception as e:
        print(f"Error processing GitLab event: {e}")


@router.get("/github/status")
async def github_status():
    """Get GitHub integration status."""
    try:
        gh_integration = get_github_integration()
        rate_limit = gh_integration.get_rate_limit_info()
        
        return {
            "status": "active",
            "rate_limit": rate_limit,
            "webhook_secret_configured": bool(gh_integration.webhook_secret)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/gitlab/status")
async def gitlab_status():
    """Get GitLab integration status."""
    try:
        gl_integration = get_gitlab_integration()
        
        return {
            "status": "active",
            "gitlab_url": gl_integration.url,
            "webhook_secret_configured": bool(gl_integration.webhook_secret)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/github/setup/{repo_name:path}")
async def setup_github_webhook(repo_name: str, webhook_url: str):
    """Set up GitHub webhook for a repository."""
    try:
        gh_integration = get_github_integration()
        success = await gh_integration.setup_repository_webhooks(repo_name, webhook_url)
        
        if success:
            return {
                "status": "success",
                "message": f"Webhook configured for {repo_name}",
                "webhook_url": webhook_url
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to set up webhook")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gitlab/setup/{project_id}")
async def setup_gitlab_webhook(project_id: int, webhook_url: str):
    """Set up GitLab webhook for a project."""
    try:
        gl_integration = get_gitlab_integration()
        success = await gl_integration.setup_project_webhooks(project_id, webhook_url)
        
        if success:
            return {
                "status": "success",
                "message": f"Webhook configured for project {project_id}",
                "webhook_url": webhook_url
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to set up webhook")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/github/analyze/{repo_name:path}")
async def analyze_github_repository(
    repo_name: str,
    background_tasks: BackgroundTasks,
    max_files: int = 20
):
    """Trigger manual analysis of a GitHub repository."""
    try:
        gh_integration = get_github_integration()
        
        # Start analysis in background
        background_tasks.add_task(
            _analyze_github_repo_background,
            gh_integration,
            repo_name,
            max_files
        )
        
        return {
            "status": "started",
            "message": f"Analysis started for {repo_name}",
            "max_files": max_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gitlab/analyze/{project_id}")
async def analyze_gitlab_project(
    project_id: int,
    background_tasks: BackgroundTasks,
    max_files: int = 20
):
    """Trigger manual analysis of a GitLab project."""
    try:
        gl_integration = get_gitlab_integration()
        
        # Start analysis in background
        background_tasks.add_task(
            _analyze_gitlab_project_background,
            gl_integration,
            project_id,
            max_files
        )
        
        return {
            "status": "started",
            "message": f"Analysis started for project {project_id}",
            "max_files": max_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _analyze_github_repo_background(integration: GitHubIntegration, repo_name: str, max_files: int):
    """Analyze GitHub repository in background."""
    try:
        print(f"Starting analysis of GitHub repository: {repo_name}")
        result = await integration.analyze_repository(repo_name, max_files)
        
        if 'error' in result:
            print(f"Repository analysis failed: {result['error']}")
        else:
            print(f"Repository analysis completed:")
            print(f"  - Files analyzed: {result['files_analyzed']}")
            print(f"  - Success rate: {result['workflow_result']['success_rate']:.1%}")
            print(f"  - Execution time: {result['workflow_result']['execution_time_ms']:.0f}ms")
            
    except Exception as e:
        print(f"Error analyzing GitHub repository {repo_name}: {e}")


async def _analyze_gitlab_project_background(integration: GitLabIntegration, project_id: int, max_files: int):
    """Analyze GitLab project in background."""
    try:
        print(f"Starting analysis of GitLab project: {project_id}")
        result = await integration.analyze_project(project_id, max_files)
        
        if 'error' in result:
            print(f"Project analysis failed: {result['error']}")
        else:
            print(f"Project analysis completed:")
            print(f"  - Files analyzed: {result['files_analyzed']}")
            print(f"  - Success rate: {result['workflow_result']['success_rate']:.1%}")
            print(f"  - Execution time: {result['workflow_result']['execution_time_ms']:.0f}ms")
            
    except Exception as e:
        print(f"Error analyzing GitLab project {project_id}: {e}")


@router.get("/events/recent")
async def get_recent_events(limit: int = 10):
    """Get recent webhook events (mock implementation for demo)."""
    # In a real implementation, you'd store events in a database
    return {
        "events": [
            {
                "id": f"event_{i}",
                "type": "pull_request" if i % 2 == 0 else "push",
                "repository": f"user/repo-{i}",
                "timestamp": datetime.now().isoformat(),
                "status": "processed"
            }
            for i in range(1, limit + 1)
        ]
    }


@router.get("/analytics")
async def get_webhook_analytics():
    """Get webhook analytics (mock implementation for demo)."""
    return {
        "total_events_today": 45,
        "pull_requests_analyzed": 12,
        "bugs_detected": 23,
        "average_score": 7.8,
        "repositories_active": 8,
        "response_time_avg_ms": 2340,
        "events_by_type": {
            "pull_request": 28,
            "push": 15,
            "issues": 2
        },
        "platforms": {
            "github": 35,
            "gitlab": 10
        }
    }