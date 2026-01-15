"""
Web dashboard for monitoring AI Code Agent activities.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from src.agents.agent_coordinator import AgentCoordinator


router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Templates setup
templates_dir = Path(__file__).parent.parent.parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Global coordinator
coordinator: Optional[AgentCoordinator] = None


def get_coordinator() -> AgentCoordinator:
    """Get or create coordinator instance."""
    global coordinator
    if coordinator is None:
        coordinator = AgentCoordinator()
    return coordinator


@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page."""
    # Create template if it doesn't exist
    template_path = templates_dir / "dashboard.html"
    if not template_path.exists():
        _create_dashboard_template()
    
    # Get dashboard data
    coord = get_coordinator()
    dashboard_data = await _get_dashboard_data()
    
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": request,
            "data": dashboard_data
        }
    )


@router.get("/api/status")
async def get_system_status():
    """Get current system status."""
    coord = get_coordinator()
    workflow_status = coord.get_workflow_status()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system_status": "healthy",
        "agents": {
            "code_reviewer": {"status": "active", "version": "1.0.0"},
            "bug_detector": {"status": "active", "version": "1.0.0"},
            "code_generator": {"status": "active", "version": "1.0.0"},
            "test_generator": {"status": "active", "version": "1.0.0"}
        },
        "workflow_status": workflow_status,
        "integrations": {
            "github": _check_github_integration(),
            "gitlab": _check_gitlab_integration()
        }
    }


@router.get("/api/metrics")
async def get_metrics():
    """Get system metrics and analytics."""
    # Mock data for demonstration
    now = datetime.now()
    
    return {
        "timestamp": now.isoformat(),
        "time_range": "24h",
        "metrics": {
            "total_analyses": 156,
            "pull_requests_reviewed": 34,
            "bugs_detected": 89,
            "code_generated": 12,
            "tests_created": 67,
            "average_quality_score": 7.6,
            "success_rate": 0.94,
            "average_response_time_ms": 2340
        },
        "hourly_activity": _generate_hourly_activity(),
        "quality_distribution": {
            "excellent": 45,  # 8-10
            "good": 67,       # 6-8
            "needs_work": 23, # 4-6
            "poor": 8         # 0-4
        },
        "bug_severity_distribution": {
            "critical": 5,
            "high": 18,
            "medium": 34,
            "low": 32
        },
        "language_distribution": {
            "python": 45,
            "javascript": 28,
            "typescript": 15,
            "java": 8,
            "other": 4
        }
    }


@router.get("/api/recent-activities")
async def get_recent_activities(limit: int = 20):
    """Get recent agent activities."""
    # Mock data for demonstration
    activities = []
    
    activity_types = [
        ("code_review", "Reviewed pull request #123 in user/project"),
        ("bug_detection", "Found 5 bugs in src/main.py"),
        ("code_generation", "Generated API endpoint for user authentication"),
        ("test_generation", "Created 12 test cases for Calculator class"),
        ("webhook", "Processed GitHub webhook for user/repo"),
        ("analysis", "Completed comprehensive analysis of 8 files")
    ]
    
    for i in range(limit):
        activity_type, description = activity_types[i % len(activity_types)]
        
        activities.append({
            "id": f"activity_{i}",
            "type": activity_type,
            "description": description,
            "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
            "status": "completed" if i % 10 != 9 else "failed",
            "agent": ["code_reviewer", "bug_detector", "code_generator", "test_generator"][i % 4],
            "metadata": {
                "execution_time_ms": 1000 + (i * 100),
                "files_processed": max(1, i % 5),
                "score": round(7 + (i % 3), 1)
            }
        })
    
    return {"activities": activities}


@router.get("/api/repositories")
async def get_monitored_repositories():
    """Get list of monitored repositories."""
    # Mock data for demonstration
    return {
        "repositories": [
            {
                "id": "1",
                "name": "user/frontend-app",
                "platform": "github",
                "url": "https://github.com/user/frontend-app",
                "language": "typescript",
                "webhook_configured": True,
                "last_analysis": (datetime.now() - timedelta(hours=2)).isoformat(),
                "quality_score": 8.2,
                "open_issues": 3,
                "recent_activity": 15
            },
            {
                "id": "2", 
                "name": "company/api-service",
                "platform": "gitlab",
                "url": "https://gitlab.com/company/api-service",
                "language": "python",
                "webhook_configured": True,
                "last_analysis": (datetime.now() - timedelta(hours=1)).isoformat(),
                "quality_score": 7.8,
                "open_issues": 7,
                "recent_activity": 8
            },
            {
                "id": "3",
                "name": "team/data-processor",
                "platform": "github", 
                "url": "https://github.com/team/data-processor",
                "language": "python",
                "webhook_configured": False,
                "last_analysis": (datetime.now() - timedelta(days=2)).isoformat(),
                "quality_score": 6.9,
                "open_issues": 12,
                "recent_activity": 3
            }
        ]
    }


@router.get("/api/agents/{agent_name}/stats")
async def get_agent_statistics(agent_name: str):
    """Get statistics for a specific agent."""
    valid_agents = ["code_reviewer", "bug_detector", "code_generator", "test_generator"]
    
    if agent_name not in valid_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Mock data specific to each agent
    base_stats = {
        "agent_name": agent_name,
        "status": "active",
        "uptime_hours": 168,
        "total_requests": 234,
        "successful_requests": 220,
        "failed_requests": 14,
        "average_response_time_ms": 1850,
        "requests_per_hour": _generate_hourly_requests()
    }
    
    if agent_name == "code_reviewer":
        base_stats.update({
            "reviews_completed": 89,
            "average_quality_score": 7.4,
            "security_issues_found": 23,
            "performance_suggestions": 45
        })
    elif agent_name == "bug_detector":
        base_stats.update({
            "bugs_detected": 156,
            "critical_bugs": 12,
            "false_positive_rate": 0.08,
            "accuracy": 0.92
        })
    elif agent_name == "code_generator":
        base_stats.update({
            "code_artifacts_generated": 34,
            "average_confidence": 0.85,
            "languages_supported": 7,
            "lines_of_code_generated": 2340
        })
    elif agent_name == "test_generator":
        base_stats.update({
            "test_suites_generated": 67,
            "total_test_cases": 890,
            "average_coverage": 0.82,
            "frameworks_supported": 4
        })
    
    return base_stats


@router.post("/api/agents/{agent_name}/restart")
async def restart_agent(agent_name: str):
    """Restart a specific agent."""
    valid_agents = ["code_reviewer", "bug_detector", "code_generator", "test_generator"]
    
    if agent_name not in valid_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # In a real implementation, this would restart the agent
    return {
        "status": "success",
        "message": f"Agent {agent_name} restarted successfully",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/api/alerts")
async def get_system_alerts():
    """Get current system alerts and warnings."""
    alerts = []
    
    # Mock some alerts
    if datetime.now().hour % 3 == 0:  # Simulate occasional alerts
        alerts.append({
            "id": "alert_1",
            "level": "warning",
            "title": "High response time detected",
            "message": "Bug detector agent response time increased to 3.2s",
            "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
            "agent": "bug_detector"
        })
    
    if datetime.now().hour % 5 == 0:
        alerts.append({
            "id": "alert_2", 
            "level": "info",
            "title": "Webhook processing spike",
            "message": "Processing 2x normal webhook volume",
            "timestamp": (datetime.now() - timedelta(minutes=8)).isoformat(),
            "agent": "all"
        })
    
    return {"alerts": alerts}


@router.get("/api/logs")
async def get_system_logs(level: str = "info", limit: int = 50):
    """Get system logs."""
    # Mock log entries
    log_levels = ["debug", "info", "warning", "error"]
    
    if level not in log_levels:
        level = "info"
    
    logs = []
    for i in range(limit):
        log_level = log_levels[i % len(log_levels)]
        if log_levels.index(log_level) < log_levels.index(level):
            continue
            
        logs.append({
            "timestamp": (datetime.now() - timedelta(minutes=i*2)).isoformat(),
            "level": log_level,
            "component": ["coordinator", "github_integration", "gitlab_integration", "api"][i % 4],
            "message": f"Log message {i} - {log_level} level event occurred",
            "metadata": {
                "request_id": f"req_{i}",
                "execution_time_ms": 100 + (i * 10)
            }
        })
    
    return {"logs": logs[:limit]}


def _check_github_integration() -> Dict[str, Any]:
    """Check GitHub integration status."""
    try:
        from src.integrations.github_integration import GitHubIntegration
        integration = GitHubIntegration()
        return {
            "status": "connected",
            "webhook_configured": bool(integration.webhook_secret),
            "rate_limit_remaining": 4500  # Mock value
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def _check_gitlab_integration() -> Dict[str, Any]:
    """Check GitLab integration status."""
    try:
        from src.integrations.gitlab_integration import GitLabIntegration
        integration = GitLabIntegration()
        return {
            "status": "connected",
            "webhook_configured": bool(integration.webhook_secret),
            "url": integration.url
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e)
        }


def _generate_hourly_activity() -> List[Dict[str, Any]]:
    """Generate mock hourly activity data."""
    activities = []
    now = datetime.now()
    
    for i in range(24):
        hour = (now - timedelta(hours=i)).hour
        # Simulate higher activity during work hours
        base_activity = 10 if 9 <= hour <= 17 else 3
        variance = max(0, base_activity + (i % 5) - 2)
        
        activities.append({
            "hour": hour,
            "timestamp": (now - timedelta(hours=i)).isoformat(),
            "total_requests": variance * 3,
            "pull_requests": max(0, variance - 2),
            "bugs_detected": variance * 2,
            "code_generated": max(0, variance - 3)
        })
    
    return list(reversed(activities))


def _generate_hourly_requests() -> List[int]:
    """Generate mock hourly request data for agents."""
    return [max(0, 10 + (i % 8) - 4) for i in range(24)]


async def _get_dashboard_data() -> Dict[str, Any]:
    """Get all dashboard data."""
    return {
        "system_status": await get_system_status(),
        "metrics": await get_metrics(),
        "recent_activities": (await get_recent_activities(10))["activities"],
        "repositories": (await get_monitored_repositories())["repositories"],
        "alerts": (await get_system_alerts())["alerts"]
    }


def _create_dashboard_template():
    """Create the dashboard HTML template."""
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Code Agent Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .card { @apply bg-white rounded-lg shadow-md p-6 mb-6; }
        .metric-card { @apply bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-lg p-6; }
        .status-badge { @apply inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium; }
        .status-active { @apply bg-green-100 text-green-800; }
        .status-error { @apply bg-red-100 text-red-800; }
        .status-warning { @apply bg-yellow-100 text-yellow-800; }
    </style>
</head>
<body class="bg-gray-100">
    <nav class="bg-white shadow-sm border-b">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16">
                <div class="flex items-center">
                    <h1 class="text-xl font-bold text-gray-900">🤖 AI Code Agent Dashboard</h1>
                </div>
                <div class="flex items-center space-x-4">
                    <span class="status-badge status-active">System Healthy</span>
                    <span class="text-sm text-gray-500">Last updated: <span id="last-updated"></span></span>
                </div>
            </div>
        </div>
    </nav>

    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Metrics Overview -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="metric-card">
                <div class="text-3xl font-bold">{{ data.metrics.metrics.total_analyses }}</div>
                <div class="text-blue-100">Total Analyses</div>
            </div>
            <div class="metric-card">
                <div class="text-3xl font-bold">{{ data.metrics.metrics.bugs_detected }}</div>
                <div class="text-blue-100">Bugs Detected</div>
            </div>
            <div class="metric-card">
                <div class="text-3xl font-bold">{{ "%.1f"|format(data.metrics.metrics.average_quality_score) }}</div>
                <div class="text-blue-100">Avg Quality Score</div>
            </div>
            <div class="metric-card">
                <div class="text-3xl font-bold">{{ "%.0f"|format(data.metrics.metrics.success_rate * 100) }}%</div>
                <div class="text-blue-100">Success Rate</div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Activity Chart -->
            <div class="card">
                <h3 class="text-lg font-semibold mb-4">24-Hour Activity</h3>
                <canvas id="activityChart" width="400" height="200"></canvas>
            </div>

            <!-- Agent Status -->
            <div class="card">
                <h3 class="text-lg font-semibold mb-4">Agent Status</h3>
                <div class="space-y-4">
                    {% for agent_name, agent_info in data.system_status.agents.items() %}
                    <div class="flex justify-between items-center">
                        <div class="flex items-center">
                            <div class="w-3 h-3 bg-green-400 rounded-full mr-3"></div>
                            <span class="font-medium">{{ agent_name.replace('_', ' ').title() }}</span>
                        </div>
                        <span class="status-badge status-active">{{ agent_info.status }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <!-- Recent Activities -->
            <div class="card">
                <h3 class="text-lg font-semibold mb-4">Recent Activities</h3>
                <div class="space-y-3">
                    {% for activity in data.recent_activities[:5] %}
                    <div class="flex justify-between items-start">
                        <div class="flex-1">
                            <p class="text-sm font-medium">{{ activity.description }}</p>
                            <p class="text-xs text-gray-500">{{ activity.agent.replace('_', ' ').title() }} • {{ activity.timestamp[:16] }}</p>
                        </div>
                        <span class="status-badge {% if activity.status == 'completed' %}status-active{% else %}status-error{% endif %}">
                            {{ activity.status }}
                        </span>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <!-- Monitored Repositories -->
            <div class="card">
                <h3 class="text-lg font-semibold mb-4">Monitored Repositories</h3>
                <div class="space-y-3">
                    {% for repo in data.repositories[:3] %}
                    <div class="flex justify-between items-center">
                        <div>
                            <p class="font-medium">{{ repo.name }}</p>
                            <p class="text-sm text-gray-500">{{ repo.platform }} • Quality: {{ repo.quality_score }}/10</p>
                        </div>
                        <span class="status-badge {% if repo.webhook_configured %}status-active{% else %}status-warning{% endif %}">
                            {% if repo.webhook_configured %}Connected{% else %}Setup Required{% endif %}
                        </span>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>

        <!-- Alerts -->
        {% if data.alerts %}
        <div class="card mt-8">
            <h3 class="text-lg font-semibold mb-4">System Alerts</h3>
            {% for alert in data.alerts %}
            <div class="bg-{{ 'yellow' if alert.level == 'warning' else 'blue' }}-50 border border-{{ 'yellow' if alert.level == 'warning' else 'blue' }}-200 rounded-md p-4 mb-3">
                <div class="flex">
                    <div class="flex-1">
                        <h4 class="text-sm font-medium text-{{ 'yellow' if alert.level == 'warning' else 'blue' }}-800">{{ alert.title }}</h4>
                        <p class="text-sm text-{{ 'yellow' if alert.level == 'warning' else 'blue' }}-700">{{ alert.message }}</p>
                    </div>
                    <span class="text-xs text-{{ 'yellow' if alert.level == 'warning' else 'blue' }}-600">{{ alert.timestamp[:16] }}</span>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>

    <script>
        // Update timestamp
        document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();

        // Activity chart
        const ctx = document.getElementById('activityChart').getContext('2d');
        const activityData = {{ data.metrics.hourly_activity | tojsonfilter }};
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: activityData.map(d => d.hour + ':00'),
                datasets: [{
                    label: 'Total Requests',
                    data: activityData.map(d => d.total_requests),
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        // Auto-refresh every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);
    </script>
</body>
</html>'''
    
    template_path = templates_dir / "dashboard.html"
    template_path.write_text(template_content)