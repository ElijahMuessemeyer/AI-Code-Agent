"""
Main API and CLI interface for the AI Code Agent system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import os
from pathlib import Path

from src.agents.agent_coordinator import AgentCoordinator, WorkflowType
from src.agents.code_generator import CodeRequirement, CodeType, CodeQuality
from src.agents.test_generator import TestFramework
from src.api.webhooks import router as webhooks_router
from src.api.dashboard import router as dashboard_router


# FastAPI app
app = FastAPI(
    title="AI Code Agent",
    description="Autonomous AI agent for code review, generation, and development",
    version="1.0.0"
)

# Global coordinator instance
coordinator: Optional[AgentCoordinator] = None

# Include routers
app.include_router(webhooks_router)
app.include_router(dashboard_router)


def get_coordinator() -> AgentCoordinator:
    """Get or create the global coordinator instance."""
    global coordinator
    if coordinator is None:
        coordinator = AgentCoordinator()
    return coordinator


# Pydantic models for API
class FileAnalysisRequest(BaseModel):
    file_paths: List[str]
    workflow_type: str = "comprehensive_analysis"


class CodeGenerationRequest(BaseModel):
    description: str
    language: str
    code_type: str = "function"
    quality_level: str = "production"
    constraints: List[str] = []
    dependencies: List[str] = []
    examples: List[str] = []
    context: str = ""


class DirectoryAnalysisRequest(BaseModel):
    directory_path: str
    max_files: int = 10


class WorkflowStatusResponse(BaseModel):
    status: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_tasks: List[Dict[str, Any]]


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Code Agent API",
        "version": "1.0.0",
        "endpoints": {
            "analyze_files": "/analyze/files",
            "analyze_directory": "/analyze/directory", 
            "generate_code": "/generate/code",
            "improve_quality": "/improve/quality",
            "workflow_status": "/status/workflow",
            "health": "/health",
            "dashboard": "/dashboard",
            "webhooks": "/webhooks",
            "github_webhook": "/webhooks/github",
            "gitlab_webhook": "/webhooks/gitlab"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    coord = get_coordinator()
    return {
        "status": "healthy",
        "agents": {
            "code_reviewer": "active",
            "bug_detector": "active", 
            "code_generator": "active",
            "test_generator": "active"
        },
        "workflow_status": coord.get_workflow_status()
    }


@app.post("/analyze/files")
async def analyze_files(request: FileAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze multiple files using specified workflow."""
    try:
        coord = get_coordinator()
        
        # Validate workflow type
        try:
            workflow_type = WorkflowType(request.workflow_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid workflow type. Must be one of: {[wf.value for wf in WorkflowType]}"
            )
        
        # Validate files exist
        for file_path in request.file_paths:
            if not Path(file_path).exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Execute workflow
        result = await coord.execute_workflow(workflow_type, request.file_paths)
        
        return {
            "workflow_type": result.workflow_type.value,
            "files_processed": len(result.target_files),
            "success_rate": result.success_rate,
            "execution_time_ms": result.execution_time_ms,
            "summary": result.summary,
            "recommendations": result.recommendations,
            "detailed_results": len(result.task_results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/directory")
async def analyze_directory(request: DirectoryAnalysisRequest):
    """Analyze all supported files in a directory."""
    try:
        coord = get_coordinator()
        
        if not Path(request.directory_path).exists():
            raise HTTPException(status_code=404, detail="Directory not found")
        
        result = await coord.analyze_codebase(request.directory_path)
        
        return {
            "directory": request.directory_path,
            "workflow_type": result.workflow_type.value,
            "files_analyzed": len(result.target_files),
            "success_rate": result.success_rate,
            "execution_time_ms": result.execution_time_ms,
            "summary": result.summary,
            "recommendations": result.recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/code")
async def generate_code(request: CodeGenerationRequest):
    """Generate code from natural language requirements."""
    try:
        coord = get_coordinator()
        
        # Create requirement object
        try:
            code_type = CodeType(request.code_type)
            quality_level = CodeQuality(request.quality_level)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        requirement = CodeRequirement(
            description=request.description,
            language=request.language,
            code_type=code_type,
            quality_level=quality_level,
            constraints=request.constraints,
            dependencies=request.dependencies,
            examples=request.examples,
            context=request.context
        )
        
        # Generate code
        result = await coord.code_generator.generate_code(requirement)
        
        return {
            "requirement": {
                "description": requirement.description,
                "language": requirement.language,
                "code_type": requirement.code_type.value
            },
            "generated_code": {
                "code": result.generated_code.code,
                "documentation": result.generated_code.documentation,
                "quality_score": result.generated_code.quality_score,
                "examples": result.generated_code.examples
            },
            "confidence": result.confidence,
            "generation_time_ms": result.generation_time_ms,
            "alternatives_available": len(result.alternative_solutions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/improve/quality")
async def improve_code_quality(request: FileAnalysisRequest):
    """Improve code quality using multiple agents."""
    try:
        coord = get_coordinator()
        
        # Validate files exist
        for file_path in request.file_paths:
            if not Path(file_path).exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        result = await coord.improve_code_quality(request.file_paths)
        
        return {
            "files_improved": len(result.target_files),
            "success_rate": result.success_rate,
            "execution_time_ms": result.execution_time_ms,
            "summary": result.summary,
            "recommendations": result.recommendations,
            "improvements_made": len(result.task_results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/workflow")
async def get_workflow_status():
    """Get current workflow status."""
    try:
        coord = get_coordinator()
        status = coord.get_workflow_status()
        
        return WorkflowStatusResponse(
            status="running" if status["active_tasks"] else "idle",
            total_tasks=status["total_tasks"],
            completed_tasks=status["by_status"].get("completed", 0),
            failed_tasks=status["by_status"].get("failed", 0),
            active_tasks=status["active_tasks"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_type}/status")
async def get_agent_status(agent_type: str):
    """Get status of a specific agent."""
    coord = get_coordinator()
    
    valid_agents = ["code_reviewer", "bug_detector", "code_generator", "test_generator"]
    if agent_type not in valid_agents:
        raise HTTPException(
            status_code=404, 
            detail=f"Agent not found. Valid agents: {valid_agents}"
        )
    
    return {
        "agent_type": agent_type,
        "status": "active",
        "capabilities": _get_agent_capabilities(agent_type)
    }


def _get_agent_capabilities(agent_type: str) -> List[str]:
    """Get capabilities of a specific agent."""
    capabilities = {
        "code_reviewer": [
            "Code quality analysis",
            "Security vulnerability detection", 
            "Performance optimization suggestions",
            "Maintainability assessment"
        ],
        "bug_detector": [
            "Runtime error detection",
            "Logic bug identification",
            "Edge case analysis",
            "Security issue detection"
        ],
        "code_generator": [
            "Code generation from requirements",
            "Multiple programming languages",
            "Production-quality code",
            "Documentation generation"
        ],
        "test_generator": [
            "Unit test generation",
            "Edge case testing",
            "Multiple test frameworks",
            "Coverage analysis"
        ]
    }
    
    return capabilities.get(agent_type, [])


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)