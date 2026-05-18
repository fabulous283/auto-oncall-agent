"""AIOps 智能运维接口"""

import json

from fastapi import APIRouter
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.models.aiops import AIOpsRequest
from app.services.aiops_service import aiops_service
from app.services.context_memory_service import context_memory_service

router = APIRouter()


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest):
    session_id = request.session_id or "default"
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求（流式）")

    async def event_generator():
        final_message = ""
        try:
            async for event in aiops_service.diagnose(session_id=session_id):
                if event.get("type") == "report" and event.get("report"):
                    final_message = event["report"]
                elif event.get("type") == "complete":
                    final_message = event.get("response") or final_message

                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False),
                }

                if event.get("type") in ["complete", "error"]:
                    break

            if final_message:
                await context_memory_service.persist_assistant_message(session_id, final_message)

            logger.info(f"[会话 {session_id}] AIOps 诊断流式响应完成")
        except Exception as e:
            logger.error(f"[会话 {session_id}] AIOps 诊断流式响应异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "error",
                        "stage": "exception",
                        "message": f"诊断异常: {str(e)}",
                    },
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())
