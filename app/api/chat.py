"""对话接口"""

import json

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.models.intent import IntentHandoff, IntentType
from app.models.request import ChatRequest, ClearRequest
from app.models.response import ApiResponse, SessionInfoResponse
from app.services.context_memory_service import context_memory_service
from app.services.intent_recognition_service import intent_recognition_service
from app.services.rag_agent_service import rag_agent_service

router = APIRouter()


def build_aiops_handoff(session_id: str) -> IntentHandoff:
    return IntentHandoff(session_id=session_id)


def log_routing(session_id: str, question: str, intent: str, confidence: float, routing_target: str):
    logger.info(
        f"[会话 {session_id}] 意图分流 "
        f"question={question!r} intent={intent} confidence={confidence} routing={routing_target}"
    )


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"[会话 {request.id}] 收到快速对话请求: {request.question}")
        decision = await intent_recognition_service.recognize(
            request.question,
            session_id=request.id,
        )

        if decision.intent == IntentType.AIOPS_DIAGNOSIS:
            handoff = build_aiops_handoff(request.id)
            await context_memory_service.persist_chat_turn(
                request.id,
                request.question,
                handoff.message,
            )
            log_routing(
                session_id=request.id,
                question=request.question,
                intent=decision.intent.value,
                confidence=decision.confidence,
                routing_target="aiops_confirmation",
            )
            return {
                "code": 200,
                "message": "success",
                "data": {
                    "success": True,
                    "answer": handoff.message,
                    "errorMessage": None,
                    "intent": decision.intent.value,
                    "handoff": handoff.model_dump(),
                },
            }

        answer = await rag_agent_service.query(
            request.question,
            session_id=request.id,
        )
        await context_memory_service.persist_chat_turn(
            request.id,
            request.question,
            answer,
        )
        log_routing(
            session_id=request.id,
            question=request.question,
            intent=decision.intent.value,
            confidence=decision.confidence,
            routing_target=decision.target_flow,
        )

        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": answer,
                "errorMessage": None,
                "intent": decision.intent.value,
                "handoff": None,
            },
        }
    except Exception as e:
        logger.error(f"对话接口错误: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e),
                "intent": None,
                "handoff": None,
            },
        }


@router.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    logger.info(f"[会话 {request.id}] 收到流式对话请求: {request.question}")

    async def event_generator():
        try:
            decision = await intent_recognition_service.recognize(
                request.question,
                session_id=request.id,
            )

            if decision.intent == IntentType.AIOPS_DIAGNOSIS:
                handoff = build_aiops_handoff(request.id)
                await context_memory_service.persist_chat_turn(
                    request.id,
                    request.question,
                    handoff.message,
                )
                log_routing(
                    session_id=request.id,
                    question=request.question,
                    intent=decision.intent.value,
                    confidence=decision.confidence,
                    routing_target="aiops_confirmation",
                )
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {
                            "type": "handoff",
                            "data": handoff.model_dump(),
                        },
                        ensure_ascii=False,
                    ),
                }
                return

            log_routing(
                session_id=request.id,
                question=request.question,
                intent=decision.intent.value,
                confidence=decision.confidence,
                routing_target=decision.target_flow,
            )

            full_response = ""
            async for chunk in rag_agent_service.query_stream(request.question, session_id=request.id):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                if chunk_type == "debug":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "debug",
                                "node": chunk.get("node", "unknown"),
                                "message_type": chunk.get("message_type", "unknown"),
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "tool_call":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "tool_call",
                                "data": chunk_data,
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "search_results":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "search_results",
                                "data": chunk_data,
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "content":
                    full_response += chunk_data or ""
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "content",
                                "data": chunk_data,
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "complete":
                    await context_memory_service.persist_chat_turn(
                        request.id,
                        request.question,
                        full_response,
                    )
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "done",
                                "data": chunk_data,
                            },
                            ensure_ascii=False,
                        ),
                    }
                elif chunk_type == "error":
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "error",
                                "data": str(chunk_data),
                            },
                            ensure_ascii=False,
                        ),
                    }

            logger.info(f"[会话 {request.id}] 流式对话完成")
        except Exception as e:
            logger.error(f"流式对话接口错误: {e}")
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "error",
                        "data": str(e),
                    },
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest):
    try:
        success = rag_agent_service.clear_session(request.session_id)
        logger.info(f"清空会话: {request.session_id}, 结果: {success}")

        return ApiResponse(
            status="success" if success else "error",
            message="会话已清空" if success else "清空会话失败",
            data=None,
        )
    except Exception as e:
        logger.error(f"清空会话错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str) -> SessionInfoResponse:
    try:
        history = rag_agent_service.get_session_history(session_id)
        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(history),
            history=history,
        )
    except Exception as e:
        logger.error(f"获取会话信息错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
