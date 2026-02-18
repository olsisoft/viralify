"""
Notification Service - Real-time notifications and alerts
Handles push notifications, email alerts, and in-app notifications
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum
import asyncio
import json
import os

# Database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Boolean, JSON, select, update, desc
from sqlalchemy.dialects.postgresql import UUID as PGUUID
import uuid

# Redis
from redis.asyncio import Redis

# Message Queue
import aio_pika

# ========================================
# Configuration
# ========================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://viralify_prod:password@localhost:5432/viralify_production")
# Ensure async driver is used
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
REDIS_URL = os.getenv("REDIS_URL", "redis://:redis_secure_2024@localhost:6379/3")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://tiktok:rabbitmq_secure_2024@localhost:5672/")

# ========================================
# FastAPI App
# ========================================

app = FastAPI(
    title="Notification Service",
    description="Real-time notifications, alerts, and communication",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Database Models
# ========================================

class Base(DeclarativeBase):
    pass

class Notification(Base):
    __tablename__ = "notifications"
    
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSON, default={})
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Redis client
redis_client: Optional[Redis] = None

# ========================================
# WebSocket Connection Manager
# ========================================

class ConnectionManager:
    """Manages WebSocket connections for real-time notifications"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
    
    async def send_to_user(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_connections.add(connection)
            
            # Clean up dead connections
            for dead in dead_connections:
                self.active_connections[user_id].discard(dead)
    
    async def broadcast(self, message: dict):
        for user_id in self.active_connections:
            await self.send_to_user(user_id, message)


manager = ConnectionManager()

# ========================================
# Pydantic Models
# ========================================

class NotificationType(str, Enum):
    POST_PUBLISHED = "post_published"
    POST_FAILED = "post_failed"
    POST_SCHEDULED = "post_scheduled"
    TREND_ALERT = "trend_alert"
    ENGAGEMENT_MILESTONE = "engagement_milestone"
    AI_SUGGESTION = "ai_suggestion"
    SYSTEM = "system"
    WEEKLY_REPORT = "weekly_report"

class CreateNotificationRequest(BaseModel):
    user_id: UUID
    type: NotificationType
    title: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class NotificationResponse(BaseModel):
    id: UUID
    type: str
    title: str
    message: Optional[str]
    data: Dict[str, Any]
    is_read: bool
    created_at: datetime

class NotificationPreferences(BaseModel):
    email_enabled: bool = True
    push_enabled: bool = True
    post_published: bool = True
    post_failed: bool = True
    trend_alerts: bool = True
    engagement_milestones: bool = True
    ai_suggestions: bool = True
    weekly_report: bool = True

# ========================================
# Notification Templates
# ========================================

NOTIFICATION_TEMPLATES = {
    NotificationType.POST_PUBLISHED: {
        "title": "🎉 Post Published Successfully!",
        "message": "Your video '{title}' is now live on TikTok!"
    },
    NotificationType.POST_FAILED: {
        "title": "❌ Post Publishing Failed",
        "message": "Failed to publish '{title}': {error}"
    },
    NotificationType.POST_SCHEDULED: {
        "title": "📅 Post Scheduled",
        "message": "'{title}' will be published on {scheduled_time}"
    },
    NotificationType.TREND_ALERT: {
        "title": "🔥 New Trend Alert!",
        "message": "'{trend_name}' is trending in your niche. Perfect time to create content!"
    },
    NotificationType.ENGAGEMENT_MILESTONE: {
        "title": "🏆 Milestone Reached!",
        "message": "Your post '{title}' just hit {milestone} {metric}!"
    },
    NotificationType.AI_SUGGESTION: {
        "title": "💡 AI Content Suggestion",
        "message": "{suggestion}"
    },
    NotificationType.WEEKLY_REPORT: {
        "title": "📊 Your Weekly Report is Ready",
        "message": "You gained {views} views and {followers} followers this week!"
    }
}

# ========================================
# Notification Service
# ========================================

class NotificationService:
    """Core notification logic"""
    
    async def create_notification(
        self, 
        user_id: UUID, 
        notification_type: NotificationType,
        title: Optional[str] = None,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> NotificationResponse:
        """Create and send a notification"""
        
        # Use template if title/message not provided
        template = NOTIFICATION_TEMPLATES.get(notification_type, {})
        final_title = title or template.get("title", "Notification")
        final_message = message or template.get("message", "")
        
        # Format message with data
        if data and final_message:
            try:
                final_message = final_message.format(**data)
            except KeyError:
                pass
        
        async with async_session() as session:
            notification = Notification(
                user_id=user_id,
                type=notification_type.value,
                title=final_title,
                message=final_message,
                data=data or {}
            )
            
            session.add(notification)
            await session.commit()
            await session.refresh(notification)
            
            response = NotificationResponse(
                id=notification.id,
                type=notification.type,
                title=notification.title,
                message=notification.message,
                data=notification.data,
                is_read=notification.is_read,
                created_at=notification.created_at
            )
            
            # Send real-time notification via WebSocket
            await manager.send_to_user(str(user_id), {
                "type": "notification",
                "data": response.dict()
            })
            
            # Publish to Redis for other services
            if redis_client:
                await redis_client.publish(
                    f"notifications:{user_id}",
                    json.dumps(response.dict(), default=str)
                )
            
            return response
    
    async def get_user_notifications(
        self, 
        user_id: UUID, 
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> List[NotificationResponse]:
        """Get notifications for a user"""
        
        async with async_session() as session:
            query = select(Notification).where(Notification.user_id == user_id)
            
            if unread_only:
                query = query.where(Notification.is_read == False)
            
            query = query.order_by(desc(Notification.created_at)).offset(offset).limit(limit)
            
            result = await session.execute(query)
            notifications = result.scalars().all()
            
            return [
                NotificationResponse(
                    id=n.id,
                    type=n.type,
                    title=n.title,
                    message=n.message,
                    data=n.data,
                    is_read=n.is_read,
                    created_at=n.created_at
                )
                for n in notifications
            ]
    
    async def mark_as_read(self, notification_id: UUID, user_id: UUID) -> bool:
        """Mark a notification as read"""
        
        async with async_session() as session:
            result = await session.execute(
                update(Notification)
                .where(
                    Notification.id == notification_id,
                    Notification.user_id == user_id
                )
                .values(is_read=True, read_at=datetime.utcnow())
            )
            await session.commit()
            return result.rowcount > 0
    
    async def mark_all_as_read(self, user_id: UUID) -> int:
        """Mark all notifications as read for a user"""
        
        async with async_session() as session:
            result = await session.execute(
                update(Notification)
                .where(
                    Notification.user_id == user_id,
                    Notification.is_read == False
                )
                .values(is_read=True, read_at=datetime.utcnow())
            )
            await session.commit()
            return result.rowcount
    
    async def get_unread_count(self, user_id: UUID) -> int:
        """Get count of unread notifications"""
        
        # Check cache first
        if redis_client:
            cached = await redis_client.get(f"notifications:unread:{user_id}")
            if cached:
                return int(cached)
        
        async with async_session() as session:
            from sqlalchemy import func
            result = await session.execute(
                select(func.count(Notification.id))
                .where(
                    Notification.user_id == user_id,
                    Notification.is_read == False
                )
            )
            count = result.scalar() or 0
            
            # Cache for 1 minute
            if redis_client:
                await redis_client.setex(f"notifications:unread:{user_id}", 60, str(count))
            
            return count
    
    async def delete_notification(self, notification_id: UUID, user_id: UUID) -> bool:
        """Delete a notification"""
        
        async with async_session() as session:
            result = await session.execute(
                select(Notification).where(
                    Notification.id == notification_id,
                    Notification.user_id == user_id
                )
            )
            notification = result.scalar_one_or_none()
            
            if notification:
                await session.delete(notification)
                await session.commit()
                return True
            return False


notification_service = NotificationService()

# ========================================
# Startup & Shutdown
# ========================================

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = Redis.from_url(REDIS_URL)
    
    # Start message queue consumer
    asyncio.create_task(consume_notification_queue())

@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()

async def consume_notification_queue():
    """Consume notifications from RabbitMQ"""
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        channel = await connection.channel()
        
        queue = await channel.declare_queue("notifications", durable=True)
        
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        data = json.loads(message.body.decode())
                        await notification_service.create_notification(
                            user_id=UUID(data["user_id"]),
                            notification_type=NotificationType(data["type"]),
                            title=data.get("title"),
                            message=data.get("message"),
                            data=data.get("data")
                        )
                    except Exception as e:
                        print(f"Error processing notification: {e}")
    except Exception as e:
        print(f"RabbitMQ connection error: {e}")

# ========================================
# API Endpoints
# ========================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "notification-service"}

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time notifications"""
    await manager.connect(websocket, user_id)
    try:
        # Send unread count on connect
        count = await notification_service.get_unread_count(UUID(user_id))
        await websocket.send_json({"type": "unread_count", "count": count})
        
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "mark_read":
                notification_id = message.get("notification_id")
                if notification_id:
                    await notification_service.mark_as_read(UUID(notification_id), UUID(user_id))
                    count = await notification_service.get_unread_count(UUID(user_id))
                    await websocket.send_json({"type": "unread_count", "count": count})
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)

@app.post("/api/v1/notifications", response_model=NotificationResponse)
async def create_notification(request: CreateNotificationRequest):
    """Create a new notification"""
    return await notification_service.create_notification(
        user_id=request.user_id,
        notification_type=request.type,
        title=request.title,
        message=request.message,
        data=request.data
    )

@app.get("/api/v1/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    user_id: UUID = Query(...),
    unread_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """Get notifications for a user"""
    return await notification_service.get_user_notifications(user_id, unread_only, limit, offset)

@app.get("/api/v1/notifications/unread-count")
async def get_unread_count(user_id: UUID = Query(...)):
    """Get unread notification count"""
    count = await notification_service.get_unread_count(user_id)
    return {"count": count}

@app.put("/api/v1/notifications/{notification_id}/read")
async def mark_as_read(notification_id: UUID, user_id: UUID = Query(...)):
    """Mark a notification as read"""
    success = await notification_service.mark_as_read(notification_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "marked_as_read"}

@app.put("/api/v1/notifications/read-all")
async def mark_all_as_read(user_id: UUID = Query(...)):
    """Mark all notifications as read"""
    count = await notification_service.mark_all_as_read(user_id)
    return {"status": "success", "marked_count": count}

@app.delete("/api/v1/notifications/{notification_id}")
async def delete_notification(notification_id: UUID, user_id: UUID = Query(...)):
    """Delete a notification"""
    success = await notification_service.delete_notification(notification_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "deleted"}

# ========================================
# Trigger Endpoints (for other services)
# ========================================

@app.post("/api/v1/notifications/trigger/post-published")
async def trigger_post_published(user_id: UUID, post_title: str, share_url: str):
    """Trigger notification when post is published"""
    return await notification_service.create_notification(
        user_id=user_id,
        notification_type=NotificationType.POST_PUBLISHED,
        data={"title": post_title, "share_url": share_url}
    )

@app.post("/api/v1/notifications/trigger/post-failed")
async def trigger_post_failed(user_id: UUID, post_title: str, error: str):
    """Trigger notification when post fails"""
    return await notification_service.create_notification(
        user_id=user_id,
        notification_type=NotificationType.POST_FAILED,
        data={"title": post_title, "error": error}
    )

@app.post("/api/v1/notifications/trigger/trend-alert")
async def trigger_trend_alert(user_id: UUID, trend_name: str, trend_score: float):
    """Trigger trend alert notification"""
    return await notification_service.create_notification(
        user_id=user_id,
        notification_type=NotificationType.TREND_ALERT,
        data={"trend_name": trend_name, "trend_score": trend_score}
    )

@app.post("/api/v1/notifications/trigger/engagement-milestone")
async def trigger_engagement_milestone(
    user_id: UUID, 
    post_title: str, 
    milestone: int, 
    metric: str
):
    """Trigger engagement milestone notification"""
    return await notification_service.create_notification(
        user_id=user_id,
        notification_type=NotificationType.ENGAGEMENT_MILESTONE,
        data={"title": post_title, "milestone": milestone, "metric": metric}
    )

# ========================================
# Run Application
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
