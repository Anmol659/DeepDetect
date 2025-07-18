# 🚀 DeepDetect Production Deployment Guide

## 📋 Overview
This guide provides comprehensive instructions for deploying DeepDetect in production environments, making it real-world ready with enterprise-grade features.

## 🏗️ Production Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Servers   │    │   AI Workers   │
│   (Nginx/HAProxy)│◄──►│   (Flask/Gunicorn)│◄──►│   (GPU Nodes)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CDN/Cache     │    │   Database      │    │   File Storage  │
│   (CloudFlare)  │    │   (PostgreSQL)  │    │   (S3/MinIO)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 1. Model Optimization for Production

### A. Model Quantization
```python
# backend/optimize_model.py
import torch
import torch.quantization as quantization
from inference import model

def optimize_model():
    """Optimize model for production deployment"""
    
    # 1. Dynamic Quantization (Reduces model size by 75%)
    model_quantized = quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    # 2. TorchScript Compilation (Faster inference)
    model_scripted = torch.jit.script(model_quantized)
    
    # 3. Save optimized model
    torch.jit.save(model_scripted, 'checkpoints/model_optimized.pt')
    
    print("✓ Model optimized for production")
    return model_scripted

if __name__ == "__main__":
    optimize_model()
```

### B. ONNX Conversion (Cross-platform)
```python
# backend/convert_to_onnx.py
import torch
import torch.onnx
from inference import model

def convert_to_onnx():
    """Convert PyTorch model to ONNX for broader deployment"""
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 380, 380)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "checkpoints/model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("✓ Model converted to ONNX format")

if __name__ == "__main__":
    convert_to_onnx()
```

## 🐳 2. Docker Containerization

### A. Production Dockerfile
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY checkpoints/ ./checkpoints/
COPY templates/ ./templates/
COPY run_extension.py .

# Create non-root user
RUN useradd -m -u 1000 deepdetect && \
    chown -R deepdetect:deepdetect /app
USER deepdetect

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "backend.app:app"]
```

### B. Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  deepdetect:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/deepdetect
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=deepdetect
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - deepdetect

volumes:
  postgres_data:
  redis_data:
```

## ⚡ 3. Performance Optimization

### A. Caching Layer
```python
# backend/cache.py
import redis
import json
import hashlib
from functools import wraps

# Redis connection
cache = redis.Redis(
    host='localhost', 
    port=6379, 
    db=0,
    decode_responses=True
)

def cache_result(expiration=3600):
    """Cache analysis results to avoid recomputation"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from file hash
            if args and hasattr(args[0], 'read'):
                file_data = args[0]
                file_hash = hashlib.md5(file_data).hexdigest()
                cache_key = f"analysis:{file_hash}"
                
                # Check cache
                cached_result = cache.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Store in cache
                cache.setex(
                    cache_key, 
                    expiration, 
                    json.dumps(result)
                )
                
                return result
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage in inference.py
@cache_result(expiration=7200)  # Cache for 2 hours
def predict_image(image_bytes):
    # ... existing code
    pass
```

### B. Async Processing
```python
# backend/async_worker.py
import asyncio
import aioredis
from celery import Celery

# Celery configuration
celery_app = Celery(
    'deepdetect',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task
def analyze_image_async(image_data, task_id):
    """Asynchronous image analysis task"""
    try:
        result = predict_image(image_data)
        
        # Store result with task ID
        cache.setex(f"task:{task_id}", 3600, json.dumps(result))
        
        return result
    except Exception as e:
        error_result = {"error": str(e), "status": "failed"}
        cache.setex(f"task:{task_id}", 3600, json.dumps(error_result))
        return error_result

# Modified Flask endpoint
@app.route("/analyze_async", methods=["POST"])
def analyze_async():
    """Start asynchronous analysis"""
    file = request.files["file"]
    task_id = str(uuid.uuid4())
    
    # Queue task
    analyze_image_async.delay(file.read(), task_id)
    
    return jsonify({
        "task_id": task_id,
        "status": "processing",
        "check_url": f"/status/{task_id}"
    })

@app.route("/status/<task_id>")
def check_status(task_id):
    """Check analysis status"""
    result = cache.get(f"task:{task_id}")
    if result:
        return jsonify(json.loads(result))
    else:
        return jsonify({"status": "processing"})
```

## 🔒 4. Security Hardening

### A. Input Validation & Sanitization
```python
# backend/security.py
import magic
from werkzeug.utils import secure_filename

class SecurityValidator:
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'mp4', 'avi', 'mov'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    @staticmethod
    def validate_file(file):
        """Comprehensive file validation"""
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset
        
        if size > SecurityValidator.MAX_FILE_SIZE:
            raise ValueError("File too large")
        
        if size == 0:
            raise ValueError("Empty file")
        
        # Check file extension
        filename = secure_filename(file.filename)
        if not filename or '.' not in filename:
            raise ValueError("Invalid filename")
        
        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in SecurityValidator.ALLOWED_EXTENSIONS:
            raise ValueError("File type not allowed")
        
        # Check MIME type using python-magic
        file_data = file.read(1024)  # Read first 1KB
        file.seek(0)  # Reset
        
        mime_type = magic.from_buffer(file_data, mime=True)
        if not mime_type.startswith(('image/', 'video/')):
            raise ValueError("Invalid file type")
        
        return True

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "10 per minute"]
)

@app.route("/analyze", methods=["POST"])
@limiter.limit("5 per minute")  # Strict limit for analysis
def analyze():
    # ... existing code with validation
    SecurityValidator.validate_file(request.files["file"])
    # ... continue processing
```

### B. API Authentication
```python
# backend/auth.py
import jwt
from functools import wraps
from datetime import datetime, timedelta

class APIAuth:
    SECRET_KEY = "your-secret-key-here"  # Use environment variable
    
    @staticmethod
    def generate_token(user_id, expires_in=3600):
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, APIAuth.SECRET_KEY, algorithm='HS256')
    
    @staticmethod
    def verify_token(token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, APIAuth.SECRET_KEY, algorithms=['HS256'])
            return payload['user_id']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

def require_auth(f):
    """Decorator for protected endpoints"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        user_id = APIAuth.verify_token(token)
        if not user_id:
            return jsonify({'error': 'Invalid token'}), 401
        
        request.user_id = user_id
        return f(*args, **kwargs)
    
    return decorated

# Protected endpoint
@app.route("/analyze", methods=["POST"])
@require_auth
def analyze():
    # ... existing code
    pass
```

## 📊 5. Monitoring & Logging

### A. Structured Logging
```python
# backend/logging_config.py
import logging
import structlog
from pythonjsonlogger import jsonlogger

def setup_logging():
    """Configure structured logging"""
    
    # JSON formatter
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    
    # File handler
    file_handler = logging.FileHandler('logs/deepdetect.log')
    file_handler.setFormatter(json_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Usage
logger = structlog.get_logger()

@app.route("/analyze", methods=["POST"])
def analyze():
    logger.info("Analysis started", 
                filename=file.filename, 
                file_size=len(file_data),
                user_id=getattr(request, 'user_id', 'anonymous'))
    
    # ... processing
    
    logger.info("Analysis completed", 
                result=result['label'], 
                confidence=result['confidence'],
                processing_time=processing_time)
```

### B. Metrics Collection
```python
# backend/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('deepdetect_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('deepdetect_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('deepdetect_active_connections', 'Active connections')
MODEL_PREDICTIONS = Counter('deepdetect_predictions_total', 'Total predictions', ['label'])

@app.before_request
def before_request():
    request.start_time = time.time()
    ACTIVE_CONNECTIONS.inc()

@app.after_request
def after_request(response):
    REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint).inc()
    REQUEST_DURATION.observe(time.time() - request.start_time)
    ACTIVE_CONNECTIONS.dec()
    return response

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

# Usage in analysis
def predict_image(image_bytes):
    result = # ... prediction logic
    MODEL_PREDICTIONS.labels(label=result['label']).inc()
    return result
```

## 🌐 6. Cloud Deployment Options

### A. AWS Deployment
```yaml
# aws-deployment.yml (AWS ECS)
version: '3'
services:
  deepdetect:
    image: your-registry/deepdetect:latest
    cpu: 2048
    memory: 4096
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    logging:
      driver: awslogs
      options:
        awslogs-group: /ecs/deepdetect
        awslogs-region: us-west-2
```

### B. Kubernetes Deployment
```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepdetect
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepdetect
  template:
    metadata:
      labels:
        app: deepdetect
    spec:
      containers:
      - name: deepdetect
        image: your-registry/deepdetect:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: deepdetect-secrets
              key: database-url
---
apiVersion: v1
kind: Service
metadata:
  name: deepdetect-service
spec:
  selector:
    app: deepdetect
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

## 📈 7. Scaling Strategies

### A. Horizontal Pod Autoscaler (HPA)
```yaml
# hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: deepdetect-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deepdetect
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### B. Load Balancing Configuration
```nginx
# nginx.conf
upstream deepdetect_backend {
    least_conn;
    server deepdetect-1:5000 weight=1 max_fails=3 fail_timeout=30s;
    server deepdetect-2:5000 weight=1 max_fails=3 fail_timeout=30s;
    server deepdetect-3:5000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name deepdetect.yourdomain.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://deepdetect_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # File upload size
        client_max_body_size 50M;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://deepdetect_backend/health;
        access_log off;
    }
    
    # Static files
    location /static/ {
        alias /var/www/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## 🔍 8. Monitoring Dashboard

### A. Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "DeepDetect Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(deepdetect_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, deepdetect_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Model Predictions",
        "type": "pie",
        "targets": [
          {
            "expr": "deepdetect_predictions_total",
            "legendFormat": "{{label}}"
          }
        ]
      }
    ]
  }
}
```

## 🚀 9. Deployment Checklist

### Pre-deployment
- [ ] Model optimization completed
- [ ] Security hardening implemented
- [ ] Performance testing completed
- [ ] Monitoring setup configured
- [ ] Backup strategy defined
- [ ] SSL certificates obtained
- [ ] Environment variables configured
- [ ] Database migrations ready

### Deployment
- [ ] Infrastructure provisioned
- [ ] Application deployed
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] DNS records updated
- [ ] SSL/TLS enabled
- [ ] Monitoring alerts active
- [ ] Backup systems running

### Post-deployment
- [ ] Performance monitoring active
- [ ] Error tracking configured
- [ ] User feedback collection
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Incident response plan ready
- [ ] Scaling policies tested
- [ ] Disaster recovery tested

## 📞 Support & Maintenance

### Regular Maintenance Tasks
1. **Model Updates**: Retrain with new data monthly
2. **Security Patches**: Apply updates weekly
3. **Performance Monitoring**: Review metrics daily
4. **Backup Verification**: Test backups weekly
5. **Capacity Planning**: Review scaling monthly

### Troubleshooting Common Issues
1. **High Memory Usage**: Check for memory leaks, optimize batch sizes
2. **Slow Response Times**: Review database queries, check GPU utilization
3. **Authentication Failures**: Verify JWT tokens, check API keys
4. **Model Loading Errors**: Validate checkpoint files, check GPU memory

This deployment guide provides a comprehensive foundation for running DeepDetect in production environments with enterprise-grade reliability, security, and scalability.