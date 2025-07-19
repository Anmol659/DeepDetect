# 🚀 DeepDetect Deployment Guide

## 📋 Overview
This guide covers deploying DeepDetect using Docker containers on various cloud platforms.

**Note**: Netlify doesn't support Docker containers or backend services. Use the alternatives below.

## 🐳 Docker Deployment Options

### 1. **Local Docker Deployment**

#### Build and Run
```bash
# Build the Docker image
docker build -t deepdetect .

# Run the container
docker run -p 5000:5000 -v $(pwd)/checkpoints:/app/checkpoints deepdetect
```

#### Using Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. **Cloud Deployment Options**

#### **Option A: Railway (Recommended - Easy)**
1. Fork this repository to your GitHub
2. Go to [Railway.app](https://railway.app)
3. Connect your GitHub account
4. Deploy from repository
5. Railway will automatically detect the Dockerfile

```bash
# Add railway.json for configuration
{
  "build": {
    "builder": "dockerfile"
  },
  "deploy": {
    "startCommand": "python backend/app.py",
    "healthcheckPath": "/health"
  }
}
```

#### **Option B: Render**
1. Go to [Render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new Web Service
4. Use these settings:
   - **Build Command**: `docker build -t deepdetect .`
   - **Start Command**: `python backend/app.py`
   - **Port**: `5000`

#### **Option C: Google Cloud Run**
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/deepdetect

# Deploy to Cloud Run
gcloud run deploy deepdetect \
  --image gcr.io/YOUR_PROJECT_ID/deepdetect \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000 \
  --memory 2Gi \
  --cpu 2
```

#### **Option D: AWS ECS/Fargate**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker build -t deepdetect .
docker tag deepdetect:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/deepdetect:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/deepdetect:latest

# Deploy using ECS task definition
```

#### **Option E: DigitalOcean App Platform**
1. Connect your GitHub repository
2. DigitalOcean will auto-detect the Dockerfile
3. Configure:
   - **Port**: `5000`
   - **Health Check**: `/health`
   - **Instance Size**: Basic ($12/month minimum)

### 3. **Heroku Deployment**
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-deepdetect-app

# Set stack to container
heroku stack:set container -a your-deepdetect-app

# Deploy
git push heroku main
```

Create `heroku.yml`:
```yaml
build:
  docker:
    web: Dockerfile
run:
  web: python backend/app.py
```

## 🔧 **Environment Configuration**

### Required Environment Variables
```bash
FLASK_ENV=production
PYTHONPATH=/app
PORT=5000  # For some platforms
```

### Optional Environment Variables
```bash
MODEL_PATH=/app/checkpoints/best_model_3way.pth
MAX_FILE_SIZE=52428800  # 50MB
UPLOAD_FOLDER=/app/uploads
```

## 🌐 **Chrome Extension Configuration**

After deployment, update your Chrome extension:

1. **Get your deployment URL** (e.g., `https://your-app.railway.app`)

2. **Update extension manifest.json**:
```json
{
  "host_permissions": [
    "https://your-app.railway.app/*",
    "http://localhost:5000/*"
  ]
}
```

3. **Update extension settings**:
   - Open extension popup
   - Go to Settings tab
   - Change "Backend URL" to your deployment URL

## 📊 **Performance Optimization**

### For Production Deployment:
```dockerfile
# Add to Dockerfile for production
ENV FLASK_ENV=production
ENV PYTHONOPTIMIZE=1

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "backend.app:app"]
```

### Resource Requirements:
- **CPU**: 1-2 cores minimum
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 1GB minimum
- **Network**: HTTPS required for Chrome extension

## 🔒 **Security Considerations**

1. **HTTPS**: Required for Chrome extension to work
2. **CORS**: Properly configured for extension domains
3. **File Upload Limits**: 50MB maximum
4. **Rate Limiting**: Consider adding for production

## 🚨 **Troubleshooting**

### Common Issues:
1. **Model not loading**: Ensure `checkpoints/best_model_3way.pth` exists
2. **CORS errors**: Check host_permissions in manifest.json
3. **Memory issues**: Increase container memory to 2GB+
4. **Timeout errors**: Increase request timeout settings

### Health Check:
```bash
curl https://your-deployment-url/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0"
}
```

## 📈 **Monitoring**

Add monitoring endpoints:
- `/health` - Health check
- `/metrics` - Performance metrics (if implemented)
- Logs via platform-specific logging

## 💰 **Cost Estimates**

- **Railway**: ~$5-10/month
- **Render**: ~$7-25/month  
- **Google Cloud Run**: ~$5-15/month (pay per use)
- **DigitalOcean**: ~$12-24/month
- **Heroku**: ~$7-25/month

Choose based on your traffic and performance needs!