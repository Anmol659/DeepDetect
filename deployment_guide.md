# 🚀 DeepDetect Deployment Guide

## 📋 Quick Deployment Options

### 1. **Render.com (Recommended for Backend)**

#### Step 1: Prepare Your Repository
```bash
# Ensure your trained model is in checkpoints/
cp your_model.pth checkpoints/best_model_3way.pth

# Commit to Git
git add .
git commit -m "Add trained model for deployment"
git push origin main
```

#### Step 2: Deploy to Render
1. Go to [render.com](https://render.com) and sign up
2. Connect your GitHub repository
3. Create a new **Web Service**
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python backend/app.py`
   - **Environment**: Python 3.9
5. Add environment variables:
   ```
   FLASK_ENV=production
   PORT=5000
   ```
6. Deploy!

Your API will be available at: `https://your-app-name.onrender.com`

### 2. **Railway (Alternative)**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### 3. **Heroku**

```bash
# Install Heroku CLI
heroku create your-deepdetect-app
git push heroku main
```

## 🔧 Extension Configuration for Production

Update your Chrome extension to use the deployed backend:

1. Open `extension/popup.js`
2. Change the default server URL:
```javascript
this.serverUrl = 'https://your-app-name.onrender.com';
```

3. Update `extension/manifest.json` host permissions:
```json
"host_permissions": [
  "https://your-app-name.onrender.com/*"
]
```

## 📦 Production Checklist

- [ ] Trained model file (`best_model_3way.pth`) is included
- [ ] Environment variables are set
- [ ] CORS is properly configured
- [ ] Health check endpoint works
- [ ] Extension points to production URL
- [ ] SSL/HTTPS is enabled

## 🔍 Testing Deployment

```bash
# Test health endpoint
curl https://your-app-name.onrender.com/health

# Test analysis endpoint
curl -X POST -F "file=@test_image.jpg" https://your-app-name.onrender.com/analyze
```

## 💡 Performance Tips

1. **Model Optimization**: Use model quantization for faster inference
2. **Caching**: Implement Redis for repeated requests
3. **CDN**: Use CloudFlare for static assets
4. **Monitoring**: Set up logging and error tracking