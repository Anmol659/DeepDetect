
# DeepDetect
Browser extension + AI backend to detect tampered or fake images

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server
```bash
python run_extension.py
```
Or manually:
```bash
cd backend
python app.py
```

### 3. Load Chrome Extension
1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `extension` folder from this project
5. The DeepDetect extension should now appear

### 4. Usage
- Click the DeepDetect icon in your Chrome toolbar
- Use the popup to scan pages or upload individual files
- The extension will automatically highlight suspicious images when auto-scan is enabled

## API Endpoints

- `GET /health` - Health check for extension connectivity
- `POST /analyze` - Analyze uploaded media files
- `GET /` - Web interface

## Extension Features

- **Auto-scan**: Automatically scan images on page load
- **Manual scan**: Click to scan current page
- **File upload**: Upload individual files for analysis
- **Visual indicators**: Highlight suspicious content
- **Confidence scores**: Show detection confidence levels

## Configuration

The extension can be configured through its popup interface:
- Backend server URL (default: http://localhost:5000)
- Auto-scan settings
- Confidence thresholds
- Visual highlighting options