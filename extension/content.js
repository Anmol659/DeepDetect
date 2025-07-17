// Content script for DeepDetect extension
class DeepDetectContent {
    constructor() {
        this.serverUrl = 'http://localhost:5000';
        this.settings = {};
        this.processedImages = new Set();
        this.overlays = new Map();
        this.isScanning = false;
        this.scanResults = [];
        
        this.init();
    }

    async init() {
        console.log('DeepDetect content script initializing...');
        await this.loadSettings();
        this.setupImageObserver();
        
        // Wait for page to be fully loaded before auto-scanning
        if (document.readyState === 'complete') {
            if (this.settings.autoScan) {
                setTimeout(() => this.scanPageImages(), 1000);
            }
        } else {
            window.addEventListener('load', () => {
                if (this.settings.autoScan) {
                    setTimeout(() => this.scanPageImages(), 1000);
                }
            });
        }
        
        // Listen for messages from popup and background
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep message channel open for async responses
        });
        
        // Mark content script as loaded
        window.deepDetectLoaded = true;
        window.deepDetectContent = this;
        
        console.log('DeepDetect content script initialized');
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.sync.get({
                serverUrl: 'http://localhost:5000',
                autoScan: false,
                showConfidence: true,
                highlightSuspicious: true,
                confidenceThreshold: 0.7
            });
            
            this.settings = result;
            this.serverUrl = result.serverUrl;
            console.log('Settings loaded:', this.settings);
        } catch (error) {
            console.error('Failed to load settings:', error);
            this.settings = {
                autoScan: false,
                showConfidence: true,
                highlightSuspicious: true,
                confidenceThreshold: 0.7
            };
        }
    }

    setupImageObserver() {
        // Observer for dynamically loaded images
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        const images = node.tagName === 'IMG' ? [node] : node.querySelectorAll('img');
                        images.forEach(img => this.processNewImage(img));
                    }
                });
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Process existing images
        document.querySelectorAll('img').forEach(img => this.processNewImage(img));
    }

    processNewImage(img) {
        if (this.processedImages.has(img.src)) return;
        
        // Wait for image to load
        if (!img.complete) {
            img.addEventListener('load', () => this.processNewImage(img));
            return;
        }
        
        // Filter out small images, icons, etc.
        if (img.naturalWidth < 100 || img.naturalHeight < 100) return;
        if (img.src.startsWith('data:')) return;
        if (img.src.includes('icon') || img.src.includes('logo')) return;
        
        this.processedImages.add(img.src);
        
        if (this.settings.autoScan && !this.isScanning) {
            this.analyzeImage(img);
        }
    }

    async analyzeImage(imgElement) {
        try {
            // Show loading indicator
            this.showImageOverlay(imgElement, 'loading', 'Analyzing...');
            
            // Convert image to blob with error handling
            let blob;
            try {
                const response = await fetch(imgElement.src, {
                    mode: 'cors',
                    credentials: 'omit'
                });
                if (!response.ok) {
                    throw new Error(`Failed to fetch image: ${response.status}`);
                }
                blob = await response.blob();
            } catch (fetchError) {
                console.warn('Failed to fetch image directly, trying canvas conversion:', fetchError);
                blob = await this.convertImageToBlob(imgElement);
            }
            
            // Create form data
            const formData = new FormData();
            formData.append('file', blob, 'image.jpg');
            
            // Send to backend with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
            
            const analysisResponse = await fetch(`${this.serverUrl}/analyze`, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!analysisResponse.ok) {
                throw new Error(`Analysis failed: ${analysisResponse.status}`);
            }
            
            const result = await analysisResponse.json();
            this.handleAnalysisResult(imgElement, result);
            
            // Store result for popup access
            this.scanResults.push({
                src: imgElement.src,
                result: result,
                timestamp: Date.now()
            });
            
        } catch (error) {
            console.error('Image analysis error:', error);
            this.showImageOverlay(imgElement, 'error', 'Analysis failed');
        }
    }

    async convertImageToBlob(imgElement) {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = imgElement.naturalWidth;
            canvas.height = imgElement.naturalHeight;
            
            try {
                ctx.drawImage(imgElement, 0, 0);
                canvas.toBlob((blob) => {
                    if (blob) {
                        resolve(blob);
                    } else {
                        reject(new Error('Failed to convert image to blob'));
                    }
                }, 'image/jpeg', 0.8);
            } catch (error) {
                reject(error);
            }
        });
    }

    handleAnalysisResult(imgElement, result) {
        const maxProb = Math.max(
            result.class_probs?.ai_generated || 0,
            result.class_probs?.deepfake || 0,
            result.class_probs?.real || 0
        );
        const confidence = Math.round(maxProb * 100);
        const isSuspicious = result.label !== 'real';
        
        if (isSuspicious && this.settings.highlightSuspicious) {
            this.highlightSuspiciousImage(imgElement, result);
        }
        
        if (this.settings.showConfidence) {
            const status = result.label === 'real' ? 'authentic' : 
                          result.label === 'fake' ? 'suspicious' : 'uncertain';
            this.showImageOverlay(imgElement, status, `${confidence}% confidence`);
        } else {
            this.removeImageOverlay(imgElement);
        }
        
        // Store result for popup access
        imgElement.dataset.deepdetectResult = JSON.stringify(result);
    }

    highlightSuspiciousImage(imgElement, result) {
        imgElement.style.border = '3px solid #ef4444';
        imgElement.style.boxShadow = '0 0 10px rgba(239, 68, 68, 0.5)';
        imgElement.style.borderRadius = '4px';
        
        // Add warning icon
        const warningIcon = document.createElement('div');
        warningIcon.className = 'deepdetect-warning-icon';
        warningIcon.innerHTML = '⚠️';
        
        // Make parent relative if needed
        const parent = imgElement.parentElement;
        if (getComputedStyle(parent).position === 'static') {
            parent.style.position = 'relative';
        }
        
        parent.appendChild(warningIcon);
        
        // Add click handler for details
        warningIcon.addEventListener('click', () => {
            this.showDetailedResult(imgElement, result);
        });
    }

    showImageOverlay(imgElement, type, text) {
        this.removeImageOverlay(imgElement);
        
        const overlay = document.createElement('div');
        overlay.className = `deepdetect-overlay deepdetect-${type}`;
        overlay.textContent = text;
        
        // Make parent relative if needed
        const parent = imgElement.parentElement;
        if (getComputedStyle(parent).position === 'static') {
            parent.style.position = 'relative';
        }
        
        parent.appendChild(overlay);
        this.overlays.set(imgElement, overlay);
        
        // Auto-hide loading overlays
        if (type === 'loading') {
            setTimeout(() => {
                if (overlay.parentNode) {
                    overlay.remove();
                }
            }, 10000);
        }
    }

    removeImageOverlay(imgElement) {
        const existingOverlay = this.overlays.get(imgElement);
        if (existingOverlay && existingOverlay.parentNode) {
            existingOverlay.remove();
            this.overlays.delete(imgElement);
        }
    }

    showDetailedResult(imgElement, result) {
        const modal = document.createElement('div');
        modal.className = 'deepdetect-modal';
        
        const content = document.createElement('div');
        content.style.cssText = `
            background: #1a1a1a;
            color: white;
            padding: 24px;
            border-radius: 12px;
            max-width: 400px;
            width: 90%;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
        `;
        
        const confidence = Math.round((result.class_probs?.real || 0) * 100);
        const aiGenerated = Math.round((result.class_probs?.ai_generated || 0) * 100);
        const deepfake = Math.round((result.class_probs?.deepfake || 0) * 100);
        
        content.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <h3 style="margin: 0; font-size: 18px;">Analysis Result</h3>
                <button id="closeModal" style="background: none; border: none; color: #999; font-size: 20px; cursor: pointer;">×</button>
            </div>
            <div style="margin-bottom: 16px;">
                <img src="${imgElement.src}" style="width: 100%; max-height: 200px; object-fit: contain; border-radius: 8px;">
            </div>
            <div style="margin-bottom: 16px;">
                <div style="font-weight: 600; margin-bottom: 8px;">Classification: ${result.label}</div>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Real</span>
                        <span>${confidence}%</span>
                    </div>
                    <div style="background: #333; height: 6px; border-radius: 3px; overflow: hidden;">
                        <div style="background: #10b981; height: 100%; width: ${confidence}%; transition: width 0.3s;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>AI Generated</span>
                        <span>${aiGenerated}%</span>
                    </div>
                    <div style="background: #333; height: 6px; border-radius: 3px; overflow: hidden;">
                        <div style="background: #ef4444; height: 100%; width: ${aiGenerated}%; transition: width 0.3s;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span>Deepfake</span>
                        <span>${deepfake}%</span>
                    </div>
                    <div style="background: #333; height: 6px; border-radius: 3px; overflow: hidden;">
                        <div style="background: #f59e0b; height: 100%; width: ${deepfake}%; transition: width 0.3s;"></div>
                    </div>
                </div>
            </div>
            <div style="font-size: 12px; color: #999; text-align: center;">
                Analysis powered by DeepDetect AI
            </div>
        `;
        
        modal.appendChild(content);
        document.body.appendChild(modal);
        
        // Close handlers
        const closeBtn = content.querySelector('#closeModal');
        closeBtn.addEventListener('click', () => modal.remove());
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
        
        // Close on escape
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }

    async scanPageImages() {
        if (this.isScanning) {
            console.log('Scan already in progress');
            return { success: false, message: 'Scan already in progress' };
        }

        this.isScanning = true;
        console.log('Starting page scan...');

        try {
            const images = Array.from(document.querySelectorAll('img'))
                .filter(img => {
                    return img.complete &&
                           img.naturalWidth > 100 && 
                           img.naturalHeight > 100 && 
                           img.src && 
                           !img.src.startsWith('data:') &&
                           !img.src.includes('icon') &&
                           !img.src.includes('logo');
                });

            console.log(`Found ${images.length} images to analyze`);

            if (images.length === 0) {
                return { success: true, message: 'No images found to analyze', count: 0 };
            }

            let analyzed = 0;
            let suspicious = 0;

            // Send progress update
            chrome.runtime.sendMessage({
                action: 'scanProgress',
                total: images.length,
                analyzed: 0,
                suspicious: 0
            }).catch(() => {}); // Ignore errors if popup is closed

            for (let i = 0; i < images.length; i++) {
                const img = images[i];
                
                try {
                    await this.analyzeImage(img);
                    analyzed++;
                    
                    // Check if image was marked as suspicious
                    const result = img.dataset.deepdetectResult;
                    if (result) {
                        const parsedResult = JSON.parse(result);
                        if (parsedResult.label !== 'real') {
                            suspicious++;
                        }
                    }
                    
                    // Send progress update
                    chrome.runtime.sendMessage({
                        action: 'scanProgress',
                        total: images.length,
                        analyzed: analyzed,
                        suspicious: suspicious
                    }).catch(() => {}); // Ignore errors if popup is closed
                    
                } catch (error) {
                    console.error(`Error analyzing image ${i + 1}:`, error);
                }
                
                // Small delay to prevent overwhelming the server
                await new Promise(resolve => setTimeout(resolve, 100));
            }

            console.log(`Scan complete: ${analyzed} analyzed, ${suspicious} suspicious`);
            
            // Update badge
            chrome.runtime.sendMessage({
                action: 'updateBadge',
                count: analyzed,
                suspicious: suspicious
            }).catch(() => {});

            return { 
                success: true, 
                message: `Scan complete: ${analyzed} images analyzed, ${suspicious} suspicious`,
                count: analyzed,
                suspicious: suspicious
            };

        } catch (error) {
            console.error('Scan error:', error);
            return { success: false, message: error.message };
        } finally {
            this.isScanning = false;
        }
    }

    async handleMessage(request, sender, sendResponse) {
        console.log('Content script received message:', request);
        
        try {
            switch (request.action) {
                case 'scanPage':
                    const result = await this.scanPageImages();
                    sendResponse(result);
                    break;
                    
                case 'getPageImages':
                    const images = Array.from(document.querySelectorAll('img'))
                        .filter(img => {
                            return img.complete &&
                                   img.naturalWidth > 100 && 
                                   img.naturalHeight > 100 && 
                                   img.src && 
                                   !img.src.startsWith('data:');
                        })
                        .map(img => ({
                            src: img.src,
                            width: img.naturalWidth,
                            height: img.naturalHeight,
                            alt: img.alt || '',
                            result: img.dataset.deepdetectResult ? 
                                   JSON.parse(img.dataset.deepdetectResult) : null
                        }));
                    
                    sendResponse({ images });
                    break;
                    
                case 'updateSettings':
                    this.settings = { ...this.settings, ...request.settings };
                    this.serverUrl = request.settings.serverUrl || this.serverUrl;
                    sendResponse({ success: true });
                    break;

                case 'getScanResults':
                    sendResponse({ results: this.scanResults });
                    break;

                case 'clearResults':
                    this.scanResults = [];
                    // Clear all overlays
                    this.overlays.forEach(overlay => {
                        if (overlay.parentNode) overlay.remove();
                    });
                    this.overlays.clear();
                    sendResponse({ success: true });
                    break;
                    
                default:
                    sendResponse({ error: 'Unknown action' });
            }
        } catch (error) {
            console.error('Error handling message:', error);
            sendResponse({ error: error.message });
        }
    }
}

// Initialize content script with proper timing
function initializeContentScript() {
    if (window.deepDetectContent) {
        console.log('DeepDetect content script already initialized');
        return;
    }
    
    try {
        new DeepDetectContent();
    } catch (error) {
        console.error('Failed to initialize DeepDetect content script:', error);
    }
}

// Initialize based on document state
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeContentScript);
} else {
    initializeContentScript();
}

// Also initialize on window load as backup
window.addEventListener('load', () => {
    if (!window.deepDetectContent) {
        initializeContentScript();
    }
});