// Content script for DeepDetect extension
class DeepDetectContent {
    constructor() {
        this.serverUrl = 'http://localhost:5000';
        this.settings = {};
        this.processedImages = new Set();
        this.overlays = new Map();
        this.isScanning = false;
        
        this.init();
    }

    async init() {
        await this.loadSettings();
        this.setupImageObserver();
        
        // Wait for page to be fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                setTimeout(() => {
                    if (this.settings.autoScan) {
                        this.scanPageImages();
                    }
                }, 1000);
            });
        } else {
            setTimeout(() => {
                if (this.settings.autoScan) {
                    this.scanPageImages();
                }
            }, 1000);
        }
        
        // Listen for messages from popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep message channel open for async responses
        });
        
        // Mark content script as loaded
        window.deepDetectLoaded = true;
        window.deepDetectContent = this;
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
                        images.forEach(img => {
                            // Wait for image to load
                            if (img.complete) {
                                this.processNewImage(img);
                            } else {
                                img.addEventListener('load', () => this.processNewImage(img));
                            }
                        });
                    }
                });
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Process existing images
        setTimeout(() => {
            document.querySelectorAll('img').forEach(img => {
                if (img.complete) {
                    this.processNewImage(img);
                } else {
                    img.addEventListener('load', () => this.processNewImage(img));
                }
            });
        }, 500);
    }

    processNewImage(img) {
        if (!img.src || this.processedImages.has(img.src)) return;
        
        // Filter out small images, icons, etc.
        if (img.naturalWidth < 50 || img.naturalHeight < 50) return;
        if (img.src.startsWith('data:')) return;
        if (img.src.includes('icon') || img.src.includes('logo')) return;
        
        this.processedImages.add(img.src);
        
        if (this.settings.autoScan) {
            // Add small delay to avoid overwhelming the server
            setTimeout(() => this.analyzeImage(img), Math.random() * 2000);
        }
    }

    async analyzeImage(imgElement) {
        if (this.isScanning) return;
        
        try {
            // Show loading indicator
            this.showImageOverlay(imgElement, 'loading', 'Analyzing...');
            
            // Convert image to blob with proper error handling
            const response = await fetch(imgElement.src, {
                mode: 'cors',
                credentials: 'omit'
            });
            
            if (!response.ok) {
                throw new Error(`Failed to fetch image: ${response.status}`);
            }
            
            const blob = await response.blob();
            
            // Check if blob is valid
            if (blob.size === 0) {
                throw new Error('Empty image file');
            }
            
            // Create form data
            const formData = new FormData();
            formData.append('file', blob, 'image.jpg');
            
            // Send to backend
            const analysisResponse = await fetch(`${this.serverUrl}/analyze`, {
                method: 'POST',
                body: formData
                signal: AbortSignal.timeout(15000) // 15 second timeout
            }
            )
            
            if (!analysisResponse.ok) {
                const errorText = await analysisResponse.text();
                throw new Error(`Analysis failed: ${analysisResponse.status} - ${errorText}`);
            }
            
            const result = await analysisResponse.json();
            
            if (!result || !result.class_probs) {
                throw new Error('Invalid response from server');
            }
            
            this.handleAnalysisResult(imgElement, result);
            
        } catch (error) {
            console.error('Image analysis error:', error);
            this.showImageOverlay(imgElement, 'error', `Error: ${error.message}`);
            
            // Remove overlay after 3 seconds for errors
            setTimeout(() => {
                this.removeImageOverlay(imgElement);
            }, 3000);
        }
    }

    handleAnalysisResult(imgElement, result) {
        const confidence = Math.round((result.class_probs?.real || result.probabilities?.real || 0) * 100);
        const isSuspicious = result.label !== 'real';
        
        if (isSuspicious && this.settings.highlightSuspicious) {
            this.highlightSuspiciousImage(imgElement, result);
        }
        
        if (this.settings.showConfidence) {
            const status = result.label === 'real' ? 'authentic' : 
                          result.label === 'ai_generated' ? 'suspicious' : 
                          result.label === 'deepfake' ? 'suspicious' : 'uncertain';
            this.showImageOverlay(imgElement, status, `${confidence}% confidence`);
        } else {
            this.removeImageOverlay(imgElement);
        }
        
        // Store result for popup access
        imgElement.dataset.deepdetectResult = JSON.stringify(result);
        
        // Notify background script about scan results
        chrome.runtime.sendMessage({
            action: 'updateBadge',
            suspicious: isSuspicious ? 1 : 0,
            total: 1
        }).catch(() => {
            // Ignore errors if background script is not available
        });
    }

    highlightSuspiciousImage(imgElement, result) {
        imgElement.style.border = '3px solid #ef4444';
        imgElement.style.boxShadow = '0 0 10px rgba(239, 68, 68, 0.5)';
        imgElement.style.borderRadius = '4px';
        
        // Add warning icon
        const warningIcon = document.createElement('div');
        warningIcon.className = 'deepdetect-warning-icon';
        warningIcon.innerHTML = '⚠️';
        warningIcon.style.cssText = `
            position: absolute;
            top: 5px;
            right: 5px;
            background: #ef4444;
            color: white;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            z-index: 1000;
            cursor: pointer;
        `;
        
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
        
        const colors = {
            loading: '#f59e0b',
            authentic: '#10b981',
            suspicious: '#ef4444',
            uncertain: '#f59e0b',
            error: '#6b7280'
        };
        
        overlay.style.cssText = `
            position: absolute;
            bottom: 5px;
            left: 5px;
            background: ${colors[type] || '#6b7280'};
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
            z-index: 1000;
            pointer-events: none;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;
        
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
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;
        
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
        
        const confidence = Math.round(result.probabilities.real * 100);
        const aiGenerated = Math.round(result.probabilities.ai_generated * 100);
        const deepfake = Math.round(result.probabilities.deepfake * 100);
        
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
            return;
        }
        
        this.isScanning = true;
        
        try {
            // Clear previous results
            this.processedImages.clear();
            
            // Find all images on the page
            const images = Array.from(document.querySelectorAll('img'))
                .filter(img => {
                    return img.naturalWidth > 50 && 
                           img.naturalHeight > 50 && 
                           img.src && 
                           !img.src.startsWith('data:') &&
                           !img.src.includes('icon') &&
                           !img.src.includes('logo') &&
                           img.complete;
                });
            
            console.log(`Found ${images.length} images to analyze`);
            
            if (images.length === 0) {
                console.log('No suitable images found on this page');
                return;
            }

            let analyzed = 0;
            let suspicious = 0;
            
            // Analyze images with controlled concurrency
            const batchSize = 3; // Process 3 images at a time
            for (let i = 0; i < images.length; i += batchSize) {
                const batch = images.slice(i, i + batchSize);
                
                await Promise.allSettled(
                    batch.map(async (img) => {
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
                        } catch (error) {
                            console.error('Error analyzing image:', error);
                        }
                    })
                );
                
                // Small delay between batches
                if (i + batchSize < images.length) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }
            
            console.log(`Scan complete: ${analyzed} analyzed, ${suspicious} suspicious`);
            
            // Update badge
            chrome.runtime.sendMessage({
                action: 'updateBadge',
                total: analyzed,
                suspicious: suspicious
            }).catch(() => {
                // Ignore errors if background script is not available
            });
            
        } catch (error) {
            console.error('Page scan error:', error);
        } finally {
            this.isScanning = false;
        }
        const images = Array.from(document.querySelectorAll('img'))
            .filter(img => {
                return img.naturalWidth > 100 && 
                       img.naturalHeight > 100 && 
                       img.src && 
                       !img.src.startsWith('data:') &&
                       img.complete;
            });
    }

    handleMessage(request, sender, sendResponse) {
        switch (request.action) {
            case 'scanPage':
                this.scanPageImages().then(() => {
                    sendResponse({ success: true });
                }).catch(error => {
                    sendResponse({ success: false, error: error.message });
                });
                return true; // Keep message channel open for async response
                
            case 'getPageImages':
                const images = Array.from(document.querySelectorAll('img'))
                    .filter(img => {
                        return img.naturalWidth > 50 && 
                               img.naturalHeight > 50 && 
                               img.src && 
                               !img.src.startsWith('data:') &&
                               !img.src.includes('icon') &&
                               !img.src.includes('logo') &&
                               img.complete;
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
                this.settings = request.settings;
                this.serverUrl = request.settings.serverUrl;
                sendResponse({ success: true });
                break;
                
            default:
                sendResponse({ error: 'Unknown action' });
        }
    }
}

// Initialize content script
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new DeepDetectContent();
    });
} else {
    new DeepDetectContent();
}