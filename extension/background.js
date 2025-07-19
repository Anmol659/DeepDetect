// Background service worker for DeepDetect extension
class DeepDetectBackground {
    constructor() {
        this.init();
    }

    init() {
        // Handle extension installation
        chrome.runtime.onInstalled.addListener((details) => {
            this.handleInstallation(details);
        });

        // Handle messages from content scripts and popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
        });

        // Handle tab updates
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            this.handleTabUpdate(tabId, changeInfo, tab);
        });

        // Handle storage changes
        chrome.storage.onChanged.addListener((changes, namespace) => {
            this.handleStorageChange(changes, namespace);
        });
    }

    handleInstallation(details) {
        if (details.reason === 'install') {
            // Set default settings on first install
            chrome.storage.sync.set({
                serverUrl: 'https://deepdetect-api.onrender.com',
                autoScan: false,
                showConfidence: true,
                highlightSuspicious: true,
                confidenceThreshold: 0.7
            });

            // Show welcome notification
            chrome.notifications.create({
                type: 'basic',
                //iconUrl: 'icons/icon48.png',
                title: 'DeepDetect Installed',
                message: 'AI-powered media authentication is now active. Click the extension icon to get started.'
            });
        }
    }

    async handleMessage(request, sender, sendResponse) {
        try {
            switch (request.action) {
                case 'checkServerConnection':
                    const isConnected = await this.checkServerConnection(request.serverUrl);
                    sendResponse({ connected: isConnected });
                    break;

                case 'analyzeImage':
                    const result = await this.analyzeImage(request.imageData, request.serverUrl);
                    sendResponse({ result });
                    break;

                case 'getTabImages':
                    const images = await this.getTabImages(sender.tab.id);
                    sendResponse({ images });
                    break;

                case 'updateBadge':
                    this.updateBadge(sender.tab.id, request.count, request.suspicious);
                    break;

                case 'scanPage':
                    await this.scanPage(sender.tab.id);
                    sendResponse({ success: true });
                    break;

                default:
                    sendResponse({ error: 'Unknown action' });
            }
        } catch (error) {
            console.error('Background script error:', error);
            sendResponse({ error: error.message });
        }

        return true; // Keep message channel open for async responses
    }

    async handleTabUpdate(tabId, changeInfo, tab) {
        if (changeInfo.status === 'complete' && tab.url) {
            // Check if auto-scan is enabled
            const settings = await chrome.storage.sync.get(['autoScan']);
            
            if (settings.autoScan) {
                // Small delay to ensure page is fully loaded
                setTimeout(async () => {
                    try {
                        // Check if content script is already loaded
                        const results = await chrome.scripting.executeScript({
                            target: { tabId },
                            function: () => window.deepDetectLoaded || false
                        });
                        
                        const isLoaded = results[0]?.result || false;
                        
                        if (!isLoaded) {
                            // Inject content script and CSS
                            await chrome.scripting.executeScript({
                                target: { tabId },
                                files: ['content.js']
                            });
                            
                            await chrome.scripting.insertCSS({
                                target: { tabId },
                                files: ['content.css']
                            });
                        }
                    } catch (error) {
                        console.error('Failed to inject content script:', error);
                    }
                }, 2000);
            }
        }
    }

    handleStorageChange(changes, namespace) {
        if (namespace === 'sync') {
            // Notify all content scripts about settings changes
            chrome.tabs.query({}, (tabs) => {
                tabs.forEach(tab => {
                    chrome.tabs.sendMessage(tab.id, {
                        action: 'updateSettings',
                        settings: changes
                    }).catch(() => {
                        // Ignore errors for tabs without content script
                    });
                });
            });
        }
    }

    async checkServerConnection(serverUrl) {
        try {
            const response = await fetch(`${serverUrl}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });
            if (response.ok) {
                const data = await response.json();
                return data.status === 'healthy';
            }
            return false;
        } catch (error) {
            console.error('Server connection check failed:', error);
            return false;
        }
    }

    async analyzeImage(imageData, serverUrl) {
        try {
            const formData = new FormData();
            
            if (imageData instanceof Blob) {
                formData.append('file', imageData, 'image.jpg');
            } else if (typeof imageData === 'string') {
                // Convert data URL to blob
                const response = await fetch(imageData);
                const blob = await response.blob();
                formData.append('file', blob, 'image.jpg');
            } else {
                throw new Error('Invalid image data format');
            }

            const response = await fetch(`${serverUrl}/analyze`, {
                method: 'POST',
                body: formData,
                signal: AbortSignal.timeout(30000) // 30 second timeout
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            throw new Error(`Image analysis failed: ${error.message}`);
        }
    }

    async getTabImages(tabId) {
        try {
            const results = await chrome.scripting.executeScript({
                target: { tabId },
                function: () => {
                    return Array.from(document.querySelectorAll('img'))
                        .filter(img => {
                            return img.naturalWidth > 100 && 
                                   img.naturalHeight > 100 && 
                                   img.src && 
                                   !img.src.startsWith('data:') &&
                                   img.complete;
                        })
                        .map(img => ({
                            src: img.src,
                            width: img.naturalWidth,
                            height: img.naturalHeight,
                            alt: img.alt || ''
                        }));
                }
            });

            return results[0]?.result || [];
        } catch (error) {
            console.error('Failed to get tab images:', error);
            return [];
        }
    }

    updateBadge(tabId, total, suspicious) {
        if (suspicious > 0) {
            chrome.action.setBadgeText({
                text: suspicious.toString(),
                tabId
            });
            chrome.action.setBadgeBackgroundColor({
                color: '#ef4444',
                tabId
            });
            chrome.action.setTitle({
                title: `DeepDetect: ${suspicious} suspicious image${suspicious > 1 ? 's' : ''} found`,
                tabId
            });
        } else if (total > 0) {
            chrome.action.setBadgeText({
                text: '✓',
                tabId
            });
            chrome.action.setBadgeBackgroundColor({
                color: '#10b981',
                tabId
            });
            chrome.action.setTitle({
                title: `DeepDetect: ${total} image${total > 1 ? 's' : ''} scanned, all authentic`,
                tabId
            });
        } else {
            chrome.action.setBadgeText({
                text: '',
                tabId
            });
            chrome.action.setTitle({
                title: 'DeepDetect: AI Media Authentication',
                tabId
            });
        }
    }

    async scanPage(tabId) {
        try {
            await chrome.scripting.executeScript({
                target: { tabId },
                function: () => {
                    if (window.deepDetectContent) {
                        window.deepDetectContent.scanPageImages();
                    }
                }
            });
        } catch (error) {
            console.error('Failed to scan page:', error);
        }
    }
}

// Initialize background script
new DeepDetectBackground();