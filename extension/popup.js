// Popup script for DeepDetect extension
class DeepDetectPopup {
    constructor() {
        this.serverUrl = 'http://localhost:5000';
        this.isConnected = false;
        this.scanResults = [];
        this.currentFilter = 'all';
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.setupTabs();
        await this.loadSettings();
        await this.checkServerConnection();
        await this.loadScanResults();
        this.updateUI();
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Scanner controls
        document.getElementById('scanPageBtn').addEventListener('click', () => {
            this.scanCurrentPage();
        });

        document.getElementById('clearResultsBtn').addEventListener('click', () => {
            this.clearResults();
        });

        // File upload
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setFilter(e.target.dataset.filter);
            });
        });

        // Settings
        document.getElementById('serverUrl').addEventListener('change', (e) => {
            this.updateServerUrl(e.target.value);
        });

        document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
            this.updateConfidenceThreshold(e.target.value);
        });

        // Toggle switches
        document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.saveSettings();
            });
        });
    }

    setupTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabPanels = document.querySelectorAll('.tab-panel');

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.dataset.tab;
                
                // Update button states
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Update panel states
                tabPanels.forEach(panel => {
                    panel.classList.remove('active');
                    if (panel.id === targetTab) {
                        panel.classList.add('active');
                    }
                });
            });
        });
    }

    async checkServerConnection() {
        try {
            const response = await fetch(`${this.serverUrl}/health`, {
                method: 'GET',
                timeout: 5000
            });
            
            if (response.ok) {
                this.isConnected = true;
                this.updateConnectionStatus('connected', 'Connected');
            } else {
                throw new Error('Server not responding');
            }
        } catch (error) {
            this.isConnected = false;
            this.updateConnectionStatus('error', 'Disconnected');
            console.error('Server connection failed:', error);
        }
    }

    updateConnectionStatus(status, text) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.getElementById('statusText');
        
        statusDot.className = `status-dot ${status}`;
        statusText.textContent = text;
    }

    async scanCurrentPage() {
        if (!this.isConnected) {
            this.showToast('error', 'Connection Error', 'Backend server is not available');
            return;
        }

        const scanBtn = document.getElementById('scanPageBtn');
        const progressSection = document.getElementById('scanProgress');
        
        scanBtn.disabled = true;
        const originalContent = scanBtn.innerHTML;
        scanBtn.innerHTML = '<div class="spinner"></div> Scanning...';
        progressSection.style.display = 'block';
        this.updateProgress(0, 'Starting scan...');
        
        try {
            // Get current tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            this.updateProgress(10, 'Preparing page scan...');
            
            // Always inject content script to ensure it's loaded
            try {
                await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    files: ['content.js']
                });
                
                await chrome.scripting.insertCSS({
                    target: { tabId: tab.id },
                    files: ['content.css']
                });
                
                // Wait for content script to initialize
                await new Promise(resolve => setTimeout(resolve, 2000));
            } catch (injectionError) {
                console.error('Failed to inject content script:', injectionError);
                throw new Error('Failed to initialize page scanner');
            }
            
            this.updateProgress(30, 'Finding images on page...');
            
            // Get images directly from page
            const imageResults = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: () => {
                    // Wait for page to be ready
                    if (document.readyState !== 'complete') {
                        return { error: 'Page not fully loaded' };
                    }
                    
                    const images = Array.from(document.querySelectorAll('img'));
                    console.log(`Found ${images.length} total img elements`);
                    
                    const validImages = images.filter(img => {
                        // Check if image is loaded and has valid dimensions
                        const isLoaded = img.complete && img.naturalWidth > 0 && img.naturalHeight > 0;
                        const hasValidSize = img.naturalWidth >= 32 && img.naturalHeight >= 32;
                        const hasValidSrc = img.src && img.src.trim() !== '' && !img.src.startsWith('data:');
                        const isNotIcon = !img.src.toLowerCase().includes('icon') && 
                                         !img.src.toLowerCase().includes('logo') &&
                                         !img.src.toLowerCase().includes('favicon');
                        
                        console.log(`Image ${img.src}: loaded=${isLoaded}, size=${hasValidSize}, src=${hasValidSrc}, notIcon=${isNotIcon}`);
                        
                        return isLoaded && hasValidSize && hasValidSrc && isNotIcon;
                    });
                    
                    console.log(`Filtered to ${validImages.length} valid images`);
                    
                    return {
                        totalFound: images.length,
                        validImages: validImages.map(img => ({
                            src: img.src,
                            width: img.naturalWidth,
                            height: img.naturalHeight,
                            alt: img.alt || '',
                            className: img.className || '',
                            id: img.id || ''
                        }))
                    };
                }
            });
            
            const result = imageResults[0]?.result;
            
            if (result?.error) {
                throw new Error(result.error);
            }
            
            const images = result?.validImages || [];
            const totalFound = result?.totalFound || 0;
            
            console.log(`Scan results: ${totalFound} total images, ${images.length} valid for analysis`);
            
            this.updateStats(images.length, 0, 0);
            this.updateProgress(50, `Found ${images.length} images (${totalFound} total)`);
            
            if (images.length === 0) {
                this.showToast('warning', 'No Images Found', 
                    totalFound > 0 ? 
                    `Found ${totalFound} images but none are suitable for analysis (too small, icons, or data URLs)` :
                    'No images detected on this page');
                progressSection.style.display = 'none';
                scanBtn.disabled = false;
                scanBtn.innerHTML = originalContent;
                return;
            }

            this.updateProgress(70, 'Starting analysis...');
            
            // Analyze images one by one with progress updates
            let analyzed = 0;
            let suspicious = 0;
            
            for (let i = 0; i < images.length; i++) {
                const img = images[i];
                const progress = 70 + (i / images.length) * 25;
                this.updateProgress(progress, `Analyzing image ${i + 1}/${images.length}...`);
                
                try {
                    // Analyze image directly through backend
                    const analysisResult = await this.analyzeImageDirectly(img, tab.id);
                    
                    if (analysisResult?.success && analysisResult.result) {
                        analyzed++;
                        
                        // Add to results
                        this.scanResults.push({
                            id: Date.now() + i,
                            url: img.src,
                            thumbnail: img.src,
                            result: analysisResult.result,
                            timestamp: new Date(),
                            pageUrl: tab.url
                        });
                        
                        if (analysisResult.result.label !== 'real') {
                            suspicious++;
                        }
                        
                        // Update content script with result for visual overlay
                        chrome.tabs.sendMessage(tab.id, {
                            action: 'updateImageResult',
                            imageUrl: img.src,
                            result: analysisResult.result
                        }).catch(() => {
                            // Ignore if content script not available
                        });
                    }
                } catch (error) {
                    console.error(`Failed to analyze image ${i + 1}:`, error);
                }
                
                // Small delay between analyses
                await new Promise(resolve => setTimeout(resolve, 500));
            }
            
            this.updateProgress(100, `Scan complete: ${analyzed} analyzed`);
            await this.saveScanResults();
            this.updateResultsList();
            this.updateStats(images.length, analyzed, suspicious);
            
            this.showToast('success', 'Scan Complete', `Analyzed ${analyzed} images, found ${suspicious} suspicious`);
            
        } catch (error) {
            console.error('Scan error:', error);
            this.showToast('error', 'Scan Failed', `Error: ${error.message || 'Unknown error occurred'}`);
        } finally {
            progressSection.style.display = 'none';
            scanBtn.disabled = false;
            scanBtn.innerHTML = originalContent;
        }
    }

    async analyzeImageDirectly(imageData, tabId) {
        try {
            console.log(`Analyzing image directly: ${imageData.src}`);
            
            // Convert image URL to blob
            let blob;
            
            try {
                // Try to fetch the image directly
                const response = await fetch(imageData.src, {
                    method: 'GET',
                    mode: 'cors',
                    credentials: 'omit',
                    signal: AbortSignal.timeout(10000)
                });
                
                if (!response.ok) {
                    throw new Error(`Failed to fetch image: ${response.status}`);
                }
                
                blob = await response.blob();
                
                if (blob.size === 0) {
                    throw new Error('Empty image file');
                }
                
            } catch (fetchError) {
                console.warn('Direct fetch failed, trying canvas conversion:', fetchError);
                
                // Fallback: Use content script to convert image
                const canvasResult = await chrome.tabs.sendMessage(tabId, {
                    action: 'convertImageToBlob',
                    imageUrl: imageData.src
                });
                
                if (!canvasResult?.success || !canvasResult.blob) {
                    throw new Error('Failed to convert image to blob');
                }
                
                // Convert base64 to blob
                const base64Response = await fetch(canvasResult.blob);
                blob = await base64Response.blob();
            }
            
            // Validate blob
            if (!blob || blob.size === 0) {
                throw new Error('Invalid image blob');
            }
            
            // Ensure proper MIME type
            if (!blob.type.startsWith('image/')) {
                blob = new Blob([blob], { type: 'image/jpeg' });
            }
            
            console.log(`Sending to backend: ${blob.type}, size: ${blob.size}`);
            
            // Create form data
            const formData = new FormData();
            const filename = blob.type.includes('png') ? 'image.png' : 'image.jpg';
            formData.append('file', blob, filename);
            
            // Send to backend
            const response = await fetch(`${this.serverUrl}/analyze`, {
                method: 'POST',
                body: formData,
                signal: AbortSignal.timeout(30000)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Backend analysis failed: ${response.status} - ${errorText}`);
            }
            
            const result = await response.json();
            
            if (!result || (!result.class_probs && !result.probabilities)) {
                throw new Error('Invalid response from backend');
            }
            
            console.log(`Analysis successful: ${result.label} (${result.confidence})`);
            
            return { success: true, result };
            
        } catch (error) {
            console.error('Direct image analysis error:', error);
            return { success: false, error: error.message };
        }
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            console.log(`Selected file: ${file.name}, type: ${file.type}, size: ${file.size}`);
            await this.analyzeUploadedFile(file);
        }
    }

    async handleDrop(event) {
        event.preventDefault();
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            console.log(`Dropped file: ${files[0].name}, type: ${files[0].type}, size: ${files[0].size}`);
            await this.analyzeUploadedFile(files[0]);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        document.getElementById('uploadArea').classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
    }

    async analyzeUploadedFile(file) {
        if (!this.isConnected) {
            this.showToast('error', 'Connection Error', 'Backend server is not available');
            return;
        }

        const progressSection = document.getElementById('scanProgress');
        progressSection.style.display = 'block';
        this.updateProgress(0, 'Uploading file...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            this.updateProgress(50, 'Analyzing...');

            const response = await fetch(`${this.serverUrl}/analyze`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.status}`);
            }

            const result = await response.json();
            
            // Add to results
            this.scanResults.push({
                id: Date.now(),
                url: file.name,
                thumbnail: URL.createObjectURL(file),
                result: result,
                timestamp: new Date(),
                pageUrl: 'Uploaded file'
            });

            this.updateProgress(100, 'Analysis complete');
            await this.saveScanResults();
            this.updateResultsList();
            this.updateStats();

            const status = result.label === 'real' ? 'authentic' : 'suspicious';
            this.showToast('success', 'Analysis Complete', `File analyzed: ${status}`);

        } catch (error) {
            console.error('File analysis error:', error);
            this.showToast('error', 'Analysis Failed', 'Failed to analyze uploaded file');
        } finally {
            progressSection.style.display = 'none';
        }
    }

    updateProgress(percent, text) {
        document.getElementById('progressFill').style.width = `${percent}%`;
        document.getElementById('progressText').textContent = text;
        document.getElementById('progressPercent').textContent = `${Math.round(percent)}%`;
    }

    updateStats(found = null, scanned = null, suspicious = null) {
        if (found !== null) document.getElementById('imagesFound').textContent = found;
        if (scanned !== null) document.getElementById('imagesScanned').textContent = scanned;
        if (suspicious !== null) document.getElementById('suspiciousFound').textContent = suspicious;
        
        // If no parameters provided, calculate from results
        if (found === null && scanned === null && suspicious === null) {
            const total = this.scanResults.length;
            const suspiciousCount = this.scanResults.filter(r => r.result.label !== 'real').length;
            
            document.getElementById('imagesScanned').textContent = total;
            document.getElementById('suspiciousFound').textContent = suspiciousCount;
        }
    }

    setFilter(filter) {
        this.currentFilter = filter;
        
        // Update button states
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });
        
        this.updateResultsList();
    }

    updateResultsList() {
        const resultsList = document.getElementById('resultsList');
        let filteredResults = this.scanResults;

        // Apply filter
        if (this.currentFilter === 'suspicious') {
            filteredResults = this.scanResults.filter(r => r.result.label !== 'real');
        } else if (this.currentFilter === 'authentic') {
            filteredResults = this.scanResults.filter(r => r.result.label === 'real');
        }

        if (filteredResults.length === 0) {
            resultsList.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                        <circle cx="11" cy="11" r="8" stroke="currentColor" stroke-width="2"/>
                        <path d="21 21l-4.35-4.35" stroke="currentColor" stroke-width="2"/>
                    </svg>
                    <p>No results found</p>
                    <span>Try scanning some images first</span>
                </div>
            `;
            return;
        }

        resultsList.innerHTML = filteredResults.map(result => {
            const confidence = Math.round(result.result.probabilities.real * 100);
            const status = result.result.label === 'real' ? 'authentic' : 
                          result.result.label === 'fake' ? 'suspicious' : 'uncertain';
            
            return `
                <div class="result-item" data-id="${result.id}">
                    <img src="${result.thumbnail}" alt="Thumbnail" class="result-thumbnail">
                    <div class="result-info">
                        <div class="result-url">${result.url}</div>
                        <div class="result-status status-${status}">
                            ${status === 'authentic' ? '✓' : status === 'suspicious' ? '⚠' : '?'} 
                            ${status.charAt(0).toUpperCase() + status.slice(1)}
                        </div>
                    </div>
                    <div class="confidence-score">${confidence}%</div>
                </div>
            `;
        }).join('');
    }

    async clearResults() {
        this.scanResults = [];
        await this.saveScanResults();
        this.updateResultsList();
        this.updateStats(0, 0, 0);
        this.showToast('success', 'Results Cleared', 'All scan results have been cleared');
    }

    updateServerUrl(url) {
        this.serverUrl = url;
        this.saveSettings();
        this.checkServerConnection();
    }

    updateConfidenceThreshold(value) {
        const percent = Math.round(value * 100);
        document.querySelector('.range-value').textContent = `${percent}%`;
        this.saveSettings();
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.sync.get({
                serverUrl: ' https://deepdetect-api-nnp6.onrender.com',
                autoScan: false,
                showConfidence: true,
                highlightSuspicious: true,
                confidenceThreshold: 0.7
            });

            this.serverUrl = result.serverUrl;
            document.getElementById('serverUrl').value = result.serverUrl;
            document.getElementById('autoScan').checked = result.autoScan;
            document.getElementById('showConfidence').checked = result.showConfidence;
            document.getElementById('highlightSuspicious').checked = result.highlightSuspicious;
            document.getElementById('confidenceThreshold').value = result.confidenceThreshold;
            
            const percent = Math.round(result.confidenceThreshold * 100);
            document.querySelector('.range-value').textContent = `${percent}%`;
        } catch (error) {
            console.error('Failed to load settings:', error);
        }
    }

    async saveSettings() {
        try {
            await chrome.storage.sync.set({
                serverUrl: document.getElementById('serverUrl').value,
                autoScan: document.getElementById('autoScan').checked,
                showConfidence: document.getElementById('showConfidence').checked,
                highlightSuspicious: document.getElementById('highlightSuspicious').checked,
                confidenceThreshold: parseFloat(document.getElementById('confidenceThreshold').value)
            });
        } catch (error) {
            console.error('Failed to save settings:', error);
        }
    }

    async loadScanResults() {
        try {
            const result = await chrome.storage.local.get(['scanResults']);
            this.scanResults = result.scanResults || [];
        } catch (error) {
            console.error('Failed to load scan results:', error);
            this.scanResults = [];
        }
    }

    async saveScanResults() {
        try {
            await chrome.storage.local.set({ scanResults: this.scanResults });
        } catch (error) {
            console.error('Failed to save scan results:', error);
        }
    }

    updateUI() {
        this.updateStats();
        this.updateResultsList();
    }

    showToast(type, title, message) {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">×</button>
        `;
        
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            toast.remove();
        });
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 4 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 4000);
    }

    switchTab(tabName) {
        // This is handled by the setupTabs method
        // Just trigger the click event
        document.querySelector(`[data-tab="${tabName}"]`).click();
    }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DeepDetectPopup();
});