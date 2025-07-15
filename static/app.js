// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const fileUploadArea = document.getElementById('fileUploadArea');
const filePreview = document.getElementById('filePreview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFile = document.getElementById('removeFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const resultContent = document.getElementById('resultContent');
const loadingSection = document.getElementById('loadingSection');

let selectedFile = null;

// File Upload Handlers
fileUploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('dragover');
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('dragover');
});

fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

removeFile.addEventListener('click', () => {
    clearFileSelection();
});

// File Selection Handler
function handleFileSelection(file) {
    selectedFile = file;
    
    // Update file preview
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    // Update preview icon based on file type
    const previewIcon = filePreview.querySelector('.preview-icon i');
    if (file.type.startsWith('image/')) {
        previewIcon.className = 'fas fa-file-image';
    } else if (file.type.startsWith('video/')) {
        previewIcon.className = 'fas fa-file-video';
    } else {
        previewIcon.className = 'fas fa-file';
    }
    
    // Show preview and enable button
    fileUploadArea.style.display = 'none';
    filePreview.style.display = 'block';
    analyzeBtn.disabled = false;
    
    // Hide previous results
    resultSection.style.display = 'none';
}

function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    
    // Reset UI
    fileUploadArea.style.display = 'block';
    filePreview.style.display = 'none';
    analyzeBtn.disabled = true;
    resultSection.style.display = 'none';
    loadingSection.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Form Submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
        showError("Please select a file.");
        return;
    }
    
    // Show loading state
    showLoading();
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            showResults(data);
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showError('Failed to analyze the file. Please try again.');
    } finally {
        hideLoading();
    }
});

function showLoading() {
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';
    analyzeBtn.disabled = true;
    
    // Animate loading steps
    const steps = loadingSection.querySelectorAll('.step');
    steps.forEach((step, index) => {
        setTimeout(() => {
            step.classList.add('active');
        }, index * 1000);
    });
}

function hideLoading() {
    loadingSection.style.display = 'none';
    analyzeBtn.disabled = false;
    
    // Reset loading steps
    const steps = loadingSection.querySelectorAll('.step');
    steps.forEach(step => {
        step.classList.remove('active');
    });
}

function showResults(data) {
    const { label, probabilities } = data;
    
    // Determine result class and icon
    let resultClass, resultIcon, resultTitle, resultDescription;
    
    switch (label) {
        case 'real':
            resultClass = 'real';
            resultIcon = 'fas fa-check-circle';
            resultTitle = 'Authentic Content';
            resultDescription = 'This media appears to be genuine and unmanipulated.';
            break;
        case 'possibly fake':
            resultClass = 'possibly-fake';
            resultIcon = 'fas fa-exclamation-triangle';
            resultTitle = 'Possibly Manipulated';
            resultDescription = 'This media shows signs of potential manipulation.';
            break;
        case 'fake':
            resultClass = 'fake';
            resultIcon = 'fas fa-times-circle';
            resultTitle = 'Manipulated Content';
            resultDescription = 'This media appears to be artificially generated or manipulated.';
            break;
        default:
            resultClass = 'fake';
            resultIcon = 'fas fa-question-circle';
            resultTitle = 'Unknown';
            resultDescription = 'Unable to determine authenticity.';
    }
    
    resultContent.innerHTML = `
        <div class="result-main ${resultClass}">
            <div class="result-icon">
                <i class="${resultIcon}"></i>
            </div>
            <div class="result-text">
                <h5>${resultTitle}</h5>
                <p>${resultDescription}</p>
            </div>
        </div>
        
        <div class="probability-bars">
            <div class="probability-item">
                <div class="probability-header">
                    <span class="probability-label">AI Generated</span>
                    <span class="probability-value">${(probabilities.ai_generated * 100).toFixed(1)}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill ai-generated" style="width: ${probabilities.ai_generated * 100}%"></div>
                </div>
            </div>
            
            <div class="probability-item">
                <div class="probability-header">
                    <span class="probability-label">Deepfake</span>
                    <span class="probability-value">${(probabilities.deepfake * 100).toFixed(1)}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill deepfake" style="width: ${probabilities.deepfake * 100}%"></div>
                </div>
            </div>
            
            <div class="probability-item">
                <div class="probability-header">
                    <span class="probability-label">Real</span>
                    <span class="probability-value">${(probabilities.real * 100).toFixed(1)}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill real" style="width: ${probabilities.real * 100}%"></div>
                </div>
            </div>
        </div>
    `;
    
    resultSection.style.display = 'block';
    
    // Animate probability bars
    setTimeout(() => {
        const fills = resultSection.querySelectorAll('.probability-fill');
        fills.forEach(fill => {
            fill.style.width = fill.style.width;
        });
    }, 100);
}

function showError(message) {
    resultContent.innerHTML = `
        <div class="result-main fake">
            <div class="result-icon">
                <i class="fas fa-exclamation-circle"></i>
            </div>
            <div class="result-text">
                <h5>Error</h5>
                <p>${message}</p>
            </div>
        </div>
    `;
    
    resultSection.style.display = 'block';
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Header scroll effect
window.addEventListener('scroll', () => {
    const header = document.querySelector('.header');
    if (window.scrollY > 100) {
        header.style.background = 'rgba(255, 255, 255, 0.98)';
        header.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
    } else {
        header.style.background = 'rgba(255, 255, 255, 0.95)';
        header.style.boxShadow = 'none';
    }
});