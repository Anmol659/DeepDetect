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
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const mobileMenu = document.getElementById('mobileMenu');
const toastContainer = document.getElementById('toastContainer');

let selectedFile = null;

// Mobile Menu Toggle
mobileMenuBtn?.addEventListener('click', () => {
    mobileMenu.classList.toggle('active');
});

// Close mobile menu when clicking outside
mobileMenu?.addEventListener('click', (e) => {
    if (e.target === mobileMenu) {
        mobileMenu.classList.remove('active');
    }
});

// Close mobile menu when clicking nav links
document.querySelectorAll('.mobile-nav-link').forEach(link => {
    link.addEventListener('click', () => {
        mobileMenu.classList.remove('active');
    });
});

// File Upload Handlers
fileUploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('dragover');
});

fileUploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    if (!fileUploadArea.contains(e.relatedTarget)) {
        fileUploadArea.classList.remove('dragover');
    }
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
    // Validate file size (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB in bytes
    if (file.size > maxSize) {
        showToast('error', 'File Too Large', 'Please select a file smaller than 50MB.');
        return;
    }

    // Validate file type
    const allowedTypes = [
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp',
        'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/flv'
    ];
    
    if (!allowedTypes.includes(file.type)) {
        showToast('error', 'Invalid File Type', 'Please select a valid image or video file.');
        return;
    }

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
    
    showToast('success', 'File Selected', `${file.name} is ready for analysis.`);
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
        showToast('error', 'No File Selected', 'Please select a file to analyze.');
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
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            showToast('error', 'Analysis Failed', data.error);
            showError(data.error);
        } else {
            showResults(data);
            showToast('success', 'Analysis Complete', 'Your media has been successfully analyzed.');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        const errorMessage = error.message || 'Failed to analyze the file. Please try again.';
        showToast('error', 'Connection Error', errorMessage);
        showError(errorMessage);
    } finally {
        hideLoading();
    }
});

function showLoading() {
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';
    
    // Update button state
    const btnContent = analyzeBtn.querySelector('.btn-content');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    btnContent.style.display = 'none';
    btnLoader.style.display = 'flex';
    analyzeBtn.disabled = true;
    
    // Animate loading steps
    const steps = ['step1', 'step2', 'step3'];
    steps.forEach((stepId, index) => {
        setTimeout(() => {
            const step = document.getElementById(stepId);
            if (step) {
                step.classList.add('active');
            }
        }, index * 1500);
    });
}

function hideLoading() {
    loadingSection.style.display = 'none';
    
    // Reset button state
    const btnContent = analyzeBtn.querySelector('.btn-content');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    btnContent.style.display = 'flex';
    btnLoader.style.display = 'none';
    analyzeBtn.disabled = false;
    
    // Reset loading steps
    const steps = ['step1', 'step2', 'step3'];
    steps.forEach(stepId => {
        const step = document.getElementById(stepId);
        if (step) {
            step.classList.remove('active');
        }
    });
}

function showResults(data) {
    const { label, probabilities } = data;
    
    // Determine result class and content
    let resultClass, resultIcon, resultTitle, resultDescription;
    
    switch (label) {
        case 'real':
            resultClass = 'real';
            resultIcon = 'fas fa-check-circle';
            resultTitle = 'Authentic Content';
            resultDescription = 'This media appears to be genuine and unmanipulated. Our AI analysis indicates a high probability of authenticity.';
            break;
        case 'possibly fake':
            resultClass = 'possibly-fake';
            resultIcon = 'fas fa-exclamation-triangle';
            resultTitle = 'Possibly Manipulated';
            resultDescription = 'This media shows signs of potential manipulation. Please exercise caution and verify from additional sources.';
            break;
        case 'fake':
            resultClass = 'fake';
            resultIcon = 'fas fa-times-circle';
            resultTitle = 'Manipulated Content';
            resultDescription = 'This media appears to be artificially generated or significantly manipulated. High confidence in detection.';
            break;
        default:
            resultClass = 'fake';
            resultIcon = 'fas fa-question-circle';
            resultTitle = 'Analysis Inconclusive';
            resultDescription = 'Unable to determine authenticity with confidence. Consider additional verification methods.';
    }
    
    // Calculate confidence percentage
    const maxProb = Math.max(probabilities.ai_generated, probabilities.deepfake, probabilities.real);
    const confidencePercent = (maxProb * 100).toFixed(1);
    
    resultContent.innerHTML = `
        <div class="result-main ${resultClass}">
            <div class="result-icon">
                <i class="${resultIcon}"></i>
            </div>
            <div class="result-text">
                <h5>${resultTitle}</h5>
                <p>${resultDescription}</p>
                <div style="margin-top: 12px; font-size: 14px; color: var(--gray-600);">
                    <strong>Confidence: ${confidencePercent}%</strong>
                </div>
            </div>
        </div>
        
        <div class="probability-bars">
            <div class="probability-item">
                <div class="probability-header">
                    <span class="probability-label">
                        <i class="fas fa-robot" style="margin-right: 8px; color: var(--error-color);"></i>
                        AI Generated
                    </span>
                    <span class="probability-value">${(probabilities.ai_generated * 100).toFixed(1)}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill ai-generated" style="width: ${probabilities.ai_generated * 100}%"></div>
                </div>
            </div>
            
            <div class="probability-item">
                <div class="probability-header">
                    <span class="probability-label">
                        <i class="fas fa-user-secret" style="margin-right: 8px; color: var(--warning-color);"></i>
                        Deepfake
                    </span>
                    <span class="probability-value">${(probabilities.deepfake * 100).toFixed(1)}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill deepfake" style="width: ${probabilities.deepfake * 100}%"></div>
                </div>
            </div>
            
            <div class="probability-item">
                <div class="probability-header">
                    <span class="probability-label">
                        <i class="fas fa-certificate" style="margin-right: 8px; color: var(--success-color);"></i>
                        Real
                    </span>
                    <span class="probability-value">${(probabilities.real * 100).toFixed(1)}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill real" style="width: ${probabilities.real * 100}%"></div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 24px; padding: 16px; background: var(--gray-100); border-radius: var(--border-radius-md); font-size: 14px; color: var(--gray-600);">
            <i class="fas fa-info-circle" style="margin-right: 8px; color: var(--primary-color);"></i>
            <strong>Note:</strong> This analysis is based on AI detection algorithms and should be used as a guide. For critical decisions, consider additional verification methods.
        </div>
    `;
    
    resultSection.style.display = 'block';
    
    // Smooth scroll to results
    setTimeout(() => {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
    
    // Animate probability bars
    setTimeout(() => {
        const fills = resultSection.querySelectorAll('.probability-fill');
        fills.forEach(fill => {
            const width = fill.style.width;
            fill.style.width = '0%';
            setTimeout(() => {
                fill.style.width = width;
            }, 100);
        });
    }, 200);
}

function showError(message) {
    resultContent.innerHTML = `
        <div class="result-main fake">
            <div class="result-icon">
                <i class="fas fa-exclamation-circle"></i>
            </div>
            <div class="result-text">
                <h5>Analysis Error</h5>
                <p>${message}</p>
                <div style="margin-top: 16px;">
                    <button onclick="clearFileSelection()" style="background: var(--primary-color); color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px;">
                        Try Again
                    </button>
                </div>
            </div>
        </div>
    `;
    
    resultSection.style.display = 'block';
}

// Toast Notification System
function showToast(type, title, message) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const iconMap = {
        success: 'fas fa-check-circle',
        error: 'fas fa-times-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };
    
    toast.innerHTML = `
        <div class="toast-icon">
            <i class="${iconMap[type] || iconMap.info}"></i>
        </div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add close functionality
    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => {
        removeToast(toast);
    });
    
    // Add to container
    toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        removeToast(toast);
    }, 5000);
}

function removeToast(toast) {
    if (toast && toast.parentNode) {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }
}

// Add slideOut animation to CSS dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const headerHeight = document.querySelector('.header').offsetHeight;
            const targetPosition = target.offsetTop - headerHeight - 20;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Header scroll effect
let lastScrollY = window.scrollY;

window.addEventListener('scroll', () => {
    const header = document.querySelector('.header');
    const currentScrollY = window.scrollY;
    
    if (currentScrollY > 100) {
        header.style.background = 'rgba(255, 255, 255, 0.98)';
        header.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
    } else {
        header.style.background = 'rgba(255, 255, 255, 0.95)';
        header.style.boxShadow = 'none';
    }
    
    // Hide/show header on scroll
    if (currentScrollY > lastScrollY && currentScrollY > 200) {
        header.style.transform = 'translateY(-100%)';
    } else {
        header.style.transform = 'translateY(0)';
    }
    
    lastScrollY = currentScrollY;
});

// Add intersection observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', () => {
    const animateElements = document.querySelectorAll('.feature-card, .step-card, .tech-item');
    animateElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Escape key to close mobile menu
    if (e.key === 'Escape' && mobileMenu.classList.contains('active')) {
        mobileMenu.classList.remove('active');
    }
    
    // Ctrl/Cmd + U to upload file
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        fileInput.click();
    }
});

// Performance optimization: Debounce scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Apply debouncing to scroll handler
const debouncedScrollHandler = debounce(() => {
    // Any additional scroll handling can go here
}, 10);

window.addEventListener('scroll', debouncedScrollHandler);

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    // Show welcome toast
    setTimeout(() => {
        showToast('info', 'Welcome to DeepDetect', 'Upload an image or video to start analyzing for AI manipulation.');
    }, 1000);
    
    // Check if backend is available
    fetch('/analyze', { method: 'OPTIONS' })
        .catch(() => {
            showToast('warning', 'Backend Connection', 'Make sure the Python backend server is running.');
        });
});