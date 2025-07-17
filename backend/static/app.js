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
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        showToast('error', 'File Too Large', 'Please select a file smaller than 50MB.');
        return;
    }

    const allowedTypes = [
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp',
        'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/flv'
    ];
    if (!allowedTypes.includes(file.type)) {
        showToast('error', 'Invalid File Type', 'Please select a valid image or video file.');
        return;
    }

    selectedFile = file;

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    const previewIcon = filePreview.querySelector('.preview-icon i');
    if (file.type.startsWith('image/')) {
        previewIcon.className = 'fas fa-file-image';
    } else if (file.type.startsWith('video/')) {
        previewIcon.className = 'fas fa-file-video';
    } else {
        previewIcon.className = 'fas fa-file';
    }

    fileUploadArea.style.display = 'none';
    filePreview.style.display = 'block';
    analyzeBtn.disabled = false;

    resultSection.style.display = 'none';

    showToast('success', 'File Selected', `${file.name} is ready for analysis.`);
}

function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';

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

    showLoading();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // ✅ Use relative path here
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

    const btnContent = analyzeBtn.querySelector('.btn-content');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    btnContent.style.display = 'none';
    btnLoader.style.display = 'flex';
    analyzeBtn.disabled = true;

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

    const btnContent = analyzeBtn.querySelector('.btn-content');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    btnContent.style.display = 'flex';
    btnLoader.style.display = 'none';
    analyzeBtn.disabled = false;

    const steps = ['step1', 'step2', 'step3'];
    steps.forEach(stepId => {
        const step = document.getElementById(stepId);
        if (step) {
            step.classList.remove('active');
        }
    });
}

function showResults(data) {
    const { label, class_probs:probabilities } = data;

    let resultClass, resultIcon, resultTitle, resultDescription;

    switch (label) {
        case 'real':
            resultClass = 'real';
            resultIcon = 'fas fa-check-circle';
            resultTitle = 'Authentic Content (Real)';
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
            resultDescription = 'This media appears to be artificially generated, Deepfake or significantly manipulated.';
            break;
        default:
            resultClass = 'fake';
            resultIcon = 'fas fa-question-circle';
            resultTitle = 'Manipulated Content(Possible Deepfake/GAN generated)';
            resultDescription = 'This media appears to be artificially generated, Deepfake or significantly manipulated.';
    }

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
                <div><strong>Confidence: ${confidencePercent}%</strong></div>
            </div>
        </div>
    `;

    resultSection.style.display = 'block';
    setTimeout(() => {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
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
            </div>
        </div>
    `;
    resultSection.style.display = 'block';
}

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
        <button class="toast-close"><i class="fas fa-times"></i></button>
    `;
    toast.querySelector('.toast-close').addEventListener('click', () => removeToast(toast));
    toastContainer.appendChild(toast);
    setTimeout(() => removeToast(toast), 5000);
}

function removeToast(toast) {
    if (toast && toast.parentNode) {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }
}
