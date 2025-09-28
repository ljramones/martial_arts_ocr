/**
 * Upload functionality for Martial Arts OCR
 * Handles file uploads, drag-and-drop, progress tracking, and preview
 */

class UploadManager {
    constructor() {
        this.uploadZone = null;
        this.fileInput = null;
        this.previewContainer = null;
        this.progressContainer = null;
        this.uploadButton = null;
        this.maxFileSize = 16 * 1024 * 1024; // 16MB
        this.allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp'];
        this.currentFiles = [];
        this.uploadInProgress = false;

        this.init();
    }

    init() {
        this.setupElements();
        this.bindEvents();
        this.setupDragAndDrop();
    }

    setupElements() {
        this.uploadZone = document.querySelector('.upload-zone');
        this.fileInput = document.querySelector('#file-input') || this.createFileInput();
        this.previewContainer = document.querySelector('.file-preview') || this.createPreviewContainer();
        this.progressContainer = document.querySelector('.progress-container') || this.createProgressContainer();
        this.uploadButton = document.querySelector('.upload-button');

        // Create elements if they don't exist
        if (!this.uploadZone) {
            console.error('Upload zone not found');
            return;
        }
    }

    createFileInput() {
        const input = document.createElement('input');
        input.type = 'file';
        input.id = 'file-input';
        input.name = 'file';
        input.accept = 'image/*';
        input.multiple = false;
        input.className = 'file-input';
        this.uploadZone.appendChild(input);
        return input;
    }

    createPreviewContainer() {
        const container = document.createElement('div');
        container.className = 'file-preview';
        container.style.display = 'none';
        this.uploadZone.parentNode.insertBefore(container, this.uploadZone.nextSibling);
        return container;
    }

    createProgressContainer() {
        const container = document.createElement('div');
        container.className = 'progress-container';
        container.style.display = 'none';
        container.innerHTML = `
            <div class="processing-status">
                <div class="status-indicator processing"></div>
                <div class="status-text">Processing document...</div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
            <div class="progress-details">
                <span class="progress-step">Initializing...</span>
                <span class="progress-percent">0%</span>
            </div>
        `;
        this.uploadZone.parentNode.insertBefore(container, this.uploadZone.nextSibling);
        return container;
    }

    bindEvents() {
        // File input change
        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => {
                this.handleFileSelect(e.target.files);
            });
        }

        // Upload button click
        if (this.uploadButton) {
            this.uploadButton.addEventListener('click', (e) => {
                e.preventDefault();
                if (!this.uploadInProgress) {
                    this.fileInput.click();
                }
            });
        }

        // Upload zone click
        if (this.uploadZone) {
            this.uploadZone.addEventListener('click', (e) => {
                if (e.target === this.uploadZone && !this.uploadInProgress) {
                    this.fileInput.click();
                }
            });
        }

        // Form submission
        const uploadForm = document.querySelector('#upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleFormSubmission();
            });
        }
    }

    setupDragAndDrop() {
        if (!this.uploadZone) return;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadZone.addEventListener(eventName, () => {
                this.uploadZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadZone.addEventListener(eventName, () => {
                this.uploadZone.classList.remove('dragover');
            }, false);
        });

        // Handle dropped files
        this.uploadZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            this.handleFileSelect(files);
        }, false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleFileSelect(files) {
        if (!files || files.length === 0) return;

        const file = files[0]; // Only handle single file for now

        // Validate file
        const validation = this.validateFile(file);
        if (!validation.valid) {
            this.showError(validation.message);
            return;
        }

        // Store file and show preview
        this.currentFiles = [file];
        this.showFilePreview(file);
        this.updateUploadUI(true);
    }

    validateFile(file) {
        // Check file size
        if (file.size > this.maxFileSize) {
            return {
                valid: false,
                message: `File size (${this.formatFileSize(file.size)}) exceeds maximum allowed size (${this.formatFileSize(this.maxFileSize)})`
            };
        }

        // Check file type
        if (!this.allowedTypes.includes(file.type)) {
            return {
                valid: false,
                message: `File type "${file.type}" is not supported. Please upload JPG, PNG, TIFF, or BMP files.`
            };
        }

        // Check if it's actually an image
        if (!file.type.startsWith('image/')) {
            return {
                valid: false,
                message: 'Please upload an image file.'
            };
        }

        return { valid: true };
    }

    showFilePreview(file) {
        if (!this.previewContainer) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewContainer.innerHTML = `
                <div class="preview-header">
                    <h3>File Preview</h3>
                    <button type="button" class="btn btn-sm btn-secondary" onclick="uploadManager.clearFiles()">
                        Remove File
                    </button>
                </div>
                <div class="preview-content">
                    <img src="${e.target.result}" alt="Preview" class="preview-image" />
                    <div class="file-info">
                        <dl>
                            <dt>Filename:</dt>
                            <dd>${file.name}</dd>
                            <dt>Size:</dt>
                            <dd>${this.formatFileSize(file.size)}</dd>
                            <dt>Type:</dt>
                            <dd>${file.type}</dd>
                            <dt>Last Modified:</dt>
                            <dd>${new Date(file.lastModified).toLocaleString()}</dd>
                        </dl>
                    </div>
                </div>
                <div class="preview-actions">
                    <button type="button" class="btn btn-primary btn-lg" onclick="uploadManager.startUpload()">
                        Process Document
                    </button>
                </div>
            `;
            this.previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    updateUploadUI(hasFile) {
        if (!this.uploadZone) return;

        const uploadText = this.uploadZone.querySelector('.upload-text');
        const uploadSubtext = this.uploadZone.querySelector('.upload-subtext');

        if (hasFile) {
            if (uploadText) uploadText.textContent = 'File ready for processing';
            if (uploadSubtext) uploadSubtext.textContent = 'Click "Process Document" below to start OCR';
            this.uploadZone.classList.add('has-file');
        } else {
            if (uploadText) uploadText.textContent = 'Drag and drop your scanned document here';
            if (uploadSubtext) uploadSubtext.textContent = 'or click to browse files (JPG, PNG, TIFF, BMP)';
            this.uploadZone.classList.remove('has-file');
        }
    }

    async startUpload() {
        if (this.currentFiles.length === 0) {
            this.showError('No file selected');
            return;
        }

        if (this.uploadInProgress) {
            console.warn('Upload already in progress');
            return;
        }

        this.uploadInProgress = true;
        this.showProgress(true);

        try {
            const formData = new FormData();
            formData.append('file', this.currentFiles[0]);

            const response = await this.uploadWithProgress(formData);

            if (response.success) {
                this.handleUploadSuccess(response);
            } else {
                this.handleUploadError(response.error || 'Upload failed');
            }
        } catch (error) {
            this.handleUploadError(error.message);
        }
    }

    async uploadWithProgress(formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            // Track upload progress
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    this.updateProgress(percentComplete, 'Uploading file...');
                }
            });

            // Handle completion
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    try {
                        // For redirect responses, we need to handle differently
                        if (xhr.responseURL && xhr.responseURL.includes('/process/')) {
                            // Extract document ID from redirect URL
                            const match = xhr.responseURL.match(/\/process\/(\d+)/);
                            if (match) {
                                const documentId = match[1];
                                this.startProcessingMonitor(documentId);
                                resolve({ success: true, documentId });
                            } else {
                                window.location.href = xhr.responseURL;
                            }
                        } else {
                            const response = JSON.parse(xhr.responseText);
                            resolve(response);
                        }
                    } catch (e) {
                        // If it's a redirect, follow it
                        if (xhr.responseURL) {
                            window.location.href = xhr.responseURL;
                        } else {
                            reject(new Error('Invalid response format'));
                        }
                    }
                } else {
                    reject(new Error(`Upload failed with status ${xhr.status}`));
                }
            });

            // Handle errors
            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });

            xhr.addEventListener('timeout', () => {
                reject(new Error('Upload timeout'));
            });

            // Send request
            xhr.open('POST', '/upload');
            xhr.timeout = 120000; // 2 minutes
            xhr.send(formData);
        });
    }

    async startProcessingMonitor(documentId) {
        this.updateProgress(100, 'Upload complete. Starting OCR processing...');

        // Poll for processing status
        const pollInterval = 2000; // 2 seconds
        let attempts = 0;
        const maxAttempts = 60; // 2 minutes max

        const poll = async () => {
            try {
                const response = await fetch(`/api/status/${documentId}`);
                const data = await response.json();

                attempts++;

                switch (data.status) {
                    case 'processing':
                        this.updateProgress(25 + (attempts * 2), 'Processing document with OCR...');
                        if (attempts < maxAttempts) {
                            setTimeout(poll, pollInterval);
                        } else {
                            this.handleUploadError('Processing timeout');
                        }
                        break;

                    case 'completed':
                        this.updateProgress(100, 'Processing complete!');
                        setTimeout(() => {
                            window.location.href = `/view/${documentId}`;
                        }, 1000);
                        break;

                    case 'failed':
                        this.handleUploadError(data.error_message || 'Processing failed');
                        break;

                    default:
                        if (attempts < maxAttempts) {
                            setTimeout(poll, pollInterval);
                        } else {
                            this.handleUploadError('Unknown processing status');
                        }
                }
            } catch (error) {
                console.error('Status check error:', error);
                if (attempts < maxAttempts) {
                    setTimeout(poll, pollInterval);
                } else {
                    this.handleUploadError('Failed to check processing status');
                }
            }
        };

        // Start polling after a brief delay
        setTimeout(poll, 1000);
    }

    showProgress(show) {
        if (!this.progressContainer) return;

        this.progressContainer.style.display = show ? 'block' : 'none';
        if (this.previewContainer) {
            this.previewContainer.style.display = show ? 'none' : 'block';
        }
    }

    updateProgress(percent, message) {
        if (!this.progressContainer) return;

        const progressFill = this.progressContainer.querySelector('.progress-fill');
        const progressStep = this.progressContainer.querySelector('.progress-step');
        const progressPercent = this.progressContainer.querySelector('.progress-percent');

        if (progressFill) {
            progressFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
        }

        if (progressStep) {
            progressStep.textContent = message;
        }

        if (progressPercent) {
            progressPercent.textContent = `${Math.round(percent)}%`;
        }
    }

    handleUploadSuccess(response) {
        this.uploadInProgress = false;
        this.updateProgress(100, 'Upload successful!');

        // Redirect after brief delay
        setTimeout(() => {
            if (response.redirectUrl) {
                window.location.href = response.redirectUrl;
            } else if (response.documentId) {
                window.location.href = `/view/${response.documentId}`;
            } else {
                window.location.reload();
            }
        }, 1500);
    }

    handleUploadError(errorMessage) {
        this.uploadInProgress = false;
        this.showProgress(false);
        this.showError(errorMessage);

        // Reset upload state
        setTimeout(() => {
            this.clearFiles();
        }, 3000);
    }

    async handleFormSubmission() {
        // This method handles traditional form submission as fallback
        if (this.currentFiles.length === 0) {
            this.showError('Please select a file first');
            return;
        }

        await this.startUpload();
    }

    clearFiles() {
        this.currentFiles = [];
        this.uploadInProgress = false;

        if (this.fileInput) {
            this.fileInput.value = '';
        }

        if (this.previewContainer) {
            this.previewContainer.style.display = 'none';
            this.previewContainer.innerHTML = '';
        }

        if (this.progressContainer) {
            this.progressContainer.style.display = 'none';
        }

        this.updateUploadUI(false);
        this.clearMessages();
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showMessage(message, type = 'info') {
        // Remove existing messages
        this.clearMessages();

        const messageDiv = document.createElement('div');
        messageDiv.className = `alert alert-${type}`;
        messageDiv.textContent = message;

        // Add close button
        const closeButton = document.createElement('button');
        closeButton.className = 'message-close';
        closeButton.innerHTML = '×';
        closeButton.onclick = () => messageDiv.remove();
        messageDiv.appendChild(closeButton);

        // Insert at top of container
        const container = this.uploadZone.parentNode;
        container.insertBefore(messageDiv, container.firstChild);

        // Auto-remove after delay
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.remove();
            }
        }, 5000);
    }

    clearMessages() {
        const messages = document.querySelectorAll('.alert');
        messages.forEach(msg => msg.remove());
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Public methods for external use
    reset() {
        this.clearFiles();
    }

    setMaxFileSize(sizeInBytes) {
        this.maxFileSize = sizeInBytes;
    }

    addAllowedType(mimeType) {
        if (!this.allowedTypes.includes(mimeType)) {
            this.allowedTypes.push(mimeType);
        }
    }

    removeAllowedType(mimeType) {
        const index = this.allowedTypes.indexOf(mimeType);
        if (index > -1) {
            this.allowedTypes.splice(index, 1);
        }
    }
}

// Utility functions
function formatProcessingTime(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)} seconds`;
    } else {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.round(seconds % 60);
        return `${minutes}m ${remainingSeconds}s`;
    }
}

function createImagePreview(file, container) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.className = 'preview-image';
        img.style.maxWidth = '100%';
        img.style.height = 'auto';
        container.appendChild(img);
    };
    reader.readAsDataURL(file);
}

// Initialize upload manager when DOM is ready
let uploadManager;

document.addEventListener('DOMContentLoaded', function() {
    uploadManager = new UploadManager();

    // Global error handler for upload failures
    window.addEventListener('error', function(e) {
        if (uploadManager && uploadManager.uploadInProgress) {
            console.error('Global error during upload:', e.error);
            uploadManager.handleUploadError('An unexpected error occurred');
        }
    });

    // Handle browser back/forward buttons during upload
    window.addEventListener('beforeunload', function(e) {
        if (uploadManager && uploadManager.uploadInProgress) {
            e.preventDefault();
            e.returnValue = 'Upload in progress. Are you sure you want to leave?';
            return e.returnValue;
        }
    });
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UploadManager, formatProcessingTime, createImagePreview };
}