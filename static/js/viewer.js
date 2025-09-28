/**
 * Document Viewer functionality for Martial Arts OCR
 * Handles document display, zoom, search, Japanese text interactions, and image viewing
 */

class DocumentViewer {
    constructor() {
        this.currentZoom = 1.0;
        this.minZoom = 0.25;
        this.maxZoom = 4.0;
        this.zoomStep = 0.25;
        this.viewMode = 'original'; // 'original', 'text-only', 'side-by-side'
        this.isFullscreen = false;
        this.sidebarOpen = true;
        this.searchResults = [];
        this.currentSearchIndex = 0;
        this.japaneseTooltips = new Map();

        this.init();
    }

    init() {
        this.setupElements();
        this.bindEvents();
        this.setupKeyboardShortcuts();
        this.initializeJapaneseInteractions();
        this.setupImageInteractions();
        this.loadViewerState();
    }

    setupElements() {
        // Main viewer elements
        this.viewer = document.querySelector('.document-viewer');
        this.toolbar = document.querySelector('.viewer-toolbar');
        this.sidebar = document.querySelector('.viewer-sidebar');
        this.content = document.querySelector('.viewer-content');
        this.contentContainer = document.querySelector('.content-container');

        // Controls
        this.zoomInBtn = document.querySelector('.zoom-in');
        this.zoomOutBtn = document.querySelector('.zoom-out');
        this.zoomResetBtn = document.querySelector('.zoom-reset');
        this.zoomLevel = document.querySelector('.zoom-level');

        // View mode buttons
        this.originalViewBtn = document.querySelector('.view-original');
        this.textOnlyBtn = document.querySelector('.view-text-only');
        this.sideBySideBtn = document.querySelector('.view-side-by-side');

        // Other controls
        this.fullscreenBtn = document.querySelector('.fullscreen-btn');
        this.sidebarToggle = document.querySelector('.sidebar-toggle');
        this.printBtn = document.querySelector('.print-btn');
        this.searchInput = document.querySelector('.search-input');

        // Create missing elements if needed
        this.ensureRequiredElements();
    }

    ensureRequiredElements() {
        // Create zoom level display if missing
        if (!this.zoomLevel && this.toolbar) {
            const zoomControls = this.toolbar.querySelector('.zoom-controls');
            if (zoomControls) {
                const levelSpan = document.createElement('span');
                levelSpan.className = 'zoom-level';
                levelSpan.textContent = '100%';
                zoomControls.appendChild(levelSpan);
                this.zoomLevel = levelSpan;
            }
        }

        // Create search results container if missing
        if (this.searchInput && !document.querySelector('.search-results')) {
            const resultsDiv = document.createElement('div');
            resultsDiv.className = 'search-results';
            this.searchInput.parentNode.appendChild(resultsDiv);
        }
    }

    bindEvents() {
        // Zoom controls
        if (this.zoomInBtn) {
            this.zoomInBtn.addEventListener('click', () => this.zoomIn());
        }
        if (this.zoomOutBtn) {
            this.zoomOutBtn.addEventListener('click', () => this.zoomOut());
        }
        if (this.zoomResetBtn) {
            this.zoomResetBtn.addEventListener('click', () => this.resetZoom());
        }

        // View mode controls
        if (this.originalViewBtn) {
            this.originalViewBtn.addEventListener('click', () => this.setViewMode('original'));
        }
        if (this.textOnlyBtn) {
            this.textOnlyBtn.addEventListener('click', () => this.setViewMode('text-only'));
        }
        if (this.sideBySideBtn) {
            this.sideBySideBtn.addEventListener('click', () => this.setViewMode('side-by-side'));
        }

        // Other controls
        if (this.fullscreenBtn) {
            this.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        }
        if (this.sidebarToggle) {
            this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        }
        if (this.printBtn) {
            this.printBtn.addEventListener('click', () => this.printDocument());
        }

        // Search functionality
        if (this.searchInput) {
            this.searchInput.addEventListener('input', (e) => {
                this.debounceSearch(e.target.value);
            });
            this.searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.navigateSearch(e.shiftKey ? -1 : 1);
                }
            });
        }

        // Mouse wheel zoom (with Ctrl/Cmd)
        if (this.content) {
            this.content.addEventListener('wheel', (e) => {
                if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    if (e.deltaY < 0) {
                        this.zoomIn();
                    } else {
                        this.zoomOut();
                    }
                }
            });
        }

        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Fullscreen change
        document.addEventListener('fullscreenchange', () => {
            this.handleFullscreenChange();
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Don't interfere with input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            switch (e.key) {
                case '+':
                case '=':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.zoomIn();
                    }
                    break;

                case '-':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.zoomOut();
                    }
                    break;

                case '0':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.resetZoom();
                    }
                    break;

                case 'f':
                case 'F11':
                    if (e.key === 'F11' || (e.key === 'f' && !e.ctrlKey)) {
                        e.preventDefault();
                        this.toggleFullscreen();
                    }
                    break;

                case 'Escape':
                    if (this.isFullscreen) {
                        this.exitFullscreen();
                    }
                    this.clearSearch();
                    break;

                case '/':
                    if (!e.ctrlKey && !e.metaKey) {
                        e.preventDefault();
                        this.focusSearch();
                    }
                    break;

                case 's':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.toggleSidebar();
                    }
                    break;

                case 'p':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.printDocument();
                    }
                    break;
            }
        });
    }

    // Zoom functionality
    zoomIn() {
        const newZoom = Math.min(this.maxZoom, this.currentZoom + this.zoomStep);
        this.setZoom(newZoom);
    }

    zoomOut() {
        const newZoom = Math.max(this.minZoom, this.currentZoom - this.zoomStep);
        this.setZoom(newZoom);
    }

    resetZoom() {
        this.setZoom(1.0);
    }

    setZoom(zoom) {
        this.currentZoom = zoom;

        if (this.contentContainer) {
            this.contentContainer.style.transform = `scale(${zoom})`;
            this.contentContainer.style.transformOrigin = 'top left';
        }

        if (this.zoomLevel) {
            this.zoomLevel.textContent = `${Math.round(zoom * 100)}%`;
        }

        this.updateZoomButtons();
        this.saveViewerState();
    }

    updateZoomButtons() {
        if (this.zoomInBtn) {
            this.zoomInBtn.disabled = this.currentZoom >= this.maxZoom;
        }
        if (this.zoomOutBtn) {
            this.zoomOutBtn.disabled = this.currentZoom <= this.minZoom;
        }
    }

    // View mode functionality
    setViewMode(mode) {
        this.viewMode = mode;

        if (this.contentContainer) {
            this.contentContainer.className = `content-container ${mode}-view`;
        }

        this.updateViewModeButtons();
        this.saveViewerState();
    }

    updateViewModeButtons() {
        [this.originalViewBtn, this.textOnlyBtn, this.sideBySideBtn].forEach(btn => {
            if (btn) btn.classList.remove('active');
        });

        switch (this.viewMode) {
            case 'original':
                if (this.originalViewBtn) this.originalViewBtn.classList.add('active');
                break;
            case 'text-only':
                if (this.textOnlyBtn) this.textOnlyBtn.classList.add('active');
                break;
            case 'side-by-side':
                if (this.sideBySideBtn) this.sideBySideBtn.classList.add('active');
                break;
        }
    }

    // Fullscreen functionality
    toggleFullscreen() {
        if (this.isFullscreen) {
            this.exitFullscreen();
        } else {
            this.enterFullscreen();
        }
    }

    enterFullscreen() {
        if (this.viewer && this.viewer.requestFullscreen) {
            this.viewer.requestFullscreen();
        }
    }

    exitFullscreen() {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }

    handleFullscreenChange() {
        this.isFullscreen = !!document.fullscreenElement;

        if (this.viewer) {
            this.viewer.classList.toggle('viewer-fullscreen', this.isFullscreen);
        }

        if (this.fullscreenBtn) {
            this.fullscreenBtn.textContent = this.isFullscreen ? 'Exit Fullscreen' : 'Fullscreen';
            this.fullscreenBtn.title = this.isFullscreen ? 'Exit fullscreen (F11 or Esc)' : 'Enter fullscreen (F11)';
        }
    }

    // Sidebar functionality
    toggleSidebar() {
        this.sidebarOpen = !this.sidebarOpen;

        if (this.sidebar) {
            this.sidebar.classList.toggle('collapsed', !this.sidebarOpen);
        }

        if (this.sidebarToggle) {
            this.sidebarToggle.textContent = this.sidebarOpen ? 'Hide Sidebar' : 'Show Sidebar';
        }

        this.saveViewerState();
    }

    // Search functionality
    debounceSearch(query) {
        clearTimeout(this.searchTimeout);
        this.searchTimeout = setTimeout(() => {
            this.performSearch(query);
        }, 300);
    }

    performSearch(query) {
        this.clearSearchHighlights();

        if (!query || query.length < 2) {
            this.updateSearchResults(0, 0);
            return;
        }

        const content = this.content || document;
        const textNodes = this.getTextNodes(content);
        this.searchResults = [];

        const regex = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');

        textNodes.forEach(node => {
            const text = node.textContent;
            let match;
            while ((match = regex.exec(text)) !== null) {
                this.searchResults.push({
                    node: node,
                    start: match.index,
                    end: match.index + match[0].length,
                    text: match[0]
                });
            }
        });

        this.highlightSearchResults();
        this.currentSearchIndex = 0;
        this.updateSearchResults(this.searchResults.length, this.currentSearchIndex + 1);

        if (this.searchResults.length > 0) {
            this.scrollToSearchResult(0);
        }
    }

    getTextNodes(element) {
        const textNodes = [];
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: (node) => {
                    // Skip script and style elements
                    const parent = node.parentElement;
                    if (parent && (parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE')) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    // Only include nodes with actual text content
                    return node.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                }
            }
        );

        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }

        return textNodes;
    }

    highlightSearchResults() {
        this.searchResults.forEach((result, index) => {
            const span = document.createElement('span');
            span.className = `highlight ${index === this.currentSearchIndex ? 'current' : ''}`;
            span.setAttribute('data-search-index', index);

            const parent = result.node.parentNode;
            const before = result.node.textContent.substring(0, result.start);
            const highlighted = result.node.textContent.substring(result.start, result.end);
            const after = result.node.textContent.substring(result.end);

            // Replace the text node with highlighted version
            if (before) {
                parent.insertBefore(document.createTextNode(before), result.node);
            }

            span.textContent = highlighted;
            parent.insertBefore(span, result.node);

            if (after) {
                parent.insertBefore(document.createTextNode(after), result.node);
            }

            parent.removeChild(result.node);

            // Update the result reference to the new span
            result.element = span;
        });
    }

    clearSearchHighlights() {
        const highlights = document.querySelectorAll('.highlight');
        highlights.forEach(highlight => {
            const parent = highlight.parentNode;
            parent.replaceChild(document.createTextNode(highlight.textContent), highlight);
            parent.normalize(); // Merge adjacent text nodes
        });
        this.searchResults = [];
    }

    navigateSearch(direction) {
        if (this.searchResults.length === 0) return;

        // Update current highlight
        if (this.searchResults[this.currentSearchIndex] && this.searchResults[this.currentSearchIndex].element) {
            this.searchResults[this.currentSearchIndex].element.classList.remove('current');
        }

        this.currentSearchIndex += direction;

        // Wrap around
        if (this.currentSearchIndex >= this.searchResults.length) {
            this.currentSearchIndex = 0;
        } else if (this.currentSearchIndex < 0) {
            this.currentSearchIndex = this.searchResults.length - 1;
        }

        // Highlight current result
        if (this.searchResults[this.currentSearchIndex] && this.searchResults[this.currentSearchIndex].element) {
            this.searchResults[this.currentSearchIndex].element.classList.add('current');
        }

        this.updateSearchResults(this.searchResults.length, this.currentSearchIndex + 1);
        this.scrollToSearchResult(this.currentSearchIndex);
    }

    scrollToSearchResult(index) {
        if (index >= 0 && index < this.searchResults.length && this.searchResults[index].element) {
            this.searchResults[index].element.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    }

    updateSearchResults(total, current) {
        const resultsContainer = document.querySelector('.search-results');
        if (resultsContainer) {
            if (total === 0) {
                resultsContainer.textContent = 'No results found';
            } else {
                resultsContainer.textContent = `${current} of ${total} results`;
            }
        }
    }

    clearSearch() {
        if (this.searchInput) {
            this.searchInput.value = '';
        }
        this.clearSearchHighlights();
        this.updateSearchResults(0, 0);
    }

    focusSearch() {
        if (this.searchInput) {
            this.searchInput.focus();
            this.searchInput.select();
        }
    }

    // Japanese text interactions
    initializeJapaneseInteractions() {
        this.setupJapaneseTooltips();
        this.setupJapaneseClickHandlers();
    }

    setupJapaneseTooltips() {
        const japaneseElements = document.querySelectorAll('.japanese-text');

        japaneseElements.forEach(element => {
            const romajiElement = element.parentNode.querySelector('.romaji');
            const translationElement = element.parentNode.querySelector('.translation');

            if (romajiElement || translationElement) {
                let tooltipContent = '';
                if (romajiElement) {
                    tooltipContent += `<div class="tooltip-romaji">${romajiElement.textContent}</div>`;
                }
                if (translationElement) {
                    tooltipContent += `<div class="tooltip-translation">${translationElement.textContent}</div>`;
                }

                this.createTooltip(element, tooltipContent);
            }
        });
    }

    createTooltip(element, content) {
        const tooltip = document.createElement('div');
        tooltip.className = 'japanese-tooltip';
        tooltip.innerHTML = content;
        document.body.appendChild(tooltip);

        element.addEventListener('mouseenter', (e) => {
            this.showTooltip(tooltip, e.target);
        });

        element.addEventListener('mouseleave', () => {
            this.hideTooltip(tooltip);
        });

        this.japaneseTooltips.set(element, tooltip);
    }

    showTooltip(tooltip, target) {
        const rect = target.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width / 2}px`;
        tooltip.style.top = `${rect.top - 10}px`;
        tooltip.style.opacity = '1';
        tooltip.style.visibility = 'visible';
        tooltip.style.transform = 'translateX(-50%) translateY(-100%)';
    }

    hideTooltip(tooltip) {
        tooltip.style.opacity = '0';
        tooltip.style.visibility = 'hidden';
    }

    setupJapaneseClickHandlers() {
        const japaneseElements = document.querySelectorAll('.japanese-text');

        japaneseElements.forEach(element => {
            element.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleJapaneseInfo(element);
            });
        });
    }

    toggleJapaneseInfo(element) {
        const segment = element.closest('.japanese-segment');
        if (segment) {
            segment.classList.toggle('expanded');

            // Toggle visibility of romaji and translation
            const romaji = segment.querySelector('.romaji');
            const translation = segment.querySelector('.translation');

            if (romaji) romaji.style.display = romaji.style.display === 'none' ? 'block' : 'none';
            if (translation) translation.style.display = translation.style.display === 'none' ? 'block' : 'none';
        }
    }

    // Image interactions
    setupImageInteractions() {
        this.setupImageZoom();
        this.setupImageModal();
    }

    setupImageZoom() {
        const images = document.querySelectorAll('.embedded-image, .extracted-image');

        images.forEach(img => {
            img.addEventListener('click', (e) => {
                e.preventDefault();
                this.openImageModal(img.src, img.alt || 'Document Image');
            });

            // Add zoom cursor
            img.style.cursor = 'zoom-in';
        });
    }

    setupImageModal() {
        // Create modal if it doesn't exist
        if (!document.querySelector('.image-modal')) {
            this.createImageModal();
        }
    }

    createImageModal() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay image-modal';
        modal.innerHTML = `
            <div class="modal image-modal-content">
                <div class="modal-header">
                    <h3 class="modal-title">Document Image</h3>
                    <button class="modal-close" type="button">×</button>
                </div>
                <div class="modal-body">
                    <img class="modal-image" src="" alt="" />
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary modal-close" type="button">Close</button>
                    <button class="btn btn-primary download-image" type="button">Download</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Bind modal events
        modal.addEventListener('click', (e) => {
            if (e.target === modal || e.target.classList.contains('modal-close')) {
                this.closeImageModal();
            }
        });

        const downloadBtn = modal.querySelector('.download-image');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadCurrentImage();
            });
        }

        // Keyboard support
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('active')) {
                this.closeImageModal();
            }
        });
    }

    openImageModal(src, title) {
        const modal = document.querySelector('.image-modal');
        const modalImage = modal.querySelector('.modal-image');
        const modalTitle = modal.querySelector('.modal-title');

        if (modalImage) modalImage.src = src;
        if (modalTitle) modalTitle.textContent = title;

        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    closeImageModal() {
        const modal = document.querySelector('.image-modal');
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }

    downloadCurrentImage() {
        const modal = document.querySelector('.image-modal');
        const img = modal.querySelector('.modal-image');

        if (img && img.src) {
            const a = document.createElement('a');
            a.href = img.src;
            a.download = 'document-image.jpg';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    }

    // Utility functions
    handleResize() {
        // Adjust layout for responsive design
        if (window.innerWidth <= 768 && this.sidebarOpen) {
            this.sidebarOpen = false;
            if (this.sidebar) {
                this.sidebar.classList.add('collapsed');
            }
        }
    }

    printDocument() {
        window.print();
    }

    saveViewerState() {
        const state = {
            zoom: this.currentZoom,
            viewMode: this.viewMode,
            sidebarOpen: this.sidebarOpen
        };

        try {
            localStorage.setItem('viewer-state', JSON.stringify(state));
        } catch (e) {
            console.warn('Could not save viewer state:', e);
        }
    }

    loadViewerState() {
        try {
            const saved = localStorage.getItem('viewer-state');
            if (saved) {
                const state = JSON.parse(saved);

                if (state.zoom) this.setZoom(state.zoom);
                if (state.viewMode) this.setViewMode(state.viewMode);
                if (typeof state.sidebarOpen === 'boolean') {
                    this.sidebarOpen = state.sidebarOpen;
                    if (this.sidebar) {
                        this.sidebar.classList.toggle('collapsed', !this.sidebarOpen);
                    }
                }
            }
        } catch (e) {
            console.warn('Could not load viewer state:', e);
        }
    }

    // Public API methods
    getCurrentZoom() {
        return this.currentZoom;
    }

    getCurrentViewMode() {
        return this.viewMode;
    }

    isSidebarOpen() {
        return this.sidebarOpen;
    }

    destroy() {
        // Cleanup tooltips
        this.japaneseTooltips.forEach(tooltip => {
            if (tooltip.parentNode) {
                tooltip.parentNode.removeChild(tooltip);
            }
        });
        this.japaneseTooltips.clear();

        // Clear search
        this.clearSearchHighlights();

        // Remove event listeners
        document.removeEventListener('keydown', this.keyboardHandler);
        window.removeEventListener('resize', this.resizeHandler);
    }
}

// Initialize viewer when DOM is ready
let documentViewer;

document.addEventListener('DOMContentLoaded', function() {
    // Only initialize if we're on a viewer page
    if (document.querySelector('.document-viewer')) {
        documentViewer = new DocumentViewer();
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DocumentViewer };
}