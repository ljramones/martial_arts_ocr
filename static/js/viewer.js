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
        this.searchTimeout = null;

        // New: tracked from backend response
        this.japaneseDetected = false;

        this.init();
    }

    init() {
        this.setupElements();
        this.bindEvents();
        this.setupKeyboardShortcuts();
        this.ensureJapaneseBadge();           // NEW: create/update the "Japanese: Yes/No" pill
        this.initializeJapaneseInteractions(); // will no-op until japaneseDetected = true
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

    // NEW: ensure the "Japanese: Yes/No" badge exists and is synced
    ensureJapaneseBadge() {
        if (!this.toolbar) return;
        let badge = this.toolbar.querySelector('.japanese-badge');
        if (!badge) {
            badge = document.createElement('span');
            badge.className = 'japanese-badge pill';
            // minimal styling fallback if CSS missing
            badge.style.marginLeft = 'auto';
            badge.style.padding = '4px 8px';
            badge.style.borderRadius = '999px';
            badge.style.fontSize = '12px';
            badge.style.lineHeight = '1';
            badge.style.background = '#eef1f5';
            badge.style.color = '#333';
            // try to insert at right end of toolbar
            this.toolbar.appendChild(badge);
        }
        this.updateJapaneseBadge();
    }

    // NEW: update the badge text/appearance
    updateJapaneseBadge() {
        const badge = this.toolbar ? this.toolbar.querySelector('.japanese-badge') : null;
        if (!badge) return;
        const yes = !!this.japaneseDetected;
        badge.textContent = yes ? 'Japanese: Yes' : 'Japanese: No';
        badge.title = yes
            ? 'Japanese markup & tooltips enabled'
            : 'No kana/kanji detected — Japanese helpers hidden';
        // color hint
        badge.style.background = yes ? '#e6f6ea' : '#f3f4f6';
        badge.style.color = yes ? '#0f5132' : '#333';
        badge.style.border = yes ? '1px solid #b7e2c4' : '1px solid #dcdfe4';
    }

    // NEW: public method to set the flag at runtime (from page JS)
    setJapaneseDetected(flag) {
        this.japaneseDetected = !!flag;
        this.updateJapaneseBadge();

        // If false: hide any existing JP markup and remove tooltips
        const jpBlocks = document.querySelectorAll('.japanese-container, .japanese-segment, .japanese-text');
        jpBlocks.forEach(el => {
            // collapse helpers but do not destroy content
            if (!this.japaneseDetected) {
                el.classList.remove('expanded');
                if (el.classList.contains('japanese-segment')) {
                    const romaji = el.querySelector('.romaji');
                    const translation = el.querySelector('.translation');
                    if (romaji) romaji.style.display = 'none';
                    if (translation) translation.style.display = 'none';
                }
            }
        });

        // Clear any existing tooltips if turning OFF
        if (!this.japaneseDetected) {
            this.japaneseTooltips.forEach(tooltip => tooltip?.parentNode?.removeChild(tooltip));
            this.japaneseTooltips.clear();
        } else {
            // Turning ON: initialize if not already wired by external manager
            this.initializeJapaneseInteractions();
        }
    }

    bindEvents() {
        // Zoom controls
        if (this.zoomInBtn) this.zoomInBtn.addEventListener('click', () => this.zoomIn());
        if (this.zoomOutBtn) this.zoomOutBtn.addEventListener('click', () => this.zoomOut());
        if (this.zoomResetBtn) this.zoomResetBtn.addEventListener('click', () => this.resetZoom());

        // View mode controls
        if (this.originalViewBtn) this.originalViewBtn.addEventListener('click', () => this.setViewMode('original'));
        if (this.textOnlyBtn) this.textOnlyBtn.addEventListener('click', () => this.setViewMode('text-only'));
        if (this.sideBySideBtn) this.sideBySideBtn.addEventListener('click', () => this.setViewMode('side-by-side'));

        // Other controls
        if (this.fullscreenBtn) this.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        if (this.sidebarToggle) this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        if (this.printBtn) this.printBtn.addEventListener('click', () => this.printDocument());

        // Search functionality
        if (this.searchInput) {
            this.searchInput.addEventListener('input', (e) => this.debounceSearch(e.target.value));
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
                    (e.deltaY < 0) ? this.zoomIn() : this.zoomOut();
                }
            }, {passive: false});
        }

        // Window resize
        this.resizeHandler = () => this.handleResize();
        window.addEventListener('resize', this.resizeHandler);

        // Fullscreen change
        document.addEventListener('fullscreenchange', () => this.handleFullscreenChange());
    }

    setupKeyboardShortcuts() {
        this.keyboardHandler = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

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
                    if (this.isFullscreen) this.exitFullscreen();
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
        };
        document.addEventListener('keydown', this.keyboardHandler);
    }

    // Zoom functionality
    zoomIn() {
        this.setZoom(Math.min(this.maxZoom, this.currentZoom + this.zoomStep));
    }

    zoomOut() {
        this.setZoom(Math.max(this.minZoom, this.currentZoom - this.zoomStep));
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

        if (this.zoomLevel) this.zoomLevel.textContent = `${Math.round(zoom * 100)}%`;

        this.updateZoomButtons();
        this.saveViewerState();
    }

    updateZoomButtons() {
        if (this.zoomInBtn) this.zoomInBtn.disabled = this.currentZoom >= this.maxZoom;
        if (this.zoomOutBtn) this.zoomOutBtn.disabled = this.currentZoom <= this.minZoom;
    }

    // View mode functionality
    setViewMode(mode) {
        this.viewMode = mode;
        if (this.contentContainer) this.contentContainer.className = `content-container ${mode}-view`;
        this.updateViewModeButtons();
        this.saveViewerState();
    }

    updateViewModeButtons() {
        [this.originalViewBtn, this.textOnlyBtn, this.sideBySideBtn].forEach(btn => btn && btn.classList.remove('active'));
        if (this.viewMode === 'original' && this.originalViewBtn) this.originalViewBtn.classList.add('active');
        if (this.viewMode === 'text-only' && this.textOnlyBtn) this.textOnlyBtn.classList.add('active');
        if (this.viewMode === 'side-by-side' && this.sideBySideBtn) this.sideBySideBtn.classList.add('active');
    }

    // Fullscreen functionality
    toggleFullscreen() {
        this.isFullscreen ? this.exitFullscreen() : this.enterFullscreen();
    }

    enterFullscreen() {
        if (this.viewer?.requestFullscreen) this.viewer.requestFullscreen();
    }

    exitFullscreen() {
        if (document.exitFullscreen) document.exitFullscreen();
    }

    handleFullscreenChange() {
        this.isFullscreen = !!document.fullscreenElement;
        if (this.viewer) this.viewer.classList.toggle('viewer-fullscreen', this.isFullscreen);
        if (this.fullscreenBtn) {
            this.fullscreenBtn.textContent = this.isFullscreen ? 'Exit Fullscreen' : 'Fullscreen';
            this.fullscreenBtn.title = this.isFullscreen ? 'Exit fullscreen (F11 or Esc)' : 'Enter fullscreen (F11)';
        }
    }

    // Sidebar
    toggleSidebar() {
        this.sidebarOpen = !this.sidebarOpen;
        if (this.sidebar) this.sidebar.classList.toggle('collapsed', !this.sidebarOpen);
        if (this.sidebarToggle) this.sidebarToggle.textContent = this.sidebarOpen ? 'Hide Sidebar' : 'Show Sidebar';
        this.saveViewerState();
    }

    // Search
    debounceSearch(query) {
        clearTimeout(this.searchTimeout);
        this.searchTimeout = setTimeout(() => this.performSearch(query), 300);
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
                this.searchResults.push({node, start: match.index, end: match.index + match[0].length, text: match[0]});
            }
        });

        this.highlightSearchResults();
        this.currentSearchIndex = 0;
        this.updateSearchResults(this.searchResults.length, this.currentSearchIndex + 1);
        if (this.searchResults.length > 0) this.scrollToSearchResult(0);
    }

    getTextNodes(element) {
        const textNodes = [];
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: (node) => {
                    const p = node.parentElement;
                    if (!p) return NodeFilter.FILTER_REJECT;
                    if (p.closest('.japanese-container')) return NodeFilter.FILTER_REJECT; // don't break JP markup
                    if (p.tagName === 'SCRIPT' || p.tagName === 'STYLE') return NodeFilter.FILTER_REJECT;
                    return node.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                }
            }
        );
        let n;
        while ((n = walker.nextNode())) textNodes.push(n);
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

            if (before) parent.insertBefore(document.createTextNode(before), result.node);
            span.textContent = highlighted;
            parent.insertBefore(span, result.node);
            if (after) parent.insertBefore(document.createTextNode(after), result.node);

            parent.removeChild(result.node);
            result.element = span;
        });
    }

    clearSearchHighlights() {
        const highlights = document.querySelectorAll('.highlight');
        highlights.forEach(h => {
            const parent = h.parentNode;
            if (!parent) return;
            parent.replaceChild(document.createTextNode(h.textContent), h);
            parent.normalize();
        });
        this.searchResults = [];
    }

    navigateSearch(direction) {
        if (this.searchResults.length === 0) return;

        const current = this.searchResults[this.currentSearchIndex];
        if (current?.element) current.element.classList.remove('current');

        this.currentSearchIndex += direction;
        if (this.currentSearchIndex >= this.searchResults.length) this.currentSearchIndex = 0;
        else if (this.currentSearchIndex < 0) this.currentSearchIndex = this.searchResults.length - 1;

        const next = this.searchResults[this.currentSearchIndex];
        if (next?.element) next.element.classList.add('current');

        this.updateSearchResults(this.searchResults.length, this.currentSearchIndex + 1);
        this.scrollToSearchResult(this.currentSearchIndex);
    }

    scrollToSearchResult(index) {
        const el = (index >= 0 && index < this.searchResults.length) ? this.searchResults[index].element : null;
        if (el) el.scrollIntoView({behavior: 'smooth', block: 'center'});
    }

    updateSearchResults(total, current) {
        const resultsContainer = document.querySelector('.search-results');
        if (resultsContainer) {
            resultsContainer.textContent = total === 0 ? 'No results found' : `${current} of ${total} results`;
        }
    }

    clearSearch() {
        if (this.searchInput) this.searchInput.value = '';
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
        // Avoid duplicate tooltip logic if japanese.js is managing interactions
        if (typeof window !== 'undefined' && window.japaneseTextManager) return;

        // NEW: only enable if the backend flagged it
        if (!this.japaneseDetected) return;

        this.setupJapaneseTooltips();
        this.setupJapaneseClickHandlers();
    }

    setupJapaneseTooltips() {
        const japaneseElements = document.querySelectorAll('.japanese-text');

        japaneseElements.forEach(element => {
            const segment = element.closest('.japanese-segment') || element.parentNode;
            if (!segment) return;

            const romajiElement = segment.querySelector('.romaji');
            const translationElement = segment.querySelector('.translation');

            if (!romajiElement && !translationElement) return;

            // Build tooltip content safely (no innerHTML)
            const tooltip = this.createTooltipElement();
            if (romajiElement) {
                const tr = document.createElement('div');
                tr.className = 'tooltip-romaji';
                tr.textContent = romajiElement.textContent || '';
                tooltip.appendChild(tr);
            }
            if (translationElement) {
                const tt = document.createElement('div');
                tt.className = 'tooltip-translation';
                tt.textContent = translationElement.textContent || '';
                tooltip.appendChild(tt);
            }
            document.body.appendChild(tooltip);

            element.addEventListener('mouseenter', (e) => this.showTooltip(tooltip, e.target));
            element.addEventListener('mouseleave', () => this.hideTooltip(tooltip));

            this.japaneseTooltips.set(element, tooltip);
        });
    }

    createTooltipElement() {
        const tooltip = document.createElement('div');
        tooltip.className = 'japanese-tooltip';
        tooltip.style.position = 'fixed';
        tooltip.style.opacity = '0';
        tooltip.style.visibility = 'hidden';
        tooltip.style.pointerEvents = 'none';
        return tooltip;
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
        if (!segment) return;
        segment.classList.toggle('expanded');

        const romaji = segment.querySelector('.romaji');
        const translation = segment.querySelector('.translation');
        if (romaji) romaji.style.display = (romaji.style.display === 'none') ? 'block' : 'none';
        if (translation) translation.style.display = (translation.style.display === 'none') ? 'block' : 'none';
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
            img.style.cursor = 'zoom-in';
        });
    }

    setupImageModal() {
        if (!document.querySelector('.image-modal')) {
            this.createImageModal();
        }
    }

    createImageModal() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay image-modal';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.innerHTML = `
            <div class="modal image-modal-content">
                <div class="modal-header">
                    <h3 class="modal-title">Document Image</h3>
                    <button class="modal-close" type="button" aria-label="Close">×</button>
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

        // Basic focus handling
        const closeEls = modal.querySelectorAll('.modal-close');
        closeEls.forEach(btn => btn.addEventListener('click', () => this.closeImageModal()));
        modal.addEventListener('click', (e) => {
            if (e.target === modal) this.closeImageModal();
        });

        const downloadBtn = modal.querySelector('.download-image');
        if (downloadBtn) downloadBtn.addEventListener('click', () => this.downloadCurrentImage());

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('active')) this.closeImageModal();
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

        // Move focus to close button for accessibility
        const focusTarget = modal.querySelector('.modal-close');
        if (focusTarget) focusTarget.focus();
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

    // Utility
    handleResize() {
        if (window.innerWidth <= 768 && this.sidebarOpen) {
            this.sidebarOpen = false;
            if (this.sidebar) this.sidebar.classList.add('collapsed');
        }
    }

    printDocument() {
        window.print();
    }

    saveViewerState() {
        const state = {zoom: this.currentZoom, viewMode: this.viewMode, sidebarOpen: this.sidebarOpen};
        try {
            localStorage.setItem('viewer-state', JSON.stringify(state));
        } catch (e) {
            console.warn('Could not save viewer state:', e);
        }
    }

    loadViewerState() {
        try {
            const saved = localStorage.getItem('viewer-state');
            if (!saved) return;
            const state = JSON.parse(saved);
            if (typeof state.zoom === 'number') this.setZoom(state.zoom);
            if (state.viewMode) this.setViewMode(state.viewMode);
            if (typeof state.sidebarOpen === 'boolean') {
                this.sidebarOpen = state.sidebarOpen;
                if (this.sidebar) this.sidebar.classList.toggle('collapsed', !this.sidebarOpen);
            }
        } catch (e) {
            console.warn('Could not load viewer state:', e);
        }
    }

    // Public API (fixed: use this.* not self.*)
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
        this.japaneseTooltips.forEach(tooltip => tooltip?.parentNode?.removeChild(tooltip));
        this.japaneseTooltips.clear();

        // Clear search
        this.clearSearchHighlights();

        // Remove listeners
        document.removeEventListener('keydown', this.keyboardHandler);
        window.removeEventListener('resize', this.resizeHandler);
    }
}

// Initialize viewer when DOM is ready
let documentViewer;
document.addEventListener('DOMContentLoaded', function () {
    if (document.querySelector('.document-viewer')) {
        documentViewer = new DocumentViewer();
    }
    // OPTIONAL: mirror backend flag to DOM as data-attribute
    try {
        const vc = document.querySelector('.viewer-content');
        if (vc && typeof window.__processingMeta?.has_japanese !== 'undefined') {
            vc.dataset.japaneseDetected = String(!!window.__processingMeta.has_japanese);
        }
    } catch (e) { /* no-op */
    }
});

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {DocumentViewer};
}
