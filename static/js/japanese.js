/**
 * Japanese text processing and interaction functionality for Martial Arts OCR
 * Handles Japanese character detection, romanization display, translations, and user interactions
 */

class JapaneseTextManager {
    constructor() {
        this.tooltips = new Map();
        this.romajiCache = new Map();
        this.translationCache = new Map();
        this.isEnabled = true;  // will be gated by data-japanese-detected
        this.showRomaji = true;
        this.showTranslations = true;
        this.autoDetect = true;
        this.currentLanguage = 'en'; // Target translation language

        // Japanese character ranges
        this.japaneseRanges = {
            hiragana: [0x3040, 0x309F],
            katakana: [0x30A0, 0x30FF],
            kanji: [0x4E00, 0x9FFF],
            halfWidthKatakana: [0xFF65, 0xFF9F],
            kanaExtension: [0x1B000, 0x1B0FF]
        };

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.processExistingJapaneseText();
        this.createControlPanel();
        this.loadUserPreferences();
        this.setupKeyboardShortcuts();
    }

    setupEventListeners() {
        // Listen for dynamically added Japanese text
        this.setupMutationObserver();

        // Handle text selection for translation
        document.addEventListener('mouseup', (e) => {
            this.handleTextSelection(e);
        });

        // Handle Japanese text clicks
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('japanese-text')) {
                this.handleJapaneseClick(e);
            }
        });

        // Handle hover events for tooltips
        document.addEventListener('mouseover', (e) => {
            if (this.isJapaneseElement(e.target)) {
                this.showTooltip(e.target, e);
            }
        });

        document.addEventListener('mouseout', (e) => {
            if (this.isJapaneseElement(e.target)) {
                this.hideTooltip(e.target);
            }
        });
    }

    setupMutationObserver() {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            this.processJapaneseInElement(node);
                        }
                    });
                }
            });
        });

        observer.observe(document.body, {
            childList: true, subtree: true
        });
    }

    processExistingJapaneseText() {
        // Process all existing Japanese text in the document
        this.processJapaneseInElement(document.body);
    }

    processJapaneseInElement(element) {
        if (!this.isEnabled || !element) return;

        // Find existing Japanese text elements
        const japaneseElements = element.querySelectorAll('.japanese-text, .japanese-segment');
        japaneseElements.forEach(el => this.enhanceJapaneseElement(el));

        // Auto-detect Japanese text in regular content
        if (this.autoDetect) {
            this.autoDetectJapaneseText(element);
        }
    }

    autoDetectJapaneseText(element) {
        const textNodes = this.getTextNodes(element);

        textNodes.forEach(node => {
            const text = node.textContent;
            const japaneseSegments = this.extractJapaneseSegments(text);

            if (japaneseSegments.length > 0) {
                this.wrapJapaneseSegments(node, japaneseSegments);
            }
        });
    }

    getTextNodes(element) {
        const textNodes = [];
        const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
            acceptNode: (node) => {
                // Skip already processed nodes and script/style elements
                const parent = node.parentElement;
                if (!parent) return NodeFilter.FILTER_REJECT;

                if (parent.classList.contains('japanese-text') || parent.classList.contains('romaji') || parent.classList.contains('translation') || parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE') {
                    return NodeFilter.FILTER_REJECT;
                }

                return node.textContent.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
            }
        });

        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }

        return textNodes;
    }

    extractJapaneseSegments(text) {
        const segments = [];
        const japaneseRegex = /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF65-\uFF9F]+/g;
        let match;

        while ((match = japaneseRegex.exec(text)) !== null) {
            segments.push({
                text: match[0], start: match.index, end: match.index + match[0].length
            });
        }

        return segments;
    }

    wrapJapaneseSegments(textNode, segments) {
        if (segments.length === 0) return;

        const parent = textNode.parentNode;
        const text = textNode.textContent;
        let lastIndex = 0;

        segments.forEach(segment => {
            // Add text before Japanese segment
            if (segment.start > lastIndex) {
                const beforeText = text.substring(lastIndex, segment.start);
                if (beforeText) {
                    parent.insertBefore(document.createTextNode(beforeText), textNode);
                }
            }

            // Create Japanese element
            const japaneseElement = this.createJapaneseElement(segment.text);
            parent.insertBefore(japaneseElement, textNode);

            lastIndex = segment.end;
        });

        // Add remaining text
        if (lastIndex < text.length) {
            const remainingText = text.substring(lastIndex);
            if (remainingText) {
                parent.insertBefore(document.createTextNode(remainingText), textNode);
            }
        }

        // Remove original text node
        parent.removeChild(textNode);
    }

    createJapaneseElement(japaneseText) {
        const container = document.createElement('span');
        container.className = 'japanese-container';

        const textElement = document.createElement('span');
        textElement.className = 'japanese-text';
        textElement.textContent = japaneseText;
        textElement.setAttribute('data-japanese', japaneseText);

        container.appendChild(textElement);

        // Add loading indicator
        this.addLoadingIndicator(container);

        // Process asynchronously
        this.processJapaneseTextAsync(japaneseText, container);

        return container;
    }

    async processJapaneseTextAsync(text, container) {
        try {
            // Get romanization and translation
            const [romaji, translation] = await Promise.all([this.getRomanization(text), this.getTranslation(text)]);

            this.removeLoadingIndicator(container);

            if (romaji && this.showRomaji) {
                this.addRomajiElement(container, romaji);
            }

            if (translation && this.showTranslations) {
                this.addTranslationElement(container, translation);
            }

            // Add confidence indicator if available
            this.addConfidenceIndicator(container, text);

            // Setup interactions
            this.setupElementInteractions(container);

        } catch (error) {
            console.warn('Failed to process Japanese text:', text, error);
            this.removeLoadingIndicator(container);
            this.addErrorIndicator(container);
        }
    }

    async getRomanization(text) {
        // Check cache first
        if (this.romajiCache.has(text)) {
            return this.romajiCache.get(text);
        }

        try {
            // Try client-side romanization first (if library available)
            const clientRomaji = this.clientSideRomanization(text);
            if (clientRomaji) {
                this.romajiCache.set(text, clientRomaji);
                return clientRomaji;
            }

            // Fallback to server-side romanization
            const response = await fetch('/api/romanize', {
                method: 'POST', headers: {
                    'Content-Type': 'application/json',
                }, body: JSON.stringify({text: text})
            });

            if (response.ok) {
                const data = await response.json();
                const romaji = data.romaji;
                this.romajiCache.set(text, romaji);
                return romaji;
            }
        } catch (error) {
            console.warn('Romanization failed:', error);
        }

        return null;
    }

    async getTranslation(text) {
        // Check cache first
        if (this.translationCache.has(text)) {
            return this.translationCache.get(text);
        }

        try {
            const response = await fetch('/api/translate', {
                method: 'POST', headers: {
                    'Content-Type': 'application/json',
                }, body: JSON.stringify({
                    text: text, target_language: this.currentLanguage
                })
            });

            if (response.ok) {
                const data = await response.json();
                const translation = data.translation;
                this.translationCache.set(text, translation);
                return translation;
            }
        } catch (error) {
            console.warn('Translation failed:', error);
        }

        return null;
    }

    clientSideRomanization(text) {
        // Simple client-side romanization for common characters
        // This is a fallback - server-side processing is preferred
        const romajiMap = {
            'あ': 'a',
            'い': 'i',
            'う': 'u',
            'え': 'e',
            'お': 'o',
            'か': 'ka',
            'き': 'ki',
            'く': 'ku',
            'け': 'ke',
            'こ': 'ko',
            'が': 'ga',
            'ぎ': 'gi',
            'ぐ': 'gu',
            'げ': 'ge',
            'ご': 'go',
            'さ': 'sa',
            'し': 'shi',
            'す': 'su',
            'せ': 'se',
            'そ': 'so',
            'ざ': 'za',
            'じ': 'ji',
            'ず': 'zu',
            'ぜ': 'ze',
            'ぞ': 'zo',
            'た': 'ta',
            'ち': 'chi',
            'つ': 'tsu',
            'て': 'te',
            'と': 'to',
            'だ': 'da',
            'ぢ': 'ji',
            'づ': 'zu',
            'で': 'de',
            'ど': 'do',
            'な': 'na',
            'に': 'ni',
            'ぬ': 'nu',
            'ね': 'ne',
            'の': 'no',
            'は': 'ha',
            'ひ': 'hi',
            'ふ': 'fu',
            'へ': 'he',
            'ほ': 'ho',
            'ば': 'ba',
            'び': 'bi',
            'ぶ': 'bu',
            'べ': 'be',
            'ぼ': 'bo',
            'ぱ': 'pa',
            'ぴ': 'pi',
            'ぷ': 'pu',
            'ぺ': 'pe',
            'ぽ': 'po',
            'ま': 'ma',
            'み': 'mi',
            'む': 'mu',
            'め': 'me',
            'も': 'mo',
            'や': 'ya',
            'ゆ': 'yu',
            'よ': 'yo',
            'ら': 'ra',
            'り': 'ri',
            'る': 'ru',
            'れ': 're',
            'ろ': 'ro',
            'わ': 'wa',
            'を': 'wo',
            'ん': 'n', // Common kanji
            '武': 'bu',
            '道': 'dō',
            '空': 'kara',
            '手': 'te',
            '柔': 'jū',
            '術': 'jutsu',
            '剣': 'ken',
            '型': 'kata'
        };

        let result = '';
        for (let char of text) {
            if (romajiMap[char]) {
                result += romajiMap[char];
            } else if (this.isJapaneseCharacter(char)) {
                result += char; // Keep unknown Japanese characters as-is
            } else {
                result += char; // Keep non-Japanese characters
            }
        }

        return result || null;
    }

    isJapaneseCharacter(char) {
        const code = char.charCodeAt(0);
        return Object.values(this.japaneseRanges).some(([start, end]) => {
            return code >= start && code <= end;
        });
    }

    isJapaneseElement(element) {
        return element.classList.contains('japanese-text') || element.classList.contains('japanese-container') || element.closest('.japanese-segment');
    }

    enhanceJapaneseElement(element) {
        if (!element.dataset.enhanced) {
            this.setupElementInteractions(element);
            element.dataset.enhanced = 'true';
        }
    }

    setupElementInteractions(container) {
        const japaneseText = container.querySelector('.japanese-text');
        if (!japaneseText) return;

        // Click to toggle details
        japaneseText.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleJapaneseDetails(container);
        });

        // Hover for tooltip
        japaneseText.addEventListener('mouseenter', (e) => {
            this.showTooltip(container, e);
        });

        japaneseText.addEventListener('mouseleave', () => {
            this.hideTooltip(container);
        });

        // Copy functionality
        japaneseText.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.showContextMenu(container, e);
        });
    }

    addRomajiElement(container, romaji) {
        const romajiElement = document.createElement('span');
        romajiElement.className = 'romaji';
        romajiElement.textContent = romaji;
        romajiElement.title = 'Romanization';

        if (!this.showRomaji) {
            romajiElement.style.display = 'none';
        }

        container.appendChild(romajiElement);
    }

    addTranslationElement(container, translation) {
        const translationElement = document.createElement('span');
        translationElement.className = 'translation';
        translationElement.textContent = translation;
        translationElement.title = 'Translation';

        if (!this.showTranslations) {
            translationElement.style.display = 'none';
        }

        container.appendChild(translationElement);
    }

    addLoadingIndicator(container) {
        const loading = document.createElement('span');
        loading.className = 'japanese-loading';
        loading.innerHTML = '<span class="loading-dots">...</span>';
        loading.title = 'Processing Japanese text...';
        container.appendChild(loading);
    }

    removeLoadingIndicator(container) {
        const loading = container.querySelector('.japanese-loading');
        if (loading) {
            loading.remove();
        }
    }

    addErrorIndicator(container) {
        const error = document.createElement('span');
        error.className = 'japanese-error';
        error.innerHTML = '⚠️';
        error.title = 'Failed to process Japanese text';
        container.appendChild(error);
    }

    addConfidenceIndicator(container, text) {
        // Add confidence based on character complexity
        const confidence = this.calculateConfidence(text);
        const indicator = document.createElement('span');
        indicator.className = 'confidence-indicator';
        indicator.textContent = `${Math.round(confidence * 100)}%`;
        indicator.title = `Processing confidence: ${Math.round(confidence * 100)}%`;

        if (confidence < 0.7) {
            indicator.classList.add('low-confidence');
        }

        container.appendChild(indicator);
    }

    calculateConfidence(text) {
        // Simple confidence calculation based on character types
        let score = 0.5; // Base score

        // Boost for common characters
        if (/[あいうえおかきくけこ]/.test(text)) score += 0.2;
        if (/[武道空手柔術剣型]/.test(text)) score += 0.3; // Martial arts terms

        // Penalty for mixed scripts that might indicate OCR errors
        const hasHiragana = /[\u3040-\u309F]/.test(text);
        const hasKatakana = /[\u30A0-\u30FF]/.test(text);
        const hasKanji = /[\u4E00-\u9FFF]/.test(text);

        if (hasHiragana + hasKatakana + hasKanji > 2) score -= 0.1;

        return Math.max(0, Math.min(1, score));
    }

    toggleJapaneseDetails(container) {
        container.classList.toggle('expanded');

        const romaji = container.querySelector('.romaji');
        const translation = container.querySelector('.translation');

        if (container.classList.contains('expanded')) {
            if (romaji) romaji.style.display = 'block';
            if (translation) translation.style.display = 'block';
        } else {
            if (romaji && !this.showRomaji) romaji.style.display = 'none';
            if (translation && !this.showTranslations) translation.style.display = 'none';
        }
    }

    showTooltip(container, event) {
        const japaneseText = container.querySelector('.japanese-text');
        const romaji = container.querySelector('.romaji');
        const translation = container.querySelector('.translation');

        if (!japaneseText || (!romaji && !translation)) return;

        const tooltip = this.createTooltip(japaneseText.textContent, romaji?.textContent, translation?.textContent);

        this.positionTooltip(tooltip, event);
        this.tooltips.set(container, tooltip);
    }

    hideTooltip(container) {
        const tooltip = this.tooltips.get(container);
        if (tooltip) {
            tooltip.remove();
            this.tooltips.delete(container);
        }
    }

    createTooltip(japanese, romaji, translation) {
        const tooltip = document.createElement('div');
        tooltip.className = 'japanese-tooltip';

        const tj = document.createElement('div');
        tj.className = 'tooltip-japanese';
        tj.textContent = japanese;
        tooltip.appendChild(tj);

        if (romaji) {
            const tr = document.createElement('div');
            tr.className = 'tooltip-romaji';
            tr.textContent = romaji;
            tooltip.appendChild(tr);
        }
        if (translation) {
            const tt = document.createElement('div');
            tt.className = 'tooltip-translation';
            tt.textContent = translation;
            tooltip.appendChild(tt);
        }

        document.body.appendChild(tooltip);
        return tooltip;
    }


    positionTooltip(tooltip, event) {
        const rect = event.target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();

        let left = rect.left + rect.width / 2 - tooltipRect.width / 2;
        let top = rect.top - tooltipRect.height - 10;

        // Adjust if tooltip goes off screen
        if (left < 0) left = 10;
        if (left + tooltipRect.width > window.innerWidth) {
            left = window.innerWidth - tooltipRect.width - 10;
        }
        if (top < 0) {
            top = rect.bottom + 10;
        }

        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
        tooltip.style.opacity = '1';
        tooltip.style.visibility = 'visible';
    }

    showContextMenu(container, event) {
        const japaneseText = container.querySelector('.japanese-text')?.textContent;
        const romaji = container.querySelector('.romaji')?.textContent;
        const translation = container.querySelector('.translation')?.textContent;

        const menu = document.createElement('div');
        menu.className = 'japanese-context-menu';
        menu.innerHTML = `
            <div class="context-menu-item" data-action="copy-japanese">Copy Japanese</div>
            ${romaji ? '<div class="context-menu-item" data-action="copy-romaji">Copy Romaji</div>' : ''}
            ${translation ? '<div class="context-menu-item" data-action="copy-translation">Copy Translation</div>' : ''}
            <div class="context-menu-item" data-action="copy-all">Copy All</div>
            <hr>
            <div class="context-menu-item" data-action="lookup">Lookup Online</div>
        `;

        menu.addEventListener('click', (e) => {
            const action = e.target.dataset.action;
            this.handleContextMenuAction(action, {japaneseText, romaji, translation});
            menu.remove();
        });

        document.addEventListener('click', () => menu.remove(), {once: true});

        menu.style.left = `${event.clientX}px`;
        menu.style.top = `${event.clientY}px`;
        document.body.appendChild(menu);
    }

    handleContextMenuAction(action, data) {
        switch (action) {
            case 'copy-japanese':
                this.copyToClipboard(data.japaneseText);
                break;
            case 'copy-romaji':
                this.copyToClipboard(data.romaji);
                break;
            case 'copy-translation':
                this.copyToClipboard(data.translation);
                break;
            case 'copy-all':
                const all = [data.japaneseText, data.romaji, data.translation]
                    .filter(Boolean).join(' - ');
                this.copyToClipboard(all);
                break;
            case 'lookup':
                this.lookupOnline(data.japaneseText);
                break;
        }
    }

    copyToClipboard(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                this.showNotification('Copied to clipboard');
            });
        } else {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            this.showNotification('Copied to clipboard');
        }
    }

    lookupOnline(text) {
        const url = `https://jisho.org/search/${encodeURIComponent(text)}`;
        window.open(url, '_blank');
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'japanese-notification';
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }, 2000);
    }

    handleTextSelection(event) {
        const selection = window.getSelection();
        const selectedText = selection.toString().trim();

        if (selectedText && this.containsJapanese(selectedText)) {
            this.showSelectionActions(selectedText, event);
        }
    }

    containsJapanese(text) {
        return /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]/.test(text);
    }

    showSelectionActions(text, event) {
        const existing = document.querySelector('.selection-actions');
        if (existing) existing.remove();

        const actions = document.createElement('div');
        actions.className = 'selection-actions';
        actions.innerHTML = `
            <button class="action-btn" data-action="romanize">Romanize</button>
            <button class="action-btn" data-action="translate">Translate</button>
            <button class="action-btn" data-action="lookup">Lookup</button>
        `;

        actions.addEventListener('click', (e) => {
            const action = e.target.dataset.action;
            this.handleSelectionAction(action, text);
            actions.remove();
        });

        document.body.appendChild(actions);
        this.positionSelectionActions(actions, event);

        // Auto-hide after 5 seconds
        setTimeout(() => actions.remove(), 5000);
    }

    positionSelectionActions(actions, event) {
        const rect = actions.getBoundingClientRect();
        let left = event.clientX - rect.width / 2;
        let top = event.clientY - rect.height - 10;

        if (left < 0) left = 10;
        if (left + rect.width > window.innerWidth) {
            left = window.innerWidth - rect.width - 10;
        }
        if (top < 0) top = event.clientY + 10;

        actions.style.left = `${left}px`;
        actions.style.top = `${top}px`;
    }

    async handleSelectionAction(action, text) {
        switch (action) {
            case 'romanize':
                const romaji = await this.getRomanization(text);
                if (romaji) {
                    this.showNotification(`Romaji: ${romaji}`);
                }
                break;
            case 'translate':
                const translation = await this.getTranslation(text);
                if (translation) {
                    this.showNotification(`Translation: ${translation}`);
                }
                break;
            case 'lookup':
                this.lookupOnline(text);
                break;
        }
    }

    handleJapaneseClick(event) {
        const container = event.target.closest('.japanese-container');
        if (container) {
            this.toggleJapaneseDetails(container);
        }
    }

    createControlPanel() {
        const panel = document.createElement('div');
        panel.className = 'japanese-control-panel';
        panel.innerHTML = `
            <div class="panel-header">
                <h4>Japanese Text Options</h4>
                <button class="panel-toggle">−</button>
            </div>
            <div class="panel-content">
                <label>
                    <input type="checkbox" id="show-romaji" ${this.showRomaji ? 'checked' : ''}>
                    Show Romanization
                </label>
                <label>
                    <input type="checkbox" id="show-translations" ${this.showTranslations ? 'checked' : ''}>
                    Show Translations
                </label>
                <label>
                    <input type="checkbox" id="auto-detect" ${this.autoDetect ? 'checked' : ''}>
                    Auto-detect Japanese Text
                </label>
                <div class="panel-actions">
                    <button id="refresh-japanese">Refresh All</button>
                    <button id="clear-cache">Clear Cache</button>
                </div>
            </div>
        `;

        this.bindControlPanelEvents(panel);
        document.body.appendChild(panel);
    }

    bindControlPanelEvents(panel) {
        const showRomajiCheckbox = panel.querySelector('#show-romaji');
        const showTranslationsCheckbox = panel.querySelector('#show-translations');
        const autoDetectCheckbox = panel.querySelector('#auto-detect');
        const refreshButton = panel.querySelector('#refresh-japanese');
        const clearCacheButton = panel.querySelector('#clear-cache');
        const toggleButton = panel.querySelector('.panel-toggle');

        showRomajiCheckbox.addEventListener('change', (e) => {
            this.showRomaji = e.target.checked;
            this.toggleRomajiDisplay(this.showRomaji);
            this.saveUserPreferences();
        });

        showTranslationsCheckbox.addEventListener('change', (e) => {
            this.showTranslations = e.target.checked;
            this.toggleTranslationDisplay(this.showTranslations);
            this.saveUserPreferences();
        });

        autoDetectCheckbox.addEventListener('change', (e) => {
            this.autoDetect = e.target.checked;
            this.saveUserPreferences();
        });

        refreshButton.addEventListener('click', () => {
            this.refreshAllJapaneseText();
        });

        clearCacheButton.addEventListener('click', () => {
            this.clearCache();
        });

        toggleButton.addEventListener('click', () => {
            panel.classList.toggle('collapsed');
            toggleButton.textContent = panel.classList.contains('collapsed') ? '+' : '−';
        });
    }

    toggleRomajiDisplay(show) {
        const romajiElements = document.querySelectorAll('.romaji');
        romajiElements.forEach(el => {
            el.style.display = show ? 'block' : 'none';
        });
    }

    toggleTranslationDisplay(show) {
        const translationElements = document.querySelectorAll('.translation');
        translationElements.forEach(el => {
            el.style.display = show ? 'block' : 'none';
        });
    }

    refreshAllJapaneseText() {
        // Remove all enhanced elements and reprocess
        const japaneseContainers = document.querySelectorAll('.japanese-container');
        japaneseContainers.forEach(container => {
            delete container.dataset.enhanced;
        });

        this.processExistingJapaneseText();
        this.showNotification('Japanese text refreshed');
    }

    clearCache() {
        this.romajiCache.clear();
        this.translationCache.clear();
        this.showNotification('Cache cleared');
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Only process if not in input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            switch (e.key) {
                case 'j':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.toggleJapaneseMode();
                    }
                    break;

                case 'r':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.showRomaji = !this.showRomaji;
                        this.toggleRomajiDisplay(this.showRomaji);
                        this.saveUserPreferences();
                    }
                    break;

                case 't':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.showTranslations = !this.showTranslations;
                        this.toggleTranslationDisplay(this.showTranslations);
                        this.saveUserPreferences();
                    }
                    break;
            }
        });
    }

    toggleJapaneseMode() {
        this.isEnabled = !this.isEnabled;
        const panel = document.querySelector('.japanese-control-panel');

        if (panel) {
            panel.style.display = this.isEnabled ? 'block' : 'none';
        }

        if (this.isEnabled) {
            this.processExistingJapaneseText();
            this.showNotification('Japanese mode enabled');
        } else {
            this.hideAllJapaneseElements();
            this.showNotification('Japanese mode disabled');
        }

        this.saveUserPreferences();
    }

    hideAllJapaneseElements() {
        const japaneseElements = document.querySelectorAll('.japanese-container, .romaji, .translation');
        japaneseElements.forEach(el => {
            el.style.display = 'none';
        });
    }

    saveUserPreferences() {
        const preferences = {
            isEnabled: this.isEnabled,
            showRomaji: this.showRomaji,
            showTranslations: this.showTranslations,
            autoDetect: this.autoDetect,
            currentLanguage: this.currentLanguage
        };

        try {
            localStorage.setItem('japanese-preferences', JSON.stringify(preferences));
        } catch (e) {
            console.warn('Could not save Japanese preferences:', e);
        }
    }

    loadUserPreferences() {
        try {
            const saved = localStorage.getItem('japanese-preferences');
            if (saved) {
                const preferences = JSON.parse(saved);

                this.isEnabled = preferences.isEnabled !== false; // Default to true
                this.showRomaji = preferences.showRomaji !== false;
                this.showTranslations = preferences.showTranslations !== false;
                this.autoDetect = preferences.autoDetect !== false;
                this.currentLanguage = preferences.currentLanguage || 'en';

                // Update control panel
                this.updateControlPanelFromPreferences();
            }
        } catch (e) {
            console.warn('Could not load Japanese preferences:', e);
        }
    }

    updateControlPanelFromPreferences() {
        const panel = document.querySelector('.japanese-control-panel');
        if (!panel) return;

        const showRomajiCheckbox = panel.querySelector('#show-romaji');
        const showTranslationsCheckbox = panel.querySelector('#show-translations');
        const autoDetectCheckbox = panel.querySelector('#auto-detect');

        if (showRomajiCheckbox) showRomajiCheckbox.checked = this.showRomaji;
        if (showTranslationsCheckbox) showTranslationsCheckbox.checked = this.showTranslations;
        if (autoDetectCheckbox) autoDetectCheckbox.checked = this.autoDetect;

        if (!this.isEnabled) {
            panel.style.display = 'none';
        }
    }

    // Public API methods
    enable() {
        this.isEnabled = true;
        this.processExistingJapaneseText();
        this.saveUserPreferences();
    }

    disable() {
        this.isEnabled = false;
        this.hideAllJapaneseElements();
        this.saveUserPreferences();
    }

    setRomajiDisplay(show) {
        this.showRomaji = show;
        this.toggleRomajiDisplay(show);
        this.saveUserPreferences();
    }

    setTranslationDisplay(show) {
        this.showTranslations = show;
        this.toggleTranslationDisplay(show);
        this.saveUserPreferences();
    }

    setTargetLanguage(language) {
        this.currentLanguage = language;
        this.translationCache.clear(); // Clear cache for new language
        this.saveUserPreferences();
    }

    async processText(text) {
        if (!this.containsJapanese(text)) {
            return {text, hasJapanese: false};
        }

        const [romaji, translation] = await Promise.all([this.getRomanization(text), this.getTranslation(text)]);

        return {
            text, romaji, translation, hasJapanese: true
        };
    }

    getStatistics() {
        return {
            romajiCacheSize: this.romajiCache.size,
            translationCacheSize: this.translationCache.size,
            activeTooltips: this.tooltips.size,
            isEnabled: this.isEnabled,
            preferences: {
                showRomaji: this.showRomaji,
                showTranslations: this.showTranslations,
                autoDetect: this.autoDetect,
                currentLanguage: this.currentLanguage
            }
        };
    }

    destroy() {
        // Cleanup all tooltips
        this.tooltips.forEach(tooltip => {
            if (tooltip.parentNode) {
                tooltip.parentNode.removeChild(tooltip);
            }
        });
        this.tooltips.clear();

        // Remove control panel
        const panel = document.querySelector('.japanese-control-panel');
        if (panel) {
            panel.remove();
        }

        // Clear caches
        this.romajiCache.clear();
        this.translationCache.clear();

        // Remove enhanced attributes
        const enhancedElements = document.querySelectorAll('[data-enhanced]');
        enhancedElements.forEach(el => {
            delete el.dataset.enhanced;
        });
    }
}

// Utility functions for Japanese text processing
const JapaneseUtils = {
    /**
     * Check if text contains Japanese characters
     */
    containsJapanese(text) {
        return /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]/.test(text);
    },

    /**
     * Count Japanese characters in text
     */
    countJapaneseCharacters(text) {
        const matches = text.match(/[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]/g);
        return matches ? matches.length : 0;
    },

    /**
     * Extract only Japanese characters from text
     */
    extractJapanese(text) {
        const matches = text.match(/[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+/g);
        return matches ? matches.join('') : '';
    },

    /**
     * Determine the type of Japanese characters
     */
    analyzeJapaneseText(text) {
        const hiragana = (text.match(/[\u3040-\u309F]/g) || []).length;
        const katakana = (text.match(/[\u30A0-\u30FF]/g) || []).length;
        const kanji = (text.match(/[\u4E00-\u9FFF]/g) || []).length;

        return {
            hiragana,
            katakana,
            kanji,
            total: hiragana + katakana + kanji,
            primaryScript: hiragana > katakana && hiragana > kanji ? 'hiragana' : katakana > hiragana && katakana > kanji ? 'katakana' : 'kanji'
        };
    },

    /**
     * Common martial arts terms dictionary
     */
    martialArtsTerms: {
        '武道': {romaji: 'budō', translation: 'martial way'},
        '武術': {romaji: 'bujutsu', translation: 'martial art/technique'},
        '空手': {romaji: 'karate', translation: 'empty hand'},
        '柔道': {romaji: 'jūdō', translation: 'gentle way'},
        '剣道': {romaji: 'kendō', translation: 'way of the sword'},
        '合気道': {romaji: 'aikidō', translation: 'way of harmonious spirit'},
        '型': {romaji: 'kata', translation: 'form/pattern'},
        '組手': {romaji: 'kumite', translation: 'sparring'},
        '道場': {romaji: 'dōjō', translation: 'training hall'},
        '先生': {romaji: 'sensei', translation: 'teacher'},
        '弟子': {romaji: 'deshi', translation: 'student/disciple'},
        '段': {romaji: 'dan', translation: 'rank/degree'},
        '級': {romaji: 'kyū', translation: 'grade'},
        '帯': {romaji: 'obi', translation: 'belt'},
        '黒帯': {romaji: 'kuro-obi', translation: 'black belt'},
        '白帯': {romaji: 'shiro-obi', translation: 'white belt'},
        '気': {romaji: 'ki', translation: 'spirit/energy'},
        '心': {romaji: 'kokoro/shin', translation: 'heart/mind'},
        '礼': {romaji: 'rei', translation: 'bow/respect'},
        '和': {romaji: 'wa', translation: 'harmony'}
    },

    /**
     * Look up martial arts term
     */
    lookupMartialArtsTerm(japanese) {
        return this.martialArtsTerms[japanese] || null;
    }
};

// Initialize Japanese text manager when DOM is ready
let japaneseTextManager;

document.addEventListener('DOMContentLoaded', function () {
    // read flag from data-attribute, meta, or global
    const container = document.querySelector('.viewer-content');
    const dsFlag = container?.dataset?.japaneseDetected;
    const metaFlag = document.querySelector('meta[name="x-japanese-detected"]')?.getAttribute('content');
    const globalFlag = (typeof window !== 'undefined' && window.__processingMeta) ? window.__processingMeta.has_japanese : undefined;

    const hasJapanese = (typeof globalFlag !== 'undefined' ? !!globalFlag : (typeof dsFlag !== 'undefined' ? /^(true|1|yes)$/i.test(dsFlag) : (typeof metaFlag !== 'undefined' && metaFlag !== null ? /^(true|1|yes)$/i.test(metaFlag) : false)));

    if (!hasJapanese) {
        // gate: do not initialize tools/UI if there is no real kana/kanji
        return;
    }

    japaneseTextManager = new JapaneseTextManager();
    japaneseTextManager.init();
});

// CSS for Japanese text interactions (injected dynamically)
const japaneseCSS = `
    .japanese-container {
        display: inline-block;
        position: relative;
        margin: 0 2px;
        padding: 2px 4px;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .japanese-container:hover {
        background: rgba(142, 68, 173, 0.1);
    }
    
    .japanese-container.expanded {
        background: rgba(142, 68, 173, 0.15);
        padding: 4px 6px;
    }
    
    .japanese-text {
        color: #8e44ad;
        font-weight: 600;
        cursor: pointer;
        user-select: none;
    }
    
    .romaji, .translation {
        display: block;
        font-size: 0.85em;
        margin-top: 2px;
        padding: 2px 4px;
        border-radius: 3px;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .romaji {
        color: #7f8c8d;
        font-style: italic;
    }
    
    .translation {
        color: #27ae60;
        font-weight: 500;
    }
    
    .japanese-loading {
        color: #bdc3c7;
        font-size: 0.8em;
        animation: pulse 1.5s infinite;
    }
    
    .japanese-error {
        color: #e74c3c;
        font-size: 0.9em;
    }
    
    .confidence-indicator {
        position: absolute;
        top: -8px;
        right: -8px;
        background: #3498db;
        color: white;
        font-size: 0.7em;
        padding: 1px 4px;
        border-radius: 8px;
        opacity: 0.8;
    }
    
    .confidence-indicator.low-confidence {
        background: #e74c3c;
    }
    
    .japanese-tooltip {
        position: fixed;
        background: #2c3e50;
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.9em;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        pointer-events: none;
        max-width: 300px;
    }
    
    .tooltip-japanese {
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .tooltip-romaji {
        font-style: italic;
        color: #bdc3c7;
        margin-bottom: 2px;
    }
    
    .tooltip-translation {
        color: #2ecc71;
    }
    
    .japanese-context-menu {
        position: fixed;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        z-index: 10001;
        min-width: 150px;
    }
    
    .context-menu-item {
        padding: 8px 12px;
        cursor: pointer;
        transition: background 0.2s ease;
    }
    
    .context-menu-item:hover {
        background: #f8f9fa;
    }
    
    .context-menu-item:first-child {
        border-radius: 6px 6px 0 0;
    }
    
    .context-menu-item:last-child {
        border-radius: 0 0 6px 6px;
    }
    
    .selection-actions {
        position: fixed;
        background: #34495e;
        border-radius: 6px;
        padding: 4px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        z-index: 10002;
        display: flex;
        gap: 4px;
    }
    
    .action-btn {
        padding: 6px 12px;
        background: transparent;
        border: none;
        color: white;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.85em;
        transition: background 0.2s ease;
    }
    
    .action-btn:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .japanese-notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: #2c3e50;
        color: white;
        padding: 12px 20px;
        border-radius: 6px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        z-index: 10003;
        opacity: 1;
        transition: opacity 0.3s ease;
    }
    
    .japanese-notification.fade-out {
        opacity: 0;
    }
    
    .japanese-control-panel {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        z-index: 9999;
        width: 250px;
        font-size: 0.9em;
    }
    
    .japanese-control-panel.collapsed .panel-content {
        display: none;
    }
    
    .panel-header {
        padding: 12px 16px;
        background: #f8f9fa;
        border-radius: 8px 8px 0 0;
        border-bottom: 1px solid #dee2e6;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .panel-header h4 {
        margin: 0;
        font-size: 1em;
        color: #2c3e50;
    }
    
    .panel-toggle {
        background: none;
        border: none;
        font-size: 1.2em;
        cursor: pointer;
        color: #6c757d;
        padding: 0;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .panel-content {
        padding: 16px;
    }
    
    .panel-content label {
        display: block;
        margin-bottom: 8px;
        cursor: pointer;
    }
    
    .panel-content input[type="checkbox"] {
        margin-right: 8px;
    }
    
    .panel-actions {
        margin-top: 12px;
        display: flex;
        gap: 8px;
    }
    
    .panel-actions button {
        flex: 1;
        padding: 6px 12px;
        border: 1px solid #dee2e6;
        background: white;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.85em;
        transition: all 0.2s ease;
    }
    
    .panel-actions button:hover {
        background: #f8f9fa;
        border-color: #adb5bd;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @media (max-width: 768px) {
        .japanese-control-panel {
            bottom: 10px;
            right: 10px;
            width: 200px;
        }
        
        .japanese-tooltip {
            max-width: 250px;
            font-size: 0.85em;
        }
    }
`;

// Inject CSS
const styleSheet = document.createElement('style');
styleSheet.textContent = japaneseCSS;
document.head.appendChild(styleSheet);

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {JapaneseTextManager, JapaneseUtils};
}