# Static Assets Overview

This directory contains all static assets for the Martial Arts OCR application, including stylesheets, JavaScript files, and dynamically generated content. These assets provide the complete frontend experience for document upload, processing, and viewing.

## Directory Structure

```
static/
├── css/
│   ├── main.css           # Core application styles
│   └── viewer.css         # Document viewer specific styles
├── js/
│   ├── upload.js          # File upload and processing
│   ├── viewer.js          # Document viewer functionality
│   └── japanese.js        # Japanese text interactions
├── extracted_content/     # Dynamically generated content
└── OVERVIEW.md           # This documentation file
```

## Design Philosophy

The static assets are designed around the specific needs of academic martial arts research:

- **Professional Academic Appearance**: Clean, scholarly interface suitable for research
- **Japanese Text Support**: Specialized handling of mixed English/Japanese content
- **Document Preservation**: Maintain original layout and embedded diagrams
- **Accessibility**: Keyboard navigation, screen reader support, and mobile compatibility
- **Performance**: Optimized loading, caching, and efficient interactions

---

## CSS Architecture (`css/`)

### `main.css` - Core Application Styles

**Purpose**: Provides the foundational design system for the entire application.

**Key Features**:
- **Modern CSS Reset**: Consistent baseline across browsers
- **Typography System**: Optimized for academic reading with serif fonts for content
- **Color Palette**: Professional blues and grays with martial arts-appropriate accents
- **Responsive Grid**: Flexible layouts that work on all screen sizes
- **Component Library**: Reusable buttons, forms, alerts, and cards

**Major Sections**:
```css
/* CSS Reset and Base Styles */
/* Typography and Colors */
/* Layout Components (header, nav, main, footer) */
/* Upload Interface */
/* File Preview and Processing */
/* Gallery View */
/* Buttons and Form Elements */
/* Alerts and Messages */
/* Responsive Design */
/* Utility Classes */
```

**Usage Example**:
```html
<div class="container">
    <div class="content-section">
        <h2>Document Title</h2>
        <p>Content here...</p>
        <button class="btn btn-primary">Action</button>
    </div>
</div>
```

### `viewer.css` - Document Viewer Styles

**Purpose**: Specialized styles for the document viewing experience with advanced layout control.

**Key Features**:
- **Multi-pane Layout**: Toolbar, sidebar, and content areas
- **View Modes**: Original layout, text-only, and side-by-side views
- **Zoom Controls**: Visual scaling with transform-based zoom
- **Japanese Text Styling**: Special highlighting and interaction states
- **Image Display**: Embedded image handling with modal views

**Major Sections**:
```css
/* Document Viewer Layout */
/* Toolbar and Controls */
/* Content Display Modes */
/* Text Content Styling */
/* Japanese Text Styling */
/* Image and Diagram Display */
/* Sidebar Components */
/* Fullscreen Mode */
/* Search and Highlight */
/* Loading and Error States */
/* Print Styles */
/* Responsive Design */
```

**Japanese Text Styling**:
```css
.japanese-segment {
    background: linear-gradient(120deg, #f8f5ff 0%, #f0ebff 100%);
    border-left: 4px solid #8e44ad;
}

.japanese-text {
    color: #8e44ad;
    font-weight: 600;
    cursor: help;
}

.romaji {
    color: #7f8c8d;
    font-style: italic;
}

.translation {
    color: #27ae60;
    font-weight: 500;
}
```

---

## JavaScript Architecture (`js/`)

### `upload.js` - File Upload and Processing

**Purpose**: Handles the complete file upload workflow from selection to processing completion.

**Core Class**: `UploadManager`

**Key Features**:
- **Drag-and-Drop Interface**: Visual feedback and file validation
- **Progress Tracking**: Real-time upload and processing status
- **File Validation**: Size limits, format checking, and security validation
- **Processing Monitoring**: Polls server for OCR completion status
- **Error Recovery**: Graceful error handling with user-friendly messages

**Workflow**:
```javascript
// 1. File Selection/Drop
uploadManager.handleFileSelect(files)

// 2. Validation
uploadManager.validateFile(file)

// 3. Preview
uploadManager.showFilePreview(file)

// 4. Upload with Progress
uploadManager.uploadWithProgress(formData)

// 5. Processing Monitor
uploadManager.startProcessingMonitor(documentId)

// 6. Completion/Redirect
window.location.href = `/view/${documentId}`
```

**Integration Points**:
- Flask routes: `/upload`, `/api/status/<id>`, `/view/<id>`
- CSS classes from `main.css` for styling
- Server-side processing pipeline

**Usage Example**:
```javascript
const uploadManager = new UploadManager();

// Programmatically trigger upload
uploadManager.startUpload();

// Handle upload completion
uploadManager.handleUploadSuccess(response);

// Reset for new upload
uploadManager.clearFiles();
```

### `viewer.js` - Document Viewer Functionality

**Purpose**: Provides advanced document viewing capabilities with zoom, search, and layout controls.

**Core Class**: `DocumentViewer`

**Key Features**:
- **Zoom Controls**: Mouse wheel, keyboard shortcuts, and button controls
- **View Modes**: Switch between original, text-only, and side-by-side layouts
- **Search Functionality**: Real-time text search with highlighting
- **Fullscreen Mode**: Immersive document viewing
- **Keyboard Shortcuts**: Professional shortcuts for power users

**Keyboard Shortcuts**:
```
Ctrl/Cmd + +/-/0  : Zoom in/out/reset
F11 or 'f'        : Toggle fullscreen
Escape            : Exit fullscreen, clear search
/                 : Focus search
Ctrl/Cmd + s      : Toggle sidebar
Ctrl/Cmd + p      : Print document
```

**View Modes**:
```javascript
// Set viewing mode
documentViewer.setViewMode('text-only');  // Clean text view
documentViewer.setViewMode('original');   // Preserve layout
documentViewer.setViewMode('side-by-side'); // Comparison view
```

**Search Features**:
```javascript
// Perform search
documentViewer.performSearch('budō');

// Navigate results
documentViewer.navigateSearch(1);  // Next result
documentViewer.navigateSearch(-1); // Previous result

// Clear search
documentViewer.clearSearch();
```

### `japanese.js` - Japanese Text Interactions

**Purpose**: Specialized handling of Japanese text with romanization, translation, and user interactions.

**Core Class**: `JapaneseTextManager`

**Key Features**:
- **Auto-Detection**: Finds Japanese text in any content
- **Interactive Tooltips**: Hover for romanization and translation
- **Text Selection**: Right-click or select text for actions
- **Control Panel**: User preferences for Japanese display
- **Martial Arts Dictionary**: Specialized terminology database

**Japanese Character Ranges**:
```javascript
japaneseRanges = {
    hiragana: [0x3040, 0x309F],           // あいうえお
    katakana: [0x30A0, 0x30FF],           // アイウエオ
    kanji: [0x4E00, 0x9FFF],              // 武道空手
    halfWidthKatakana: [0xFF65, 0xFF9F],  // ｱｲｳｴｵ
}
```

**Processing Pipeline**:
```javascript
// 1. Auto-detect Japanese text
JapaneseTextManager.extractJapaneseSegments(text)

// 2. Create interactive elements
JapaneseTextManager.createJapaneseElement(japaneseText)

// 3. Process asynchronously
JapaneseTextManager.processJapaneseTextAsync(text, container)

// 4. Add romanization and translation
JapaneseTextManager.addRomajiElement(container, romaji)
JapaneseTextManager.addTranslationElement(container, translation)
```

**Martial Arts Terms Dictionary**:
```javascript
martialArtsTerms = {
    '武道': { romaji: 'budō', translation: 'martial way' },
    '武術': { romaji: 'bujutsu', translation: 'martial art/technique' },
    '空手': { romaji: 'karate', translation: 'empty hand' },
    '柔道': { romaji: 'jūdō', translation: 'gentle way' },
    '剣道': { romaji: 'kendō', translation: 'way of the sword' },
    // ... more terms
}
```

**User Preferences**:
```javascript
// Toggle features
japaneseTextManager.setRomajiDisplay(true);
japaneseTextManager.setTranslationDisplay(true);
japaneseTextManager.setTargetLanguage('en');

// Control panel shortcuts
Ctrl/Cmd + J  : Toggle Japanese mode
Ctrl/Cmd + R  : Toggle romanization
Ctrl/Cmd + T  : Toggle translations
```

---

## Dynamic Content (`extracted_content/`)

This directory is automatically created and populated by the application during document processing.

**Structure**:
```
extracted_content/
├── doc_1/
│   ├── images/
│   │   ├── diagram_1.jpg
│   │   ├── diagram_2.png
│   │   └── illustration_3.jpg
│   └── thumbnails/
│       ├── thumb_diagram_1.jpg
│       └── thumb_diagram_2.jpg
├── doc_2/
└── doc_3/
```

**Content Types**:
- **Extracted Images**: Diagrams and illustrations separated from text
- **Thumbnails**: Optimized previews for web display
- **Processed Assets**: Generated content from OCR pipeline

**File Naming Convention**:
- `doc_{document_id}/` - Document-specific folders
- `diagram_{n}.{ext}` - Technical diagrams and illustrations
- `thumb_{filename}` - Thumbnail versions
- `processed_{timestamp}.{ext}` - Timestamped processed files

---

## Integration Patterns

### CSS-JavaScript Integration

**CSS Classes for JavaScript**:
```css
/* Upload states */
.upload-zone.dragover { /* Drag feedback */ }
.file-preview { /* Preview styling */ }
.processing-status { /* Progress indication */ }

/* Viewer states */
.document-viewer.viewer-fullscreen { /* Fullscreen mode */ }
.content-container.text-only-view { /* View mode styling */ }
.highlight.current { /* Search highlighting */ }

/* Japanese text */
.japanese-container.expanded { /* Interaction state */ }
.japanese-tooltip { /* Tooltip styling */ }
```

**Event-Driven Architecture**:
```javascript
// Upload events
document.addEventListener('dragover', handleDragOver);
document.addEventListener('drop', handleFileDrop);

// Viewer events
document.addEventListener('keydown', handleKeyboardShortcuts);
document.addEventListener('wheel', handleMouseWheelZoom);

// Japanese text events
document.addEventListener('mouseover', showJapaneseTooltip);
document.addEventListener('click', handleJapaneseClick);
```

### Server API Integration

**Upload Endpoints**:
```javascript
POST /upload          // File upload
GET  /api/status/{id} // Processing status
GET  /view/{id}       // View processed document
```

**Japanese Processing**:
```javascript
POST /api/romanize    // Text romanization
POST /api/translate   // Text translation
```

**Response Handling**:
```javascript
// Status polling
const response = await fetch(`/api/status/${documentId}`);
const data = await response.json();

switch (data.status) {
    case 'processing': /* Show progress */
    case 'completed':  /* Redirect to viewer */
    case 'failed':     /* Show error */
}
```

---

## Performance Optimizations

### CSS Optimizations

**Critical CSS Inlining**:
- Core layout styles loaded immediately
- Secondary styles loaded progressively
- Print styles separated for faster initial load

**Asset Optimization**:
```css
/* Efficient animations */
.fade-in { transition: opacity 0.3s ease; }

/* Hardware acceleration */
.smooth-transform { transform: translateZ(0); }

/* Efficient selectors */
.japanese-text { /* Direct class selection */ }
```

### JavaScript Optimizations

**Lazy Loading**:
```javascript
// Load Japanese processing only when needed
if (document.querySelector('.japanese-text')) {
    japaneseTextManager = new JapaneseTextManager();
}
```

**Debounced Operations**:
```javascript
// Search debouncing
debounceSearch(query) {
    clearTimeout(this.searchTimeout);
    this.searchTimeout = setTimeout(() => {
        this.performSearch(query);
    }, 300);
}
```

**Caching Strategies**:
```javascript
// Romanization cache
romajiCache = new Map();
translationCache = new Map();

// State persistence
localStorage.setItem('viewer-state', JSON.stringify(state));
```

---

## Responsive Design Strategy

### Breakpoints
```css
/* Mobile first approach */
@media (max-width: 480px)  { /* Small phones */ }
@media (max-width: 768px)  { /* Tablets/large phones */ }
@media (max-width: 1024px) { /* Small laptops */ }
@media (min-width: 1200px) { /* Large screens */ }
```

### Mobile Adaptations

**Upload Interface**:
- Larger touch targets for mobile
- Simplified drag-and-drop with clear instructions
- Progress feedback optimized for small screens

**Document Viewer**:
- Collapsible sidebar on mobile
- Touch-friendly zoom controls
- Swipe gestures for navigation

**Japanese Text**:
- Larger touch targets for Japanese text
- Simplified tooltips for mobile
- Context menus adapted for touch

---

## Accessibility Features

### Keyboard Navigation
- Full keyboard support for all interactive elements
- Visible focus indicators
- Logical tab order

### Screen Reader Support
```html
<!-- ARIA labels for complex interactions -->
<button aria-label="Zoom in document view">+</button>
<div role="tooltip" aria-describedby="japanese-translation">武道</div>
```

### Color and Contrast
- WCAG 2.1 AA compliant color ratios
- Color is not the only indicator of state
- High contrast mode support

---

## Development Workflow

### Adding New Styles
1. Add to appropriate CSS file (`main.css` or `viewer.css`)
2. Follow BEM naming convention
3. Test responsive behavior
4. Verify accessibility

### Adding New JavaScript Features
1. Choose appropriate file based on functionality
2. Follow existing class patterns
3. Add proper error handling
4. Implement responsive behavior
5. Test keyboard accessibility

### Testing Checklist
- [ ] Desktop browsers (Chrome, Firefox, Safari, Edge)
- [ ] Mobile devices (iOS Safari, Android Chrome)
- [ ] Keyboard navigation
- [ ] Screen reader compatibility
- [ ] High contrast mode
- [ ] Print functionality

---

## Future Enhancements

### Planned Features
- **Progressive Web App**: Offline functionality and app-like experience
- **Advanced Search**: Fuzzy matching and OCR confidence-based search
- **Collaborative Features**: Shared viewing and annotation
- **Enhanced Japanese Support**: Furigana rendering and advanced linguistics

### Performance Improvements
- **Bundle Splitting**: Separate chunks for different features
- **Service Worker**: Caching and offline support
- **Image Optimization**: WebP support and lazy loading
- **CSS Optimization**: Critical CSS extraction and purging

### Accessibility Enhancements
- **Voice Navigation**: Speech recognition for document control
- **High Contrast Themes**: Multiple contrast options
- **Screen Reader Optimization**: Enhanced ARIA support
- **Motor Accessibility**: Alternative input methods

This static assets directory provides a complete, professional frontend experience specifically designed for academic martial arts research with sophisticated Japanese text handling and document preservation capabilities.

