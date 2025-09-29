# Templates Overview

This directory contains all HTML templates for the Martial Arts OCR application, built using Jinja2 templating engine with Flask. These templates provide a professional, accessible interface specifically designed for academic martial arts research with sophisticated Japanese text support.

## Template Architecture

```
templates/
├── base.html          # Base template with common structure
├── index.html         # Document upload interface
├── gallery.html       # Document gallery and management
├── page_view.html     # Document viewer with OCR results
└── OVERVIEW.md        # This documentation file
```

## Design Philosophy

The templates are architected around the specific needs of academic martial arts digitization:

- **Academic Presentation**: Professional, scholarly interface suitable for research
- **Japanese Text Support**: Specialized handling of mixed English/Japanese content
- **Document Preservation**: Maintain original layout and embedded technical diagrams
- **Accessibility First**: WCAG 2.1 AA compliance with keyboard navigation and screen reader support
- **Progressive Enhancement**: Works without JavaScript, enhanced with it
- **Mobile Responsive**: Touch-friendly interface that works on all devices

---

## Template Inheritance Structure

### Base Template (`base.html`)

**Purpose**: Foundation template providing common structure, navigation, and functionality.

**Key Features**:
- **HTML5 semantic structure** with proper ARIA roles
- **Meta tags** for SEO, Open Graph, and mobile optimization
- **Progressive enhancement** with no-js/js class switching
- **Flash message system** with auto-dismiss and manual close
- **Navigation system** with active page highlighting
- **Japanese mode toggle** integrated into navigation
- **Accessibility controls** (font size, high contrast)

**Template Blocks Provided**:
```html
{% block title %}{% endblock %}           <!-- Page title -->
{% block description %}{% endblock %}     <!-- Meta description -->
{% block head %}{% endblock %}            <!-- Additional head content -->
{% block body_class %}{% endblock %}      <!-- Body CSS classes -->
{% block extra_css %}{% endblock %}       <!-- Additional stylesheets -->
{% block header_actions %}{% endblock %}  <!-- Header action buttons -->
{% block extra_nav %}{% endblock %}       <!-- Additional navigation items -->
{% block page_header %}{% endblock %}     <!-- Page header with title -->
{% block breadcrumbs %}{% endblock %}     <!-- Navigation breadcrumbs -->
{% block page_title %}{% endblock %}      <!-- Main page title -->
{% block page_subtitle %}{% endblock %}   <!-- Page subtitle/description -->
{% block container_class %}{% endblock %} <!-- Container CSS class -->
{% block content %}{% endblock %}         <!-- Main page content -->
{% block scripts %}{% endblock %}         <!-- Page-specific JavaScript -->
{% block analytics %}{% endblock %}       <!-- Analytics/tracking scripts -->
```

**Common Functionality**:
```html
<!-- Flash messages with icons and auto-dismiss -->
{% with messages = get_flashed_messages(with_categories=true) %}
    {% for category, message in messages %}
        <div class="alert alert-{{ category }}">
            <span class="alert-text">{{ message }}</span>
        </div>
    {% endfor %}
{% endwith %}

<!-- Active navigation highlighting -->
<a href="{{ url_for('index') }}" 
   class="nav-link {{ 'active' if request.endpoint == 'index' else '' }}">
    Upload
</a>

<!-- Japanese mode toggle -->
<button class="nav-link japanese-toggle" id="japanese-mode-toggle">
    <span class="toggle-text">日本語</span>
</button>
```

**JavaScript Integration**:
- **Loading screen management** with smooth transitions
- **Flash message handling** with auto-dismiss timers
- **Japanese text toggle** integration
- **Accessibility controls** with localStorage persistence
- **Global error handling** with user-friendly messages

---

## Page Templates

### Upload Interface (`index.html`)

**Purpose**: Professional document upload interface with guidance and recent activity.

**Extends**: `base.html`

**Key Sections**:

**Hero Section**:
```html
<div class="upload-hero">
    <h2>Martial Arts Document OCR</h2>
    <div class="hero-features">
        <div class="feature-item">
            <span class="feature-icon">🇯🇵</span>
            <span>Japanese Text Support</span>
        </div>
        <!-- More features... -->
    </div>
</div>
```

**Upload Instructions**:
- **Step-by-step guide** with numbered visual instructions
- **Best practices** for optimal OCR results
- **Technical specifications** (DPI, file size, formats)

**Upload Zone**:
```html
<form id="upload-form" enctype="multipart/form-data">
    <div class="upload-zone" id="upload-zone">
        <div class="upload-icon">📄</div>
        <div class="upload-text">Drag and drop your scanned document here</div>
        <input type="file" name="file" class="file-input" accept="image/*">
    </div>
</form>
```

**Supported Formats Grid**:
- **Visual format cards** for JPEG, PNG, TIFF, BMP
- **Format descriptions** and use case recommendations
- **File size limitations** and quality guidelines

**Recent Activity**:
```html
{% if recent_documents %}
<div class="recent-uploads">
    {% for document in recent_documents[:5] %}
    <div class="recent-item">
        <!-- Status indicator, metadata, quick actions -->
    </div>
    {% endfor %}
</div>
{% endif %}
```

**Integration Points**:
- **Upload.js integration** for drag-and-drop functionality
- **Real-time status updates** for processing documents
- **Auto-refresh** when documents are processing
- **Keyboard shortcuts** (Ctrl+U for upload)

### Document Gallery (`gallery.html`)

**Purpose**: Professional document management with advanced filtering, search, and batch operations.

**Extends**: `base.html`

**Key Features**:

**Advanced Controls**:
```html
<div class="gallery-controls">
    <div class="search-box">
        <input type="text" class="search-input" 
               placeholder="Search documents...">
    </div>
    <div class="filter-controls">
        <select id="status-filter">
            <option value="">All Documents</option>
            <option value="completed">Completed</option>
            <option value="processing">Processing</option>
        </select>
        <div class="view-toggle">
            <button class="view-btn" data-view="grid">⊞</button>
            <button class="view-btn" data-view="list">☰</button>
        </div>
    </div>
</div>
```

**Statistics Dashboard**:
```html
<div class="gallery-stats">
    <div class="stat-item">
        <span class="stat-value">{{ documents|length }}</span>
        <span class="stat-label">Total Documents</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">{{ completed_count }}</span>
        <span class="stat-label">Processed</span>
    </div>
</div>
```

**Document Cards**:
```html
{% for document in documents %}
<div class="document-card" data-status="{{ document.status }}">
    <!-- Thumbnail with status badge -->
    <div class="card-thumbnail-container">
        <img src="{{ thumbnail_url }}" class="card-thumbnail">
        <div class="status-badge status-{{ document.status }}">
            {% if document.status == 'completed' %}✅{% endif %}
        </div>
    </div>
    
    <!-- Metadata and actions -->
    <div class="card-content">
        <h3 class="card-title">{{ document.original_filename }}</h3>
        <div class="card-meta">
            <!-- Upload date, file size, processing stats -->
        </div>
        <div class="card-actions">
            <!-- Context-sensitive action buttons -->
        </div>
    </div>
</div>
{% endfor %}
```

**Advanced Interactions**:
- **Real-time search** with debounced input
- **Multi-criteria filtering** by status, date, name, size
- **View mode switching** (grid/list) with localStorage persistence
- **Dropdown menus** for document operations (rename, duplicate, delete)
- **Auto-refresh** for processing documents

**Empty State Handling**:
```html
{% else %}
<div class="empty-gallery">
    <div class="empty-icon">📚</div>
    <h3 class="empty-title">No Documents Found</h3>
    <p class="empty-description">Start by uploading your first document...</p>
    <a href="{{ url_for('index') }}" class="btn btn-primary btn-lg">
        Upload Your First Document
    </a>
</div>
{% endif %}
```

### Document Viewer (`page_view.html`)

**Purpose**: Comprehensive document viewing with OCR results, Japanese text processing, and embedded image display.

**Extends**: `base.html`

**Key Components**:

**Document Header**:
```html
<div class="document-header">
    <div class="document-info">
        <h2 class="document-title">{{ document.original_filename }}</h2>
        <div class="document-meta">
            <div class="meta-group">
                <span class="meta-icon">📅</span>
                <span>Uploaded: {{ document.upload_date.strftime('%Y-%m-%d %H:%M') }}</span>
            </div>
        </div>
        <div class="document-status {{ document.status }}">
            {% if document.status == 'completed' %}✅ Processing Completed{% endif %}
        </div>
    </div>
</div>
```

**Advanced Viewer Interface**:
```html
{% if document.status == 'completed' %}
<div class="document-viewer">
    <!-- Toolbar with view controls -->
    <div class="viewer-toolbar">
        <div class="viewer-actions">
            <button class="viewer-btn view-original active">📄 Original</button>
            <button class="viewer-btn view-text-only">📝 Text Only</button>
        </div>
        <div class="zoom-controls">
            <button class="zoom-btn zoom-out">−</button>
            <span class="zoom-level">100%</span>
            <button class="zoom-btn zoom-in">+</button>
        </div>
    </div>
    
    <!-- Main viewer area -->
    <div class="viewer-main">
        <div class="viewer-sidebar">
            <!-- Document info, stats, search -->
        </div>
        <div class="viewer-content">
            <!-- OCR text and extracted images -->
        </div>
    </div>
</div>
{% endif %}
```

**State-Specific Views**:

**Completed State**:
- **Full viewer interface** with all features enabled
- **OCR text display** with Japanese text processing
- **Extracted images** with modal viewing
- **Search functionality** within document text
- **Statistics sidebar** with confidence scores

**Processing State**:
```html
{% elif document.status == 'processing' %}
<div class="viewer-loading">
    <div class="loading-spinner"></div>
    <div class="loading-text">Processing document...</div>
    <button class="btn btn-secondary" onclick="refreshStatus()">Check Status</button>
</div>
{% endif %}
```

**Error State**:
```html
{% else %}
<div class="viewer-error">
    <div class="error-icon">⚠️</div>
    <div class="error-message">Unable to display document</div>
    <div class="error-actions">
        <button class="btn btn-warning" onclick="retryProcessing()">🔄 Retry</button>
    </div>
</div>
{% endif %}
```

**Japanese Text Integration**:
```html
<div class="ocr-content">
    {{ result.ocr_text|safe }}  <!-- Processed with Japanese markup -->
</div>
```

---

## Template Data Flow

### Data Models Integration

**Document Model**:
```python
# Available in templates
document.id                    # Unique identifier
document.original_filename     # User-provided filename
document.upload_date          # Upload timestamp
document.processing_date      # Processing completion
document.status              # uploaded, processing, completed, failed
document.error_message       # Error details if failed
document.file_size          # File size in bytes
```

**Processing Results**:
```python
# Available when status == 'completed'
result.ocr_text             # Extracted text content
result.ocr_confidence       # OCR accuracy percentage
result.processing_time      # Processing duration
result.has_japanese        # Japanese text detected
result.japanese_data       # Romanization/translation data
result.extracted_images    # List of embedded images
```

### Template Context Variables

**Global Context** (available in all templates):
```python
request.endpoint            # Current route name
request.url                # Current URL
get_flashed_messages()     # Flash message system
url_for()                  # URL generation
```

**Page-Specific Context**:
```python
# index.html
recent_documents           # Last 5 uploaded documents

# gallery.html
documents                 # All documents (paginated)
pagination               # Pagination object

# page_view.html
document                 # Current document object
page                    # Page object (if exists)
result                  # Processing result (if completed)
```

---

## Responsive Design Strategy

### Mobile-First Approach

**Breakpoints**:
```css
/* Templates adapt at these breakpoints */
@media (max-width: 480px)  { /* Small phones */ }
@media (max-width: 768px)  { /* Tablets/large phones */ }
@media (max-width: 1024px) { /* Small laptops */ }
```

**Mobile Adaptations**:

**Navigation**:
- **Collapsible navigation** on mobile
- **Touch-friendly buttons** with appropriate sizing
- **Simplified header** on small screens

**Upload Interface**:
- **Larger drop zones** for touch interaction
- **Simplified instructions** on mobile
- **Stacked form elements** for narrow screens

**Gallery**:
- **Single column** layout on mobile
- **List view optimization** for touch scrolling
- **Simplified filtering** controls

**Document Viewer**:
- **Collapsible sidebar** on mobile
- **Touch-friendly zoom** controls
- **Simplified toolbar** with essential functions

### Typography and Spacing

**Academic Typography**:
```css
/* Reading-optimized typography */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
}

.ocr-content {
    font-family: 'Georgia', 'Times New Roman', serif;
    line-height: 1.8;
    font-size: 1.1rem;
}

.japanese-text {
    color: #8e44ad;
    font-weight: 600;
}
```

**Consistent Spacing System**:
```css
/* Spacing scale used throughout templates */
.mt-1 { margin-top: 0.5rem; }    /* 8px */
.mt-2 { margin-top: 1rem; }      /* 16px */
.mt-3 { margin-top: 1.5rem; }    /* 24px */
.mb-1 { margin-bottom: 0.5rem; } /* 8px */
.mb-2 { margin-bottom: 1rem; }   /* 16px */
.mb-3 { margin-bottom: 1.5rem; } /* 24px */
```

---

## Accessibility Implementation

### WCAG 2.1 AA Compliance

**Semantic HTML**:
```html
<!-- Proper heading hierarchy -->
<h1>Main page title</h1>
  <h2>Section title</h2>
    <h3>Subsection title</h3>

<!-- Landmark roles -->
<header role="banner">
<nav role="navigation" aria-label="Main navigation">
<main role="main" id="main-content">
<footer role="contentinfo">

<!-- Form labels and descriptions -->
<label for="file-input">Choose document file</label>
<input type="file" id="file-input" aria-describedby="file-help">
<div id="file-help">Supported formats: JPG, PNG, TIFF, BMP</div>
```

**ARIA Attributes**:
```html
<!-- Dynamic content -->
<div class="flash-messages" role="alert" aria-live="polite">
<button aria-pressed="false" aria-label="Toggle Japanese mode">

<!-- Navigation states -->
<a href="/gallery" class="nav-link" aria-current="page">Gallery</a>

<!-- Interactive elements -->
<button class="btn" aria-describedby="btn-help">Process Document</button>
<div id="btn-help">This will start OCR processing</div>
```

**Keyboard Navigation**:
```html
<!-- Skip link for screen readers -->
<a href="#main-content" class="skip-link">Skip to main content</a>

<!-- Focus management -->
<div tabindex="0" role="button" onkeydown="handleKeydown(event)">
```

**Color and Contrast**:
- **WCAG AA compliant** color ratios (4.5:1 minimum)
- **Color is not the only indicator** of state or meaning
- **High contrast mode** support via CSS classes

### Screen Reader Support

**Image Accessibility**:
```html
<!-- Descriptive alt text -->
<img src="thumbnail.jpg" alt="Thumbnail of martial arts technique diagram showing proper stance positioning">

<!-- Decorative images -->
<span class="icon" aria-hidden="true">📄</span>

<!-- Complex images -->
<img src="chart.png" alt="Processing statistics chart" aria-describedby="chart-description">
<div id="chart-description">
    Chart showing 85% OCR confidence, 2.3 second processing time, and 3 extracted images.
</div>
```

**Dynamic Content Announcements**:
```html
<!-- Status updates -->
<div aria-live="polite" id="status-announcements"></div>

<!-- Error messages -->
<div role="alert" class="error-message">
    Processing failed: File format not supported
</div>
```

---

## JavaScript Integration

### Progressive Enhancement

**No-JavaScript Fallbacks**:
```html
<!-- Basic form submission without JavaScript -->
<form method="POST" action="{{ url_for('upload_file') }}" enctype="multipart/form-data">
    <input type="file" name="file" required>
    <button type="submit">Upload Document</button>
</form>

<!-- Navigation without JavaScript -->
<noscript>
    <style>
        .js-only { display: none !important; }
        .no-js-message { display: block; }
    </style>
</noscript>
```

**Enhanced with JavaScript**:
```html
<!-- Drag-and-drop enhancement -->
<div class="upload-zone js-enhanced">
    <!-- Enhanced upload interface -->
</div>

<!-- Real-time search -->
<input type="text" class="search-input js-enhanced" 
       placeholder="Search documents (enhanced with JavaScript)">
```

### Template-JavaScript Communication

**Data Attributes**:
```html
<!-- Passing data to JavaScript -->
<div class="document-card" 
     data-status="{{ document.status }}"
     data-filename="{{ document.original_filename|lower }}"
     data-date="{{ document.upload_date.isoformat() if document.upload_date else '' }}">
```

**Global JavaScript Variables**:
```html
<script>
// Make template data available to JavaScript
window.documentData = {
    id: {{ document.id }},
    status: '{{ document.status }}',
    hasJapanese: {{ result.has_japanese|lower if result else 'false' }},
};
</script>
```

**Event Integration**:
```html
<!-- Custom event dispatching -->
<script>
document.addEventListener('upload:complete', function(e) {
    // Handle upload completion
    window.location.href = '/view/' + e.detail.documentId;
});
</script>
```

---

## Performance Optimization

### Template Optimization

**Conditional Loading**:
```html
<!-- Load viewer assets only when needed -->
{% if document.status == 'completed' %}
<script src="{{ url_for('static', filename='js/viewer.js') }}"></script>
<script src="{{ url_for('static', filename='js/japanese.js') }}"></script>
{% endif %}

<!-- Lazy load images -->
<img src="{{ thumbnail_url }}" loading="lazy" alt="Document thumbnail">
```

**Critical CSS Inlining**:
```html
<style>
/* Critical above-the-fold CSS */
.loading-screen { /* Inline critical styles */ }
</style>

<!-- Non-critical CSS -->
<link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
```

**Asset Optimization**:
```html
<!-- Preconnect to external domains -->
<link rel="preconnect" href="https://fonts.googleapis.com">

<!-- Preload critical resources -->
<link rel="preload" href="{{ url_for('static', filename='css/main.css') }}" as="style">
```

### Caching Strategy

**Template Caching**:
```python
# Flask template caching
@app.route('/gallery')
@cache.cached(timeout=300)  # 5 minute cache
def gallery():
    return render_template('gallery.html', documents=documents)
```

**Static Asset Versioning**:
```html
<!-- Cache busting with file hashes -->
<link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}?v={{ version_hash }}">
```

---

## Error Handling and User Feedback

### Flash Message System

**Message Categories**:
```python
# Flash message types
flash('Document uploaded successfully', 'success')
flash('Processing failed', 'error')
flash('OCR confidence is low', 'warning')
flash('Processing started', 'info')
```

**Template Rendering**:
```html
{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        {% for category, message in messages %}
        <div class="alert alert-{{ category }}" role="alert">
            <span class="alert-icon">
                {% if category == 'error' %}⚠️
                {% elif category == 'success' %}✅
                {% elif category == 'warning' %}⚠️
                {% elif category == 'info' %}ℹ️
                {% endif %}
            </span>
            <span class="alert-text">{{ message }}</span>
            <button class="alert-close" aria-label="Close message">×</button>
        </div>
        {% endfor %}
    {% endif %}
{% endwith %}
```

### Error State Templates

**Graceful Degradation**:
```html
<!-- Document not found -->
{% if not document %}
<div class="error-state">
    <h2>Document Not Found</h2>
    <p>The requested document could not be found.</p>
    <a href="{{ url_for('gallery') }}" class="btn btn-primary">Return to Gallery</a>
</div>
{% endif %}

<!-- Processing failed -->
{% if document.status == 'failed' %}
<div class="error-info">
    <div class="error-title">Processing Error</div>
    <div class="error-message">{{ document.error_message or 'Unknown error occurred' }}</div>
    <button class="btn btn-warning" onclick="retryProcessing({{ document.id }})">
        Retry Processing
    </button>
</div>
{% endif %}
```

---

## Security Considerations

### Input Sanitization

**Template Auto-Escaping**:
```html
<!-- Jinja2 auto-escapes by default -->
<h1>{{ document.original_filename }}</h1>  <!-- Safe -->

<!-- Manual escaping when needed -->
<div>{{ user_content|e }}</div>

<!-- Safe HTML rendering (when content is pre-sanitized) -->
<div>{{ ocr_text|safe }}</div>
```

**XSS Prevention**:
```html
<!-- Avoid dangerous patterns -->
<!-- BAD: <script>var data = {{ json_data|tojsonfilter }};</script> -->

<!-- GOOD: Use data attributes -->
<div id="app" data-config="{{ json_data|tojsonfilter|e }}"></div>
```

### CSRF Protection

**Form Token Integration**:
```html
<!-- CSRF tokens in forms -->
<form method="POST">
    {{ csrf_token() }}
    <input type="file" name="file">
    <button type="submit">Upload</button>
</form>
```

---

## Future Enhancements

### Planned Template Features

**Progressive Web App (PWA)**:
- **Service worker integration** for offline functionality
- **App manifest** for native app-like experience
- **Push notifications** for processing completion

**Advanced Internationalization**:
- **Multi-language support** with proper RTL handling
- **Cultural adaptations** for different regions
- **Date/time localization** based on user preferences

**Enhanced Accessibility**:
- **Voice navigation** integration
- **Alternative input methods** for motor disabilities
- **Cognitive accessibility** improvements with simplified interfaces

**Performance Improvements**:
- **Template fragments** for partial page updates
- **Streaming templates** for large document lists
- **Client-side template caching** for repeated views

This template system provides a solid foundation for professional academic document digitization with sophisticated Japanese text support, comprehensive accessibility, and room for future enhancements while maintaining clean, maintainable code.

