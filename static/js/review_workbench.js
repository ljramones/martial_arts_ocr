(function () {
    const shell = document.querySelector(".review-shell");
    if (!shell) return;

    const regionTypes = JSON.parse(shell.dataset.regionTypes || "[]");
    const state = {
        project: null,
        page: null,
        selectedRegionId: null,
        drag: null,
        imageLayout: null,
    };

    const els = {
        sourceFolder: document.getElementById("review-source-folder"),
        projectId: document.getElementById("review-project-id"),
        createProject: document.getElementById("review-create-project"),
        loadProject: document.getElementById("review-load-project"),
        status: document.getElementById("review-status"),
        pageList: document.getElementById("review-page-list"),
        regionList: document.getElementById("review-region-list"),
        recognize: document.getElementById("review-recognize-page"),
        addRegion: document.getElementById("review-add-region"),
        stage: document.getElementById("review-page-stage"),
        image: document.getElementById("review-page-image"),
        overlay: document.getElementById("review-overlay"),
        emptyState: document.getElementById("review-empty-state"),
        selectedId: document.getElementById("review-selected-region-id"),
        type: document.getElementById("review-region-type"),
        bboxFields: document.getElementById("review-bbox-fields"),
        bboxX: document.getElementById("review-bbox-x"),
        bboxY: document.getElementById("review-bbox-y"),
        bboxW: document.getElementById("review-bbox-w"),
        bboxH: document.getElementById("review-bbox-h"),
        notes: document.getElementById("review-region-notes"),
        save: document.getElementById("review-save-region"),
        ignore: document.getElementById("review-ignore-region"),
        delete: document.getElementById("review-delete-region"),
        detectedType: document.getElementById("review-detected-type"),
        effectiveType: document.getElementById("review-effective-type"),
        detectedBbox: document.getElementById("review-detected-bbox"),
        effectiveBbox: document.getElementById("review-effective-bbox"),
        regionStatus: document.getElementById("review-region-status"),
        regionSource: document.getElementById("review-region-source"),
        regionDetector: document.getElementById("review-region-detector"),
        regionConfidence: document.getElementById("review-region-confidence"),
        regionMixed: document.getElementById("review-region-mixed"),
        regionNeedsReview: document.getElementById("review-region-needs-review"),
        regionLayoutFusion: document.getElementById("review-region-layout-fusion"),
        regionRole: document.getElementById("review-region-role"),
    };

    regionTypes.forEach((regionType) => {
        const option = document.createElement("option");
        option.value = regionType;
        option.textContent = regionType;
        els.type.appendChild(option);
    });

    els.createProject.addEventListener("click", createOrLoadProject);
    els.loadProject.addEventListener("click", loadProjectById);
    els.recognize.addEventListener("click", recognizePage);
    els.addRegion.addEventListener("click", addManualRegion);
    els.save.addEventListener("click", saveSelectedRegion);
    els.ignore.addEventListener("click", ignoreSelectedRegion);
    els.delete.addEventListener("click", deleteSelectedRegion);
    [els.type, els.bboxX, els.bboxY, els.bboxW, els.bboxH, els.notes].forEach((element) => {
        element.addEventListener("input", updateSelectedFromPanel);
    });

    window.addEventListener("mousemove", onDragMove);
    window.addEventListener("mouseup", () => {
        state.drag = null;
    });
    window.addEventListener("resize", () => {
        syncOverlaySize();
        renderOverlay();
    });

    async function createOrLoadProject() {
        const payload = {};
        if (els.sourceFolder.value.trim()) payload.source_folder = els.sourceFolder.value.trim();
        if (els.projectId.value.trim()) payload.project_id = els.projectId.value.trim();
        const project = await requestJson("/api/review/projects", {
            method: "POST",
            body: JSON.stringify(payload),
        });
        setProject(project);
    }

    async function loadProjectById() {
        const projectId = els.projectId.value.trim();
        if (!projectId) {
            setStatus("Enter a project ID to load.");
            return;
        }
        const project = await requestJson(`/api/review/projects/${encodeURIComponent(projectId)}`);
        setProject(project);
    }

    function setProject(project) {
        state.project = project;
        state.page = null;
        state.selectedRegionId = null;
        els.projectId.value = project.project_id || "";
        setStatus(`Loaded ${project.project_id}`);
        renderPageList();
        renderRegionList();
        renderSelectedRegion();
        clearViewer();
    }

    function renderPageList() {
        els.pageList.innerHTML = "";
        (state.project?.pages || []).forEach((page) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = state.page?.page_id === page.page_id ? "active" : "";
            button.innerHTML = `<strong>${escapeHtml(page.page_id)}</strong><span class="secondary">${escapeHtml(page.filename || page.source_path)}</span>`;
            button.addEventListener("click", () => loadPage(page.page_id));
            els.pageList.appendChild(button);
        });
    }

    async function loadPage(pageId) {
        if (!state.project) return;
        const page = await requestJson(`/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(pageId)}`);
        state.page = page;
        state.selectedRegionId = null;
        els.stage.classList.remove("is-empty");
        els.emptyState.hidden = true;
        els.image.hidden = false;
        els.overlay.hidden = false;
        els.image.src = `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(pageId)}/image?ts=${Date.now()}`;
        els.image.onload = () => {
            syncOverlaySize();
            renderOverlay();
        };
        syncOverlaySize();
        renderPageList();
        renderRegionList();
        renderOverlay();
        renderSelectedRegion();
        els.addRegion.disabled = false;
        els.recognize.disabled = false;
    }

    function syncOverlaySize() {
        if (!state.page || !els.image.complete || !els.image.clientWidth || !els.image.clientHeight) return;
        state.imageLayout = computeImageLayout({
            imageElement: els.image,
            imageWidth: state.page.width || els.image.naturalWidth || 1,
            imageHeight: state.page.height || els.image.naturalHeight || 1,
        });
        els.overlay.style.left = `${state.imageLayout.offsetX}px`;
        els.overlay.style.top = `${state.imageLayout.offsetY}px`;
        els.overlay.style.width = `${state.imageLayout.renderedWidth}px`;
        els.overlay.style.height = `${state.imageLayout.renderedHeight}px`;
        els.overlay.setAttribute("viewBox", `0 0 ${state.imageLayout.renderedWidth || 1} ${state.imageLayout.renderedHeight || 1}`);
        els.overlay.setAttribute("preserveAspectRatio", "none");
    }

    function clearViewer() {
        els.stage.classList.add("is-empty");
        els.image.hidden = true;
        els.overlay.hidden = true;
        els.emptyState.hidden = false;
        els.image.removeAttribute("src");
        els.overlay.innerHTML = "";
        state.imageLayout = null;
        els.addRegion.disabled = true;
        els.recognize.disabled = true;
    }

    function renderRegionList() {
        els.regionList.innerHTML = "";
        (state.page?.regions || []).forEach((region) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = state.selectedRegionId === region.region_id ? "active" : "";
            button.innerHTML = `<strong>${escapeHtml(region.region_id)}</strong><span class="secondary">${escapeHtml(region.effective_type || "unknown")} · ${escapeHtml(region.status || "")}</span>`;
            button.addEventListener("click", () => selectRegion(region.region_id));
            els.regionList.appendChild(button);
        });
    }

    function renderOverlay() {
        els.overlay.innerHTML = "";
        if (!state.page) return;
        syncOverlaySize();
        const layout = state.imageLayout;
        if (!layout) return;
        for (const region of state.page.regions || []) {
            const detected = region.detected_bbox;
            const bbox = region.effective_bbox;
            if (detected && JSON.stringify(detected) !== JSON.stringify(bbox)) {
                els.overlay.appendChild(svgRect(imageBBoxToScreenBBox(layout, detected), "review-detected-rect"));
            }
            if (!bbox) continue;
            const rect = svgRect(
                imageBBoxToScreenBBox(layout, bbox),
                [
                    "review-region-rect",
                    region.region_id === state.selectedRegionId ? "selected" : "",
                    region.ignored ? "ignored" : "",
                ].join(" ")
            );
            rect.dataset.regionId = region.region_id;
            rect.addEventListener("mousedown", (event) => beginDrag(event, region.region_id, "move"));
            rect.addEventListener("click", (event) => {
                event.stopPropagation();
                selectRegion(region.region_id);
            });
            els.overlay.appendChild(rect);
            if (region.region_id === state.selectedRegionId) {
                for (const handle of handlesForBBox(imageBBoxToScreenBBox(layout, bbox))) {
                    const circle = svgHandle(handle.x, handle.y, handle.name);
                    circle.addEventListener("mousedown", (event) => beginDrag(event, region.region_id, handle.name));
                    els.overlay.appendChild(circle);
                }
            }
        }
    }

    function svgRect(bbox, className) {
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", bbox[0]);
        rect.setAttribute("y", bbox[1]);
        rect.setAttribute("width", bbox[2]);
        rect.setAttribute("height", bbox[3]);
        rect.setAttribute("class", className);
        return rect;
    }

    function svgHandle(x, y, name) {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", x);
        circle.setAttribute("cy", y);
        circle.setAttribute("r", 7);
        circle.setAttribute("class", "review-handle");
        circle.dataset.handle = name;
        return circle;
    }

    function handlesForBBox([x, y, w, h]) {
        return [
            { name: "nw", x, y },
            { name: "ne", x: x + w, y },
            { name: "sw", x, y: y + h },
            { name: "se", x: x + w, y: y + h },
        ];
    }

    function selectRegion(regionId) {
        state.selectedRegionId = regionId;
        renderRegionList();
        renderOverlay();
        renderSelectedRegion();
    }

    function selectedRegion() {
        return (state.page?.regions || []).find((region) => region.region_id === state.selectedRegionId) || null;
    }

    function renderSelectedRegion() {
        const region = selectedRegion();
        const enabled = Boolean(region);
        [els.type, els.bboxX, els.bboxY, els.bboxW, els.bboxH, els.notes, els.save, els.ignore, els.delete].forEach((element) => {
            element.disabled = !enabled;
        });
        els.bboxFields.disabled = !enabled;
        if (!region) {
            els.selectedId.textContent = "No region selected.";
            els.detectedType.textContent = "-";
            els.effectiveType.textContent = "-";
            els.detectedBbox.textContent = "-";
            els.effectiveBbox.textContent = "-";
            els.regionStatus.textContent = "-";
            els.regionSource.textContent = "-";
            renderRegionMetadata(null);
            return;
        }
        const bbox = region.reviewed_bbox || region.effective_bbox || [0, 0, 1, 1];
        els.selectedId.textContent = region.region_id;
        els.type.value = region.reviewed_type || region.effective_type || "unknown_needs_review";
        [els.bboxX.value, els.bboxY.value, els.bboxW.value, els.bboxH.value] = bbox;
        els.notes.value = region.notes || "";
        els.detectedType.textContent = region.detected_type || "-";
        els.effectiveType.textContent = region.effective_type || "-";
        els.detectedBbox.textContent = region.detected_bbox ? JSON.stringify(region.detected_bbox) : "-";
        els.effectiveBbox.textContent = region.effective_bbox ? JSON.stringify(region.effective_bbox) : "-";
        els.regionStatus.textContent = region.status || "-";
        els.regionSource.textContent = region.source || "-";
        renderRegionMetadata(region);
    }

    function renderRegionMetadata(region) {
        const metadata = region?.metadata || {};
        els.regionDetector.textContent = metadata.detector || "-";
        els.regionConfidence.textContent = formatValue(region?.confidence ?? metadata.confidence);
        els.regionMixed.textContent = formatValue(metadata.mixed_region);
        els.regionNeedsReview.textContent = formatValue(region?.needs_review ?? metadata.needs_review);
        els.regionLayoutFusion.textContent = formatValue(metadata.layout_fusion_applied ?? metadata.paddle_layout_fusion);
        els.regionRole.textContent = metadata.region_role || metadata.role || "-";
    }

    function updateSelectedFromPanel() {
        const region = selectedRegion();
        if (!region) return;
        const bbox = coerceBBox([
            Number(els.bboxX.value),
            Number(els.bboxY.value),
            Number(els.bboxW.value),
            Number(els.bboxH.value),
        ]);
        region.reviewed_type = els.type.value;
        region.reviewed_bbox = bbox;
        region.effective_type = region.reviewed_type || region.detected_type || "unknown_needs_review";
        region.effective_bbox = region.reviewed_bbox || region.detected_bbox;
        region.notes = els.notes.value;
        region.status = region.reviewed_type === "ignore" ? "ignored" : "reviewed";
        region.ignored = region.reviewed_type === "ignore";
        renderRegionList();
        renderOverlay();
        renderSelectedRegion();
    }

    async function addManualRegion() {
        if (!state.project || !state.page) return;
        const region = await requestJson(
            `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/regions`,
            {
                method: "POST",
                body: JSON.stringify({ reviewed_type: "unknown_needs_review" }),
            }
        );
        state.page = region.page;
        state.selectedRegionId = region.region.region_id;
        renderRegionList();
        renderOverlay();
        renderSelectedRegion();
    }

    async function recognizePage() {
        if (!state.project || !state.page) return;
        els.recognize.disabled = true;
        setStatus("Running recognition for selected page...");
        try {
            const result = await requestJson(
                `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/recognize`,
                { method: "POST" }
            );
            state.page = result.page;
            state.selectedRegionId = null;
            renderRegionList();
            renderOverlay();
            renderSelectedRegion();
            const rejected = result.rejected_count || 0;
            const suffix = rejected ? ` ${rejected} text-like candidate(s) rejected.` : "";
            setStatus(`Recognition imported ${result.detected_count || 0} region(s).${suffix}`);
        } catch (error) {
            setStatus(error.message || "Recognition failed.");
        } finally {
            els.recognize.disabled = !state.page;
        }
    }

    async function saveSelectedRegion() {
        updateSelectedFromPanel();
        const region = selectedRegion();
        if (!state.project || !state.page || !region) return;
        const payload = {
            reviewed_type: region.reviewed_type,
            reviewed_bbox: region.reviewed_bbox,
            notes: region.notes || "",
            status: region.status || "reviewed",
        };
        const result = await requestJson(
            `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/regions/${encodeURIComponent(region.region_id)}`,
            {
                method: "PATCH",
                body: JSON.stringify(payload),
            }
        );
        state.page = result.page;
        setStatus(`Saved ${region.region_id}`);
        renderRegionList();
        renderOverlay();
        renderSelectedRegion();
    }

    async function ignoreSelectedRegion() {
        const region = selectedRegion();
        if (!region) return;
        els.type.value = "ignore";
        updateSelectedFromPanel();
        await saveSelectedRegion();
    }

    async function deleteSelectedRegion() {
        const region = selectedRegion();
        if (!state.project || !state.page || !region) return;
        const result = await requestJson(
            `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/regions/${encodeURIComponent(region.region_id)}`,
            { method: "DELETE" }
        );
        state.page = result.page;
        state.selectedRegionId = null;
        renderRegionList();
        renderOverlay();
        renderSelectedRegion();
        setStatus(`Deleted ${region.region_id}`);
    }

    function beginDrag(event, regionId, mode) {
        event.preventDefault();
        event.stopPropagation();
        selectRegion(regionId);
        const region = selectedRegion();
        const point = svgPoint(event);
        const layout = state.imageLayout;
        if (!layout) return;
        state.drag = {
            mode,
            start: point,
            screenBBox: imageBBoxToScreenBBox(layout, region.reviewed_bbox || region.effective_bbox),
        };
    }

    function onDragMove(event) {
        if (!state.drag) return;
        const region = selectedRegion();
        const layout = state.imageLayout;
        if (!region || !layout) return;
        const point = svgPoint(event);
        const dx = point.x - state.drag.start.x;
        const dy = point.y - state.drag.start.y;
        let [x, y, w, h] = state.drag.screenBBox;
        if (state.drag.mode === "move") {
            x += dx;
            y += dy;
        } else {
            if (state.drag.mode.includes("n")) {
                y += dy;
                h -= dy;
            }
            if (state.drag.mode.includes("s")) h += dy;
            if (state.drag.mode.includes("w")) {
                x += dx;
                w -= dx;
            }
            if (state.drag.mode.includes("e")) w += dx;
        }
        region.reviewed_bbox = screenBBoxToImageBBox(layout, [x, y, w, h]);
        region.effective_bbox = region.reviewed_bbox;
        region.status = region.status === "manual" ? "manual" : "reviewed";
        [els.bboxX.value, els.bboxY.value, els.bboxW.value, els.bboxH.value] = region.reviewed_bbox;
        renderOverlay();
        renderSelectedRegion();
    }

    function svgPoint(event) {
        const point = els.overlay.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        return point.matrixTransform(els.overlay.getScreenCTM().inverse());
    }

    function coerceBBox([x, y, w, h]) {
        const pageWidth = Number(state.page?.width || 1);
        const pageHeight = Number(state.page?.height || 1);
        x = Math.max(0, Math.min(Math.round(x || 0), pageWidth - 1));
        y = Math.max(0, Math.min(Math.round(y || 0), pageHeight - 1));
        w = Math.max(1, Math.min(Math.round(w || 1), pageWidth - x));
        h = Math.max(1, Math.min(Math.round(h || 1), pageHeight - y));
        return [x, y, w, h];
    }

    function computeImageLayout({ imageElement, imageWidth, imageHeight }) {
        const renderedWidth = imageElement.clientWidth || 1;
        const renderedHeight = imageElement.clientHeight || 1;
        return {
            imageWidth: Math.max(1, Number(imageWidth || 1)),
            imageHeight: Math.max(1, Number(imageHeight || 1)),
            renderedWidth,
            renderedHeight,
            offsetX: imageElement.offsetLeft || 0,
            offsetY: imageElement.offsetTop || 0,
            scaleX: renderedWidth / Math.max(1, Number(imageWidth || 1)),
            scaleY: renderedHeight / Math.max(1, Number(imageHeight || 1)),
        };
    }

    function imageBBoxToScreenBBox(layout, bbox) {
        return [
            bbox[0] * layout.scaleX,
            bbox[1] * layout.scaleY,
            bbox[2] * layout.scaleX,
            bbox[3] * layout.scaleY,
        ];
    }

    function screenBBoxToImageBBox(layout, bbox) {
        return clampImageBBox(layout, [
            bbox[0] / layout.scaleX,
            bbox[1] / layout.scaleY,
            bbox[2] / layout.scaleX,
            bbox[3] / layout.scaleY,
        ]);
    }

    function clampImageBBox(layout, [x, y, w, h]) {
        x = Math.max(0, Math.min(Math.round(x || 0), layout.imageWidth - 1));
        y = Math.max(0, Math.min(Math.round(y || 0), layout.imageHeight - 1));
        w = Math.max(1, Math.min(Math.round(w || 1), layout.imageWidth - x));
        h = Math.max(1, Math.min(Math.round(h || 1), layout.imageHeight - y));
        return [x, y, w, h];
    }

    async function requestJson(url, options = {}) {
        const response = await fetch(url, {
            headers: { "Content-Type": "application/json" },
            ...options,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.error || `Request failed: ${response.status}`);
        }
        return payload;
    }

    function setStatus(message) {
        els.status.textContent = message;
    }

    function formatValue(value) {
        if (value === null || value === undefined || value === "") return "-";
        if (typeof value === "object") return JSON.stringify(value);
        return String(value);
    }

    function escapeHtml(value) {
        return String(value || "").replace(/[&<>"']/g, (char) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
        }[char]));
    }
}());
