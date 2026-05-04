(function () {
    const shell = document.querySelector(".review-shell");
    if (!shell) return;

    const regionTypes = JSON.parse(shell.dataset.regionTypes || "[]");
    const state = {
        project: null,
        page: null,
        selectedRegionId: null,
        drag: null,
        draw: null,
        tool: "select",
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
        regionInventory: document.getElementById("review-region-inventory"),
        toolButtons: Array.from(document.querySelectorAll("[data-review-tool]")),
        quickTypeButtons: Array.from(document.querySelectorAll("[data-review-type]")),
        detectOrientation: document.getElementById("review-detect-orientation"),
        orientationDetected: document.getElementById("review-orientation-detected"),
        orientationConfidence: document.getElementById("review-orientation-confidence"),
        orientationEffective: document.getElementById("review-orientation-effective"),
        orientationStatus: document.getElementById("review-orientation-status"),
        orientationOverride: document.getElementById("review-orientation-override"),
        saveOrientation: document.getElementById("review-save-orientation"),
        orientationWarning: document.getElementById("review-orientation-warning"),
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
        duplicate: document.getElementById("review-duplicate-region"),
        duplicateLeft: document.getElementById("review-duplicate-left"),
        duplicateRight: document.getElementById("review-duplicate-right"),
        duplicateUp: document.getElementById("review-duplicate-up"),
        duplicateDown: document.getElementById("review-duplicate-down"),
        nudgeLeft: document.getElementById("review-nudge-left"),
        nudgeRight: document.getElementById("review-nudge-right"),
        nudgeUp: document.getElementById("review-nudge-up"),
        nudgeDown: document.getElementById("review-nudge-down"),
        runRegionOcr: document.getElementById("review-run-region-ocr"),
        ocrStatus: document.getElementById("review-ocr-status"),
        ocrRoute: document.getElementById("review-ocr-route"),
        ocrConfidence: document.getElementById("review-ocr-confidence"),
        ocrOutput: document.getElementById("review-ocr-output"),
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
        diagRaw: document.getElementById("review-diag-raw"),
        diagAccepted: document.getElementById("review-diag-accepted"),
        diagRejected: document.getElementById("review-diag-rejected"),
        diagSuppressed: document.getElementById("review-diag-suppressed"),
        diagMerged: document.getElementById("review-diag-merged"),
        diagImported: document.getElementById("review-diag-imported"),
        diagnosticsList: document.getElementById("review-diagnostics-list"),
    };

    regionTypes.forEach((regionType) => {
        const option = document.createElement("option");
        option.value = regionType;
        option.textContent = regionType;
        els.type.appendChild(option);
    });

    els.createProject.addEventListener("click", createOrLoadProject);
    els.loadProject.addEventListener("click", loadProjectById);
    els.detectOrientation.addEventListener("click", detectOrientation);
    els.saveOrientation.addEventListener("click", saveOrientationOverride);
    els.recognize.addEventListener("click", recognizePage);
    els.addRegion.addEventListener("click", addManualRegion);
    els.save.addEventListener("click", saveSelectedRegion);
    els.ignore.addEventListener("click", ignoreSelectedRegion);
    els.delete.addEventListener("click", deleteSelectedRegion);
    els.toolButtons.forEach((button) => {
        button.addEventListener("click", () => setTool(button.dataset.reviewTool || "select"));
    });
    els.quickTypeButtons.forEach((button) => {
        button.addEventListener("click", () => setSelectedRegionType(button.dataset.reviewType));
    });
    els.duplicate.addEventListener("click", () => duplicateSelectedRegion("same"));
    els.duplicateLeft.addEventListener("click", () => duplicateSelectedRegion("left"));
    els.duplicateRight.addEventListener("click", () => duplicateSelectedRegion("right"));
    els.duplicateUp.addEventListener("click", () => duplicateSelectedRegion("up"));
    els.duplicateDown.addEventListener("click", () => duplicateSelectedRegion("down"));
    els.nudgeLeft.addEventListener("click", () => nudgeSelectedRegion(-10, 0));
    els.nudgeRight.addEventListener("click", () => nudgeSelectedRegion(10, 0));
    els.nudgeUp.addEventListener("click", () => nudgeSelectedRegion(0, -10));
    els.nudgeDown.addEventListener("click", () => nudgeSelectedRegion(0, 10));
    els.runRegionOcr.addEventListener("click", runSelectedRegionOcr);
    [els.type, els.bboxX, els.bboxY, els.bboxW, els.bboxH, els.notes].forEach((element) => {
        element.addEventListener("input", updateSelectedFromPanel);
    });
    window.addEventListener("keydown", onKeyDown);

    els.overlay.addEventListener("mousedown", beginDraw);
    window.addEventListener("mousemove", onPointerMove);
    window.addEventListener("mouseup", onPointerUp);
    window.addEventListener("mouseleave", () => {
        state.drag = null;
        state.draw = null;
        renderOverlay();
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
        renderOrientation();
        renderRecognitionDiagnostics();
        renderSelectedRegion();
        clearViewer();
    }

    function setTool(tool) {
        state.tool = tool || "select";
        state.draw = null;
        els.toolButtons.forEach((button) => {
            button.classList.toggle("active", button.dataset.reviewTool === state.tool);
        });
        els.overlay.classList.toggle("is-drawing", state.tool.startsWith("draw_"));
        renderOverlay();
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
        renderOrientation();
        renderRegionList();
        renderRecognitionDiagnostics();
        renderOverlay();
        renderSelectedRegion();
        els.addRegion.disabled = false;
        els.detectOrientation.disabled = false;
        els.orientationOverride.disabled = false;
        els.saveOrientation.disabled = false;
        els.recognize.disabled = false;
    }

    function syncOverlaySize() {
        if (!state.page || !els.image.complete || !els.image.clientWidth || !els.image.clientHeight) return;
        state.imageLayout = computeImageLayout({
            imageElement: els.image,
            imageWidth: state.page.effective_width || state.page.width || els.image.naturalWidth || 1,
            imageHeight: state.page.effective_height || state.page.height || els.image.naturalHeight || 1,
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
        els.detectOrientation.disabled = true;
        els.orientationOverride.disabled = true;
        els.saveOrientation.disabled = true;
        els.recognize.disabled = true;
        renderOrientation();
        renderRecognitionDiagnostics();
    }

    function reloadPageImage() {
        if (!state.project || !state.page) return;
        els.image.src = `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/image?ts=${Date.now()}`;
        els.image.onload = () => {
            syncOverlaySize();
            renderOverlay();
        };
    }

    function renderOrientation() {
        const orientation = state.page?.orientation || {};
        const detected = orientation.detected_rotation_degrees ?? 0;
        const effective = orientation.effective_rotation_degrees ?? 0;
        els.orientationDetected.textContent = `${detected}°`;
        els.orientationConfidence.textContent = formatConfidence(orientation.detected_confidence);
        els.orientationEffective.textContent = `${effective}°`;
        els.orientationStatus.textContent = orientation.status || "-";
        els.orientationOverride.value = String(orientation.reviewed_rotation_degrees ?? effective);
        const stale = Boolean(state.page?.regions_stale);
        els.orientationWarning.hidden = !stale;
        els.orientationWarning.textContent = stale
            ? "Orientation changed after regions were created. Rerun recognition or review all boxes."
            : "";
    }

    async function detectOrientation() {
        if (!state.project || !state.page) return;
        els.detectOrientation.disabled = true;
        setStatus("Running page orientation detection...");
        try {
            const result = await requestJson(
                `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/orientation/detect`,
                { method: "POST" }
            );
            state.page = result.page;
            state.selectedRegionId = null;
            reloadPageImage();
            renderOrientation();
            renderRegionList();
            renderRecognitionDiagnostics();
            renderOverlay();
            renderSelectedRegion();
            setStatus(`Orientation effective ${state.page.orientation?.effective_rotation_degrees ?? 0}° (${state.page.orientation?.status || "unknown"}).`);
        } catch (error) {
            setStatus(error.message || "Orientation detection failed.");
        } finally {
            els.detectOrientation.disabled = !state.page;
        }
    }

    async function saveOrientationOverride() {
        if (!state.project || !state.page) return;
        setStatus("Saving reviewed orientation...");
        const rotation = Number(els.orientationOverride.value);
        try {
            const result = await requestJson(
                `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/orientation`,
                {
                    method: "PATCH",
                    body: JSON.stringify({ reviewed_rotation_degrees: rotation }),
                }
            );
            state.page = result.page;
            state.selectedRegionId = null;
            reloadPageImage();
            renderOrientation();
            renderRegionList();
            renderRecognitionDiagnostics();
            renderOverlay();
            renderSelectedRegion();
            setStatus(`Orientation override saved: ${rotation}°.`);
        } catch (error) {
            setStatus(error.message || "Orientation override failed.");
        }
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
        renderRegionInventory();
    }

    function renderRegionInventory() {
        if (!els.regionInventory) return;
        els.regionInventory.innerHTML = "";
        const regions = state.page?.regions || [];
        if (!regions.length) {
            els.regionInventory.textContent = "No regions marked yet.";
            return;
        }

        const groups = [
            ["Image Regions", (region) => ["image", "diagram", "photo"].includes(region.effective_type)],
            ["Text Regions", (region) => ["english_text", "romanized_japanese_text", "caption_label"].includes(region.effective_type)],
            ["Japanese Regions", (region) => ["modern_japanese_horizontal", "modern_japanese_vertical", "mixed_english_japanese"].includes(region.effective_type)],
            ["Ignored Regions", (region) => region.ignored || region.effective_type === "ignore" || region.status === "ignored"],
            ["Other Regions", (region) => true],
        ];
        const claimed = new Set();
        for (const [title, predicate] of groups) {
            const groupRegions = regions.filter((region) => !claimed.has(region.region_id) && predicate(region));
            if (!groupRegions.length) continue;
            groupRegions.forEach((region) => claimed.add(region.region_id));
            const section = document.createElement("div");
            section.className = "review-inventory-group";
            section.innerHTML = `<h4>${escapeHtml(title)} <span>${groupRegions.length}</span></h4>`;
            for (const region of groupRegions) {
                const row = document.createElement("button");
                row.type = "button";
                row.className = state.selectedRegionId === region.region_id ? "review-inventory-row active" : "review-inventory-row";
                row.innerHTML = [
                    `<strong>${escapeHtml(region.effective_type || "unknown")}</strong>`,
                    `<span>${escapeHtml(region.review_status || region.status || "-")} · ${escapeHtml(region.source || "-")}</span>`,
                    `<code>${escapeHtml(JSON.stringify(region.effective_bbox || []))}</code>`,
                ].join("");
                row.addEventListener("click", () => selectRegion(region.region_id));
                section.appendChild(row);
            }
            els.regionInventory.appendChild(section);
        }
    }

    function renderRecognitionDiagnostics() {
        const diagnostics = state.page?.recognition_diagnostics || {};
        els.diagRaw.textContent = formatValue(diagnostics.raw_candidate_count);
        els.diagAccepted.textContent = formatValue(diagnostics.accepted_count);
        els.diagRejected.textContent = formatValue(diagnostics.rejected_count);
        els.diagSuppressed.textContent = formatValue(diagnostics.suppressed_count);
        els.diagMerged.textContent = formatValue(diagnostics.merged_count);
        els.diagImported.textContent = formatValue(diagnostics.imported_count);
        els.diagnosticsList.innerHTML = "";

        const candidates = diagnostics.candidates || [];
        if (!candidates.length) {
            els.diagnosticsList.textContent = "No recognition diagnostics yet.";
            return;
        }

        for (const candidate of candidates.slice(0, 80)) {
            const row = document.createElement("div");
            row.className = `review-diagnostic-row stage-${escapeCssClass(candidate.stage || "unknown")}`;
            row.innerHTML = [
                `<strong>${escapeHtml(candidate.candidate_id || "")}</strong>`,
                `<span>${escapeHtml(candidate.stage || "-")}</span>`,
                `<span>${escapeHtml(candidate.reason || "-")}</span>`,
                `<span>${escapeHtml(candidate.region_type || "-")}</span>`,
                `<code>${escapeHtml(JSON.stringify(candidate.bbox || []))}</code>`,
            ].join("");
            els.diagnosticsList.appendChild(row);
        }

        if (candidates.length > 80) {
            const more = document.createElement("p");
            more.className = "review-muted";
            more.textContent = `Showing first 80 of ${candidates.length} diagnostic candidates.`;
            els.diagnosticsList.appendChild(more);
        }
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
        if (state.draw) {
            els.overlay.appendChild(svgRect(drawScreenBBox(), "review-draft-rect"));
        }
    }

    function beginDraw(event) {
        if (!state.page || !state.tool.startsWith("draw_") || event.target !== els.overlay) return;
        event.preventDefault();
        const point = svgPoint(event);
        state.selectedRegionId = null;
        state.draw = {
            start: point,
            current: point,
            reviewedType: regionTypeForTool(state.tool),
        };
        renderRegionList();
        renderOverlay();
        renderSelectedRegion();
    }

    function onDrawMove(event) {
        if (!state.draw) return;
        state.draw.current = svgPoint(event);
        renderOverlay();
    }

    async function finishDraw() {
        if (!state.draw || !state.project || !state.page || !state.imageLayout) return;
        const bbox = screenBBoxToImageBBox(state.imageLayout, drawScreenBBox());
        const draw = state.draw;
        state.draw = null;
        if (bbox[2] < 6 || bbox[3] < 6) {
            renderOverlay();
            return;
        }
        try {
            const result = await requestJson(
                `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/regions`,
                {
                    method: "POST",
                    body: JSON.stringify({
                        reviewed_type: draw.reviewedType,
                        reviewed_bbox: bbox,
                        status: "reviewed",
                        review_status: "manually_added",
                    }),
                }
            );
            state.page = result.page;
            state.selectedRegionId = result.region.region_id;
            setTool("select");
            renderRegionList();
            renderRecognitionDiagnostics();
            renderOverlay();
            renderSelectedRegion();
            setStatus(`Added ${result.region.region_id} as ${draw.reviewedType}.`);
        } catch (error) {
            setStatus(error.message || "Drawing region failed.");
            renderOverlay();
        }
    }

    function drawScreenBBox() {
        const start = state.draw?.start || { x: 0, y: 0 };
        const current = state.draw?.current || start;
        const x = Math.min(start.x, current.x);
        const y = Math.min(start.y, current.y);
        const w = Math.abs(current.x - start.x);
        const h = Math.abs(current.y - start.y);
        return [x, y, w, h];
    }

    function regionTypeForTool(tool) {
        if (tool === "draw_image") return "image";
        if (tool === "draw_text") return "english_text";
        if (tool === "draw_japanese") return "modern_japanese_horizontal";
        return "unknown_needs_review";
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
        [
            els.type,
            els.bboxX,
            els.bboxY,
            els.bboxW,
            els.bboxH,
            els.notes,
            els.save,
            els.ignore,
            els.delete,
            els.duplicate,
            els.duplicateLeft,
            els.duplicateRight,
            els.duplicateUp,
            els.duplicateDown,
            els.nudgeLeft,
            els.nudgeRight,
            els.nudgeUp,
            els.nudgeDown,
            els.runRegionOcr,
            ...els.quickTypeButtons,
        ].forEach((element) => {
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
            els.quickTypeButtons.forEach((button) => button.classList.remove("active"));
            renderRegionMetadata(null);
            renderRegionOcr(null);
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
        els.quickTypeButtons.forEach((button) => {
            button.classList.toggle("active", button.dataset.reviewType === els.type.value);
        });
        renderRegionMetadata(region);
        renderRegionOcr(region);
    }

    function renderRegionOcr(region) {
        const attempts = state.page?.ocr_attempts || [];
        const attemptId = region?.last_ocr_attempt_id;
        const attempt = attempts.find((item) => item.attempt_id === attemptId)
            || [...attempts].reverse().find((item) => item.region_id === region?.region_id);
        if (!region || !attempt) {
            els.ocrStatus.textContent = region ? "Not run" : "-";
            els.ocrRoute.textContent = "-";
            els.ocrConfidence.textContent = "-";
            els.ocrOutput.value = "";
            return;
        }
        const route = attempt.route || {};
        els.ocrStatus.textContent = attempt.status || "-";
        els.ocrRoute.textContent = [
            route.engine || "tesseract",
            route.language || "none",
            route.psm ? `PSM ${route.psm}` : null,
            route.preprocess_profile || null,
        ].filter(Boolean).join(" / ");
        els.ocrConfidence.textContent = formatConfidence(attempt.confidence);
        els.ocrOutput.value = attempt.cleaned_text || attempt.text || "";
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
        region.review_status = region.reviewed_type === "ignore"
            ? (region.source === "machine_detection" ? "rejected" : "ignored")
            : (region.review_status || (region.source === "machine_detection" ? "accepted" : "manually_added"));
        renderRegionList();
        renderOverlay();
        renderSelectedRegion();
    }

    function setSelectedRegionType(regionType) {
        if (!selectedRegion() || !regionType) return;
        els.type.value = regionType;
        updateSelectedFromPanel();
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
        renderRecognitionDiagnostics();
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
            renderRecognitionDiagnostics();
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
        renderRecognitionDiagnostics();
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
        renderRecognitionDiagnostics();
        renderOverlay();
        renderSelectedRegion();
        setStatus(`Deleted ${region.region_id}`);
    }

    async function duplicateSelectedRegion(direction) {
        const region = selectedRegion();
        if (!state.project || !state.page || !region) return;
        const result = await requestJson(
            `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/regions/${encodeURIComponent(region.region_id)}/duplicate`,
            {
                method: "POST",
                body: JSON.stringify({ direction }),
            }
        );
        state.page = result.page;
        state.selectedRegionId = result.region.region_id;
        renderRegionList();
        renderRecognitionDiagnostics();
        renderOverlay();
        renderSelectedRegion();
        setStatus(`Duplicated ${region.region_id} ${direction}.`);
    }

    async function runSelectedRegionOcr() {
        const region = selectedRegion();
        if (!state.project || !state.page || !region) return;
        els.runRegionOcr.disabled = true;
        setStatus(`Running OCR for ${region.region_id}...`);
        try {
            const result = await requestJson(
                `/api/review/projects/${encodeURIComponent(state.project.project_id)}/pages/${encodeURIComponent(state.page.page_id)}/regions/${encodeURIComponent(region.region_id)}/ocr`,
                { method: "POST" }
            );
            state.page = result.page;
            state.selectedRegionId = region.region_id;
            renderRegionList();
            renderRecognitionDiagnostics();
            renderOverlay();
            renderSelectedRegion();
            setStatus(`OCR ${result.attempt.status} for ${region.region_id}.`);
        } catch (error) {
            setStatus(error.message || "Region OCR failed.");
        } finally {
            els.runRegionOcr.disabled = !selectedRegion();
        }
    }

    function nudgeSelectedRegion(dx, dy) {
        const region = selectedRegion();
        if (!region) return;
        const bbox = region.reviewed_bbox || region.effective_bbox || [0, 0, 1, 1];
        region.reviewed_bbox = coerceBBox([bbox[0] + dx, bbox[1] + dy, bbox[2], bbox[3]]);
        region.effective_bbox = region.reviewed_bbox;
        region.reviewed_type = region.reviewed_type || region.effective_type || "unknown_needs_review";
        region.effective_type = region.reviewed_type;
        region.status = region.reviewed_type === "ignore" ? "ignored" : "reviewed";
        region.ignored = region.reviewed_type === "ignore";
        region.review_status = region.source === "machine_detection" ? "resized" : "manually_added";
        [els.bboxX.value, els.bboxY.value, els.bboxW.value, els.bboxH.value] = region.reviewed_bbox;
        renderRegionList();
        renderOverlay();
        renderSelectedRegion();
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
        region.review_status = region.source === "machine_detection" ? "resized" : "manually_added";
        [els.bboxX.value, els.bboxY.value, els.bboxW.value, els.bboxH.value] = region.reviewed_bbox;
        renderOverlay();
        renderSelectedRegion();
    }

    function onPointerMove(event) {
        onDragMove(event);
        onDrawMove(event);
    }

    async function onPointerUp() {
        if (state.draw) {
            await finishDraw();
        }
        state.drag = null;
    }

    function onKeyDown(event) {
        if (!selectedRegion()) return;
        if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target?.tagName)) return;
        const step = event.shiftKey ? 10 : 1;
        if (event.key === "ArrowLeft") {
            event.preventDefault();
            nudgeSelectedRegion(-step, 0);
        } else if (event.key === "ArrowRight") {
            event.preventDefault();
            nudgeSelectedRegion(step, 0);
        } else if (event.key === "ArrowUp") {
            event.preventDefault();
            nudgeSelectedRegion(0, -step);
        } else if (event.key === "ArrowDown") {
            event.preventDefault();
            nudgeSelectedRegion(0, step);
        }
    }

    function svgPoint(event) {
        const point = els.overlay.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        return point.matrixTransform(els.overlay.getScreenCTM().inverse());
    }

    function coerceBBox([x, y, w, h]) {
        const pageWidth = Number(state.page?.effective_width || state.page?.width || 1);
        const pageHeight = Number(state.page?.effective_height || state.page?.height || 1);
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

    function formatConfidence(value) {
        if (value === null || value === undefined || value === "") return "-";
        const number = Number(value);
        if (!Number.isFinite(number)) return String(value);
        return number.toFixed(3);
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

    function escapeCssClass(value) {
        return String(value || "").replace(/[^A-Za-z0-9_-]/g, "_");
    }
}());
