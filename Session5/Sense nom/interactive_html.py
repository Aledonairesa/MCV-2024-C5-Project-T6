import os

def create_interactive_viewer(output_dir, cfg_scales, steps_list, samplers, neg_options):
    """Create an advanced interactive HTML viewer for parameter comparison"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SD 3.5 Parameter Comparison Tool</title>
        <style>
            :root {
                --primary-color: #4a6fa5;
                --secondary-color: #6f9fda;
                --bg-color: #f8f9fa;
                --card-bg: white;
                --text-color: #333;
                --border-color: #ddd;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--bg-color);
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            header {
                background-color: var(--primary-color);
                color: white;
                padding: 1rem 2rem;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                margin-top: 0;
            }
            .card {
                background: var(--card-bg);
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            .filters {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .filter-group {
                margin-bottom: 15px;
            }
            .filter-group h3 {
                margin-bottom: 8px;
                font-size: 1rem;
            }
            .checkbox-group {
                display: flex;
                flex-direction: column;
                gap: 5px;
                max-height: 150px;
                overflow-y: auto;
                padding: 5px;
                border: 1px solid var(--border-color);
                border-radius: 4px;
            }
            .checkbox-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .prompt-selector {
                margin-bottom: 20px;
            }
            .btn {
                background-color: var(--primary-color);
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 600;
                transition: background-color 0.2s;
            }
            .btn:hover {
                background-color: var(--secondary-color);
            }
            .btn-group {
                display: flex;
                gap: 10px;
                margin: 15px 0;
            }
            .results-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
            }
            .image-card {
                border: 1px solid var(--border-color);
                border-radius: 8px;
                overflow: hidden;
            }
            .image-card img {
                width: 100%;
                height: auto;
                display: block;
            }
            .image-info {
                padding: 10px;
                background-color: #f3f4f6;
                font-size: 0.9rem;
            }
            .info-grid {
                display: grid;
                grid-template-columns: auto 1fr;
                gap: 5px 10px;
            }
            .info-label {
                font-weight: 600;
            }
            .prompt-preview {
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 100%;
                margin-bottom: 5px;
                cursor: pointer;
            }
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.7);
                z-index: 1000;
                overflow: auto;
            }
            .modal-content {
                background-color: white;
                margin: 5% auto;
                padding: 20px;
                width: 80%;
                max-width: 800px;
                border-radius: 8px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                position: relative;
                animation: modalopen 0.4s;
            }
            .close-btn {
                position: absolute;
                top: 10px;
                right: 20px;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            @keyframes modalopen {
                from {opacity: 0; transform: translateY(-60px);}
                to {opacity: 1; transform: translateY(0);}
            }
            .status {
                margin: 10px 0;
                font-style: italic;
            }
            .display-options {
                display: flex;
                gap: 15px;
                align-items: center;
                margin-bottom: 15px;
            }
            .display-option {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid rgba(0,0,0,0.1);
                border-radius: 50%;
                border-top-color: var(--primary-color);
                animation: spin 1s ease-in-out infinite;
                margin: 20px auto;
                display: none;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            .hidden {
                display: none;
            }
            /* Side-by-side view */
            .side-by-side {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-top: 20px;
            }
            /* Default layout for larger screens */
            .layout-toggle {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
            }
            .layout-btn {
                padding: 5px 10px;
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 4px;
                cursor: pointer;
            }
            .layout-btn.active {
                background-color: var(--primary-color);
                color: white;
                border-color: var(--primary-color);
            }
            /* A/B Comparison */
            .comparison-container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .comparison-pair {
                display: flex;
                gap: 20px;
                border: 1px solid var(--border-color);
                padding: 15px;
                border-radius: 8px;
                background-color: #f9f9f9;
            }
            .comparison-item {
                flex: 1;
            }
            .comparison-label {
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                padding: 5px;
                background-color: var(--primary-color);
                color: white;
                border-radius: 4px;
            }
            /* Responsive */
            @media (max-width: 768px) {
                .filters {
                    grid-template-columns: 1fr;
                }
                .results-grid {
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                }
                .side-by-side {
                    grid-template-columns: 1fr;
                }
                .comparison-pair {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>SD 3.5 Parameter Optimization Tool</h1>
        </header>
        
        <div class="container">
            <div class="card">
                <h2>Filter Parameters</h2>
                
                <div class="filters">
                    <div class="filter-group">
                        <h3>Prompts</h3>
                        <div class="checkbox-group" id="prompt-filters"></div>
                    </div>
                    
                    <div class="filter-group">
                        <h3>CFG Scale</h3>
                        <div class="checkbox-group" id="cfg-filters"></div>
                    </div>
                    
                    <div class="filter-group">
                        <h3>Denoising Steps</h3>
                        <div class="checkbox-group" id="steps-filters"></div>
                    </div>
                    
                    <div class="filter-group">
                        <h3>Sampler</h3>
                        <div class="checkbox-group" id="sampler-filters"></div>
                    </div>
                    
                    <div class="filter-group">
                        <h3>Negative Prompt</h3>
                        <div class="checkbox-group" id="neg-filters"></div>
                    </div>
                </div>
                
                <div class="btn-group">
                    <button id="apply-filters" class="btn">Apply Filters</button>
                    <button id="reset-filters" class="btn">Reset Filters</button>
                    <button id="select-all" class="btn">Select All</button>
                </div>
            </div>
            
            <div class="card">
                <h2>View Options</h2>
                
                <div class="layout-toggle">
                    <button class="layout-btn active" data-layout="grid">Grid View</button>
                    <button class="layout-btn" data-layout="sideBySide">Side-by-Side</button>
                    <button class="layout-btn" data-layout="comparison">A/B Comparison</button>
                </div>
                
                <div class="display-options">
                    <div class="display-option">
                        <label for="sort-by">Sort by:</label>
                        <select id="sort-by">
                            <option value="prompt_idx">Prompt</option>
                            <option value="cfg_scale">CFG Scale</option>
                            <option value="steps">Denoising Steps</option>
                            <option value="sampler">Sampler</option>
                            <option value="negative_prompt">Negative Prompt</option>
                        </select>
                    </div>
                    <div class="display-option">
                        <label for="images-per-row">Images per row:</label>
                        <select id="images-per-row">
                            <option value="2">2</option>
                            <option value="3" selected>3</option>
                            <option value="4">4</option>
                            <option value="5">5</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="card" id="comparison-options" style="display: none;">
                <h2>Comparison Options</h2>
                <p>Select which parameter to compare (all other parameters will be held constant):</p>
                <div class="btn-group">
                    <button class="comparison-param-btn active" data-param="cfg_scale">Compare CFG Scales</button>
                    <button class="comparison-param-btn" data-param="steps">Compare Steps</button>
                    <button class="comparison-param-btn" data-param="sampler">Compare Samplers</button>
                    <button class="comparison-param-btn" data-param="negative_prompt">Compare Negative Prompt</button>
                </div>
            </div>
            
            <div class="status" id="results-status">No images match the current filters.</div>
            <div class="spinner" id="loading-spinner"></div>
            
            <div id="results-container">
                <!-- Results will be inserted here dynamically -->
                <div class="results-grid" id="grid-view"></div>
                <div class="side-by-side hidden" id="side-by-side-view"></div>
                <div class="comparison-container hidden" id="comparison-view"></div>
            </div>
        </div>
        
        <!-- Prompt Modal -->
        <div id="prompt-modal" class="modal">
            <div class="modal-content">
                <span class="close-btn">&times;</span>
                <h2>Full Prompt</h2>
                <p id="full-prompt-text"></p>
            </div>
        </div>
        
        <script>
            // State
            let allResults = [];
            let allPrompts = [];
            let filteredResults = [];
            let currentLayout = 'grid';
            let comparisonParam = 'cfg_scale';
            
            // DOM elements
            const promptFilters = document.getElementById('prompt-filters');
            const cfgFilters = document.getElementById('cfg-filters');
            const stepsFilters = document.getElementById('steps-filters');
            const samplerFilters = document.getElementById('sampler-filters');
            const negFilters = document.getElementById('neg-filters');
            const applyFiltersBtn = document.getElementById('apply-filters');
            const resetFiltersBtn = document.getElementById('reset-filters');
            const selectAllBtn = document.getElementById('select-all');
            const resultsStatus = document.getElementById('results-status');
            const loadingSpinner = document.getElementById('loading-spinner');
            const gridView = document.getElementById('grid-view');
            const sideBySideView = document.getElementById('side-by-side-view');
            const comparisonView = document.getElementById('comparison-view');
            const promptModal = document.getElementById('prompt-modal');
            const fullPromptText = document.getElementById('full-prompt-text');
            const closeModalBtn = document.querySelector('.close-btn');
            const sortBySelect = document.getElementById('sort-by');
            const imagesPerRowSelect = document.getElementById('images-per-row');
            const layoutButtons = document.querySelectorAll('.layout-btn');
            const comparisonOptions = document.getElementById('comparison-options');
            const comparisonParamButtons = document.querySelectorAll('.comparison-param-btn');
            
            // Load data
            async function loadData() {
                try {
                    loadingSpinner.style.display = 'block';
                    
                    // Load prompts and results
                    const promptsResponse = await fetch('prompts.json');
                    const resultsResponse = await fetch('results.json');
                    
                    if (!promptsResponse.ok || !resultsResponse.ok) {
                        throw new Error('Failed to load data');
                    }
                    
                    allPrompts = await promptsResponse.json();
                    allResults = await resultsResponse.json();
                    filteredResults = [...allResults];
                    
                    // Initialize filters
                    initializeFilters();
                    
                    // Display initial results
                    updateResults();
                    
                } catch (error) {
                    console.error('Error loading data:', error);
                    resultsStatus.textContent = 'Error loading data. Please check the console for details.';
                } finally {
                    loadingSpinner.style.display = 'none';
                }
            }
            
            // Initialize filter checkboxes
            function initializeFilters() {
                // Get unique values
                const promptIndices = [...new Set(allResults.map(r => r.prompt_idx))];
                const cfgScales = [...new Set(allResults.map(r => r.cfg_scale))].sort((a, b) => a - b);
                const stepsList = [...new Set(allResults.map(r => r.steps))].sort((a, b) => a - b);
                const samplers = [...new Set(allResults.map(r => r.sampler))].sort();
                const negOptions = [...new Set(allResults.map(r => r.negative_prompt))].sort();
                
                // Create prompt checkboxes
                promptIndices.forEach(idx => {
                    const promptPreview = allPrompts[idx].length > 40 ? 
                        allPrompts[idx].substring(0, 40) + '...' : 
                        allPrompts[idx];
                    
                    const div = document.createElement('div');
                    div.className = 'checkbox-item';
                    div.innerHTML = `
                        <input type="checkbox" id="prompt-${idx}" value="${idx}" checked>
                        <label for="prompt-${idx}">Prompt ${idx + 1}</label>
                    `;
                    promptFilters.appendChild(div);
                });
                
                // Create CFG scale checkboxes
                cfgScales.forEach(cfg => {
                    const div = document.createElement('div');
                    div.className = 'checkbox-item';
                    div.innerHTML = `
                        <input type="checkbox" id="cfg-${cfg}" value="${cfg}" checked>
                        <label for="cfg-${cfg}">${cfg}</label>
                    `;
                    cfgFilters.appendChild(div);
                });
                
                // Create steps checkboxes
                stepsList.forEach(steps => {
                    const div = document.createElement('div');
                    div.className = 'checkbox-item';
                    div.innerHTML = `
                        <input type="checkbox" id="steps-${steps}" value="${steps}" checked>
                        <label for="steps-${steps}">${steps}</label>
                    `;
                    stepsFilters.appendChild(div);
                });
                
                // Create sampler checkboxes
                samplers.forEach(sampler => {
                    const div = document.createElement('div');
                    div.className = 'checkbox-item';
                    div.innerHTML = `
                        <input type="checkbox" id="sampler-${sampler}" value="${sampler}" checked>
                        <label for="sampler-${sampler}">${sampler}</label>
                    `;
                    samplerFilters.appendChild(div);
                });
                
                // Create negative prompt checkboxes
                negOptions.forEach(neg => {
                    const div = document.createElement('div');
                    div.className = 'checkbox-item';
                    div.innerHTML = `
                        <input type="checkbox" id="neg-${neg}" value="${neg}" checked>
                        <label for="neg-${neg}">${neg ? 'Enabled' : 'Disabled'}</label>
                    `;
                    negFilters.appendChild(div);
                });
            }
            
            // Apply filters
            function applyFilters() {
                loadingSpinner.style.display = 'block';
                
                // Get selected values
                const selectedPrompts = Array.from(promptFilters.querySelectorAll('input:checked')).map(cb => parseInt(cb.value));
                const selectedCfg = Array.from(cfgFilters.querySelectorAll('input:checked')).map(cb => parseFloat(cb.value));
                const selectedSteps = Array.from(stepsFilters.querySelectorAll('input:checked')).map(cb => parseInt(cb.value));
                const selectedSamplers = Array.from(samplerFilters.querySelectorAll('input:checked')).map(cb => cb.value);
                const selectedNeg = Array.from(negFilters.querySelectorAll('input:checked')).map(cb => cb.value === 'true');
                
                // Filter results
                filteredResults = allResults.filter(result => 
                    selectedPrompts.includes(result.prompt_idx) &&
                    selectedCfg.includes(result.cfg_scale) &&
                    selectedSteps.includes(result.steps) &&
                    selectedSamplers.includes(result.sampler) &&
                    selectedNeg.includes(result.negative_prompt)
                );
                
                updateResults();
                loadingSpinner.style.display = 'none';
            }
            
            // Reset filters
            function resetFilters() {
                // Check all checkboxes
                document.querySelectorAll('.checkbox-group input').forEach(cb => {
                    cb.checked = true;
                });
                
                // Reset to all results
                filteredResults = [...allResults];
                updateResults();
            }
            
            // Select all or none
            function toggleSelectAll() {
                const allSelected = Array.from(document.querySelectorAll('.checkbox-group input')).every(cb => cb.checked);
                
                document.querySelectorAll('.checkbox-group input').forEach(cb => {
                    cb.checked = !allSelected;
                });
                
                if (allSelected) {
                    // If all were selected, now none are, so show no results
                    filteredResults = [];
                    updateResults();
                    selectAllBtn.textContent = 'Select All';
                } else {
                    // If none or some were selected, now all are
                    resetFilters();
                    selectAllBtn.textContent = 'Select None';
                }
            }
            
            // Update results display
            function updateResults() {
                resultsStatus.textContent = `Showing ${filteredResults.length} images`;
                
                // Sort results if needed
                const sortBy = sortBySelect.value;
                filteredResults.sort((a, b) => {
                    if (typeof a[sortBy] === 'number') {
                        return a[sortBy] - b[sortBy];
                    }
                    return String(a[sortBy]).localeCompare(String(b[sortBy]));
                });
                
                // Update layout
                updateLayout();
            }
            
            // Handle layout changes
            function updateLayout() {
                // Hide all views
                gridView.classList.add('hidden');
                sideBySideView.classList.add('hidden');
                comparisonView.classList.add('hidden');
                comparisonOptions.style.display = 'none';
                
                // Show selected view
                if (currentLayout === 'grid') {
                    renderGridView();
                    gridView.classList.remove('hidden');
                } else if (currentLayout === 'sideBySide') {
                    renderSideBySideView();
                    sideBySideView.classList.remove('hidden');
                } else if (currentLayout === 'comparison') {
                    comparisonOptions.style.display = 'block';
                    renderComparisonView();
                    comparisonView.classList.remove('hidden');
                }
            }
            
            // Render grid view
            function renderGridView() {
                gridView.innerHTML = '';
                
                if (filteredResults.length === 0) {
                    gridView.innerHTML = '<p>No images match the current filters.</p>';
                    return;
                }
                
                // Set grid columns based on selected images per row
                const imagesPerRow = parseInt(imagesPerRowSelect.value);
                gridView.style.gridTemplateColumns = `repeat(${imagesPerRow}, 1fr)`;
                
                filteredResults.forEach(result => {
                    const card = document.createElement('div');
                    card.className = 'image-card';
                    
                    // Create a prompt preview element
                    const promptPreview = allPrompts[result.prompt_idx].length > 60 ? 
                        allPrompts[result.prompt_idx].substring(0, 60) + '...' : 
                        allPrompts[result.prompt_idx];
                    
                    card.innerHTML = `
                        <img src="${result.image_path}" alt="Generated image">
                        <div class="image-info">
                            <div class="prompt-preview" data-prompt="${result.prompt_idx}">
                                <strong>Prompt ${result.prompt_idx + 1}:</strong> ${promptPreview}
                            </div>
                            <div class="info-grid">
                                <span class="info-label">CFG Scale:</span>
                                <span>${result.cfg_scale}</span>
                                <span class="info-label">Steps:</span>
                                <span>${result.steps}</span>
                                <span class="info-label">Sampler:</span>
                                <span>${result.sampler}</span>
                                <span class="info-label">Neg Prompt:</span>
                                <span>${result.negative_prompt ? 'Enabled' : 'Disabled'}</span>
                            </div>
                        </div>
                    `;
                    
                    // Add event listener to prompt preview
                    card.querySelector('.prompt-preview').addEventListener('click', () => {
                        showPromptModal(result.prompt_idx);
                    });
                    
                    gridView.appendChild(card);
                });
            }
            
            // Render side-by-side view
            function renderSideBySideView() {
                sideBySideView.innerHTML = '';
                
                if (filteredResults.length === 0) {
                    sideBySideView.innerHTML = '<p>No images match the current filters.</p>';
                    return;
                }
                
                // Group results by prompt
                const resultsByPrompt = {};
                filteredResults.forEach(result => {
                    const promptIdx = result.prompt_idx;
                    if (!resultsByPrompt[promptIdx]) {
                        resultsByPrompt[promptIdx] = [];
                    }
                    resultsByPrompt[promptIdx].push(result);
                });
                
                // Create a section for each prompt
                Object.entries(resultsByPrompt).forEach(([promptIdx, results]) => {
                    const promptSection = document.createElement('div');
                    promptSection.className = 'card';
                    
                    const promptText = allPrompts[promptIdx];
                    const promptPreview = promptText.length > 100 ? promptText.substring(0, 100) + '...' : promptText;
                    
                    promptSection.innerHTML = `
                        <h3>Prompt ${parseInt(promptIdx) + 1}</h3>
                        <div class="prompt-preview" data-prompt="${promptIdx}">${promptPreview}</div>
                    `;
                    
                    // Add event listener to prompt preview
                    promptSection.querySelector('.prompt-preview').addEventListener('click', () => {
                        showPromptModal(promptIdx);
                    });
                    
                    // Create result grid for this prompt
                    const resultGrid = document.createElement('div');
                    resultGrid.className = 'results-grid';
                    resultGrid.style.gridTemplateColumns = `repeat(${parseInt(imagesPerRowSelect.value)}, 1fr)`;
                    
                    results.forEach(result => {
                        const card = document.createElement('div');
                        card.className = 'image-card';
                        card.innerHTML = `
                            <img src="${result.image_path}" alt="Generated image">
                            <div class="image-info">
                                <div class="info-grid">
                                    <span class="info-label">CFG Scale:</span>
                                    <span>${result.cfg_scale}</span>
                                    <span class="info-label">Steps:</span>
                                    <span>${result.steps}</span>
                                    <span class="info-label">Sampler:</span>
                                    <span>${result.sampler}</span>
                                    <span class="info-label">Neg Prompt:</span>
                                    <span>${result.negative_prompt ? 'Enabled' : 'Disabled'}</span>
                                </div>
                            </div>
                        `;
                        resultGrid.appendChild(card);
                    });
                    
                    promptSection.appendChild(resultGrid);
                    sideBySideView.appendChild(promptSection);
                });
            }
            
            // Render comparison view
            function renderComparisonView() {
                comparisonView.innerHTML = '';
                
                if (filteredResults.length === 0) {
                    comparisonView.innerHTML = '<p>No images match the current filters.</p>';
                    return;
                }
                
                // Group results for comparison based on selected parameter
                const resultGroups = {};
                
                filteredResults.forEach(result => {
                    // Create a key for grouping based on all parameters except the comparison parameter
                    const otherParams = ['prompt_idx', 'cfg_scale', 'steps', 'sampler', 'negative_prompt']
                        .filter(param => param !== comparisonParam)
                        .map(param => `${param}:${result[param]}`)
                        .join('_');
                    
                    if (!resultGroups[otherParams]) {
                        resultGroups[otherParams] = [];
                    }
                    
                    resultGroups[otherParams].push(result);
                });
                
                // Only keep groups with multiple values of the comparison parameter
                const validGroups = Object.entries(resultGroups)
                    .filter(([_, results]) => {
                        const uniqueValues = new Set(results.map(r => r[comparisonParam]));
                        return uniqueValues.size > 1;
                    })
                    .sort(([keyA, resultsA], [keyB, resultsB]) => {
                        // Sort by prompt index first
                        const promptIdxA = resultsA[0].prompt_idx;
                        const promptIdxB = resultsB[0].prompt_idx;
                        return promptIdxA - promptIdxB;
                    });
                
                if (validGroups.length === 0) {
                    comparisonView.innerHTML = '<p>No valid comparison groups found. Try selecting more parameter combinations.</p>';
                    return;
                }
                
                // Create comparison pairs
                validGroups.forEach(([key, results]) => {
                    // Sort results by the comparison parameter
                    results.sort((a, b) => {
                        if (typeof a[comparisonParam] === 'number') {
                            return a[comparisonParam] - b[comparisonParam];
                        }
                        return String(a[comparisonParam]).localeCompare(String(b[comparisonParam]));
                    });
                    
                    const promptIdx = results[0].prompt_idx;
                    const promptText = allPrompts[promptIdx];
                    const promptPreview = promptText.length > 100 ? promptText.substring(0, 100) + '...' : promptText;
                    
                    // Create section header
                    const sectionHeader = document.createElement('div');
                    sectionHeader.className = 'card';
                    sectionHeader.innerHTML = `
                        <h3>Prompt ${promptIdx + 1}</h3>
                        <div class="prompt-preview" data-prompt="${promptIdx}">${promptPreview}</div>
                        <p><strong>Comparing:</strong> ${getParamLabel(comparisonParam)}</p>
                        <p><strong>Fixed parameters:</strong> 
                            ${comparisonParam !== 'cfg_scale' ? `CFG Scale: ${results[0].cfg_scale}` : ''}
                            ${comparisonParam !== 'steps' ? `Steps: ${results[0].steps}` : ''}
                            ${comparisonParam !== 'sampler' ? `Sampler: ${results[0].sampler}` : ''}
                            ${comparisonParam !== 'negative_prompt' ? `Negative Prompt: ${results[0].negative_prompt ? 'Enabled' : 'Disabled'}` : ''}
                        </p>
                    `;
                    
                    // Add event listener to prompt preview
                    sectionHeader.querySelector('.prompt-preview').addEventListener('click', () => {
                        showPromptModal(promptIdx);
                    });
                    
                    comparisonView.appendChild(sectionHeader);
                    
                    // Create comparison pairs
                    for (let i = 0; i < results.length - 1; i++) {
                        for (let j = i + 1; j < results.length; j++) {
                            const pairContainer = document.createElement('div');
                            pairContainer.className = 'comparison-pair';
                            
                            // A side
                            const aItem = document.createElement('div');
                            aItem.className = 'comparison-item';
                            aItem.innerHTML = `
                                <div class="comparison-label">${formatParamValue(comparisonParam, results[i][comparisonParam])}</div>
                                <img src="${results[i].image_path}" alt="Generated image A" style="width: 100%;">
                            `;
                            
                            // B side
                            const bItem = document.createElement('div');  
                            bItem.className = 'comparison-item';
                            bItem.innerHTML = `
                                <div class="comparison-label">${formatParamValue(comparisonParam, results[j][comparisonParam])}</div>
                                <img src="${results[j].image_path}" alt="Generated image B" style="width: 100%;">
                            `;
                            
                            pairContainer.appendChild(aItem);
                            pairContainer.appendChild(bItem);
                            comparisonView.appendChild(pairContainer);
                        }
                    }
                });
            }
            
            // Helper function to get parameter label
            function getParamLabel(param) {
                const labels = {
                    'cfg_scale': 'CFG Scale',
                    'steps': 'Denoising Steps',
                    'sampler': 'Sampler',
                    'negative_prompt': 'Negative Prompt'
                };
                return labels[param] || param;
            }
            
            // Helper function to format parameter values
            function formatParamValue(param, value) {
                if (param === 'negative_prompt') {
                    return value ? 'Negative Prompt Enabled' : 'Negative Prompt Disabled';
                }
                return value;
            }
            
            // Show prompt modal
            function showPromptModal(promptIdx) {
                fullPromptText.textContent = allPrompts[promptIdx];
                promptModal.style.display = 'block';
            }
            
            // Close modal
            closeModalBtn.addEventListener('click', () => {
                promptModal.style.display = 'none';
            });
            
            // Close modal when clicking outside of it
            window.addEventListener('click', (event) => {
                if (event.target === promptModal) {
                    promptModal.style.display = 'none';
                }
            });
            
            // Event listeners
            applyFiltersBtn.addEventListener('click', applyFilters);
            resetFiltersBtn.addEventListener('click', resetFilters);
            selectAllBtn.addEventListener('click', toggleSelectAll);
            sortBySelect.addEventListener('change', updateResults);
            imagesPerRowSelect.addEventListener('change', updateResults);
            
            // Layout buttons
            layoutButtons.forEach(btn => {
                btn.addEventListener('click', () => {
                    layoutButtons.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    currentLayout = btn.dataset.layout;
                    updateLayout();
                });
            });
            
            // Comparison parameter buttons
            comparisonParamButtons.forEach(btn => {
                btn.addEventListener('click', () => {
                    comparisonParamButtons.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    comparisonParam = btn.dataset.param;
                    renderComparisonView();
                });
            });
            
            // Load data on page load
            document.addEventListener('DOMContentLoaded', loadData);
        </script>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "interactive_viewer.html"), "w") as f:
        f.write(html_content)