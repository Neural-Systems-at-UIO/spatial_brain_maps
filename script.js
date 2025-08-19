document.addEventListener('DOMContentLoaded', function () {
    // DOM elements
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const resultsList = document.getElementById('resultsList');
    const resultCount = document.getElementById('resultCount');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const structureSelect = document.getElementById('structureSelect');
    const metricSelect = document.getElementById('metricSelect');
    const sortOrder = document.getElementById('sortOrder');
    const structureLabel = document.getElementById('structureLabel');
    const metricLabel = document.getElementById('metricLabel');

    // Pagination variables
    const itemsPerPage = 10;
    let currentPage = 1;
    let filteredData = [];
    let fileData = [];
    let structureData = {};
    let structuresList = [];
    // Multi-region state
    const multiRegionContainer = document.getElementById('multiRegionContainer'); // inline region criteria holder
    const regionCriteriaList = document.getElementById('regionCriteriaList');
    const addRegionBtn = document.getElementById('addRegionBtn');
    let multiRegionCriteria = []; // {id, structure, metric(intensity|expression_pct), direction(desc|asc)}
    let multiRegionCache = {}; // cache structure metric datasets { structure: { intensity: {}, expression_pct: {} } }

    // Toggle structure/metric controls based on sort selection
    function toggleStructureControls() {
        const isStructureSort = document.querySelector('input[name="sortBy"][value="structure"]').checked;
        const isMultiRegion = document.querySelector('input[name="sortBy"][value="multi-region"]').checked;
        structureSelect.disabled = !isStructureSort;
        metricSelect.disabled = !isStructureSort;
        structureLabel.classList.toggle('disabled', !isStructureSort);
        metricLabel.classList.toggle('disabled', !isStructureSort);
        const structureGroup = document.querySelector('[data-role="structure-group"]');
        const metricGroup = document.querySelector('[data-role="metric-group"]');
        const sortGroup = document.querySelector('[data-role="sort-group"]');
        if (isMultiRegion) {
            if (multiRegionContainer) multiRegionContainer.style.display = 'flex';
            if (structureGroup) structureGroup.style.display = 'none';
            if (metricGroup) metricGroup.style.display = 'none';
            if (sortGroup) sortGroup.style.display = 'none';
            if (!multiRegionCriteria.length) createRegionCriterionRow();
        } else {
            if (multiRegionContainer) multiRegionContainer.style.display = 'none';
            if (structureGroup) structureGroup.style.display = '';
            if (metricGroup) metricGroup.style.display = '';
            if (sortGroup) sortGroup.style.display = '';
        }
        // Reflect disabled state ONLY in the main (single-structure) custom select, not the multi-region mini selects
        const mainWrapper = document.querySelector('.structure-select-wrapper:not(.mini)');
        if (mainWrapper) {
            const trigger = mainWrapper.querySelector('.structure-select-trigger');
            if (trigger) {
                if (!isStructureSort) {
                    trigger.classList.add('disabled');
                    trigger.setAttribute('aria-disabled', 'true');
                    trigger.tabIndex = -1;
                    mainWrapper.classList.remove('open');
                    trigger.classList.add('has-disabled-tooltip');
                    const tip = "Select 'Sort by structure metrics' to enable";
                    trigger.setAttribute('data-disabled-tooltip', tip);
                    trigger.setAttribute('title', tip);
                } else {
                    trigger.classList.remove('disabled');
                    trigger.removeAttribute('aria-disabled');
                    trigger.tabIndex = 0;
                    trigger.classList.remove('has-disabled-tooltip');
                    trigger.removeAttribute('data-disabled-tooltip');
                    trigger.removeAttribute('title');
                }
            }
        }
        // Always enable multi-region mini selects when in multi-region mode
        if (isMultiRegion) {
            document.querySelectorAll('.structure-select-wrapper.mini .structure-select-trigger').forEach(trig => {
                trig.classList.remove('disabled','has-disabled-tooltip');
                trig.removeAttribute('aria-disabled');
                trig.removeAttribute('data-disabled-tooltip');
                trig.removeAttribute('title');
                trig.tabIndex = 0;
            });
        }
        // Tooltip for metric select as well
        if (!isStructureSort) {
            const tip = "Select 'Sort by structure metrics' to enable";
            metricSelect.setAttribute('title', tip);
        } else {
            metricSelect.removeAttribute('title');
        }
        if (!isMultiRegion) {
            // Re-run standard sort if exiting multi-region mode
            if (multiRegionCriteria.length) {
                sortResults();
                displayResults();
            }
        } else {
            computeMultiRegionRanks();
        }
    }

    // Load structures list
    async function loadStructuresList() {
        try {
            const response = await fetch('data/structure_names.json.gz');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const compressedData = await response.arrayBuffer();
            const decompressedData = pako.inflate(compressedData, { to: 'string' });
            structuresList = JSON.parse(decompressedData);

            // Populate structure dropdown
            structureSelect.innerHTML = '';

            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Select a structure...';
            structureSelect.appendChild(defaultOption);

            structuresList.forEach(structure => {
                const option = document.createElement('option');
                option.value = structure;
                option.textContent = structure;
                structureSelect.appendChild(option);
            });

            // Initialize custom searchable dropdown for structure select
            initCustomStructureSelect();

            // Rebuild any existing multi-region criteria rows now that structures are available
            rebuildMultiRegionCriteriaUI();
        } catch (error) {
            console.error('Error loading structures list:', error);
        }
    }

    async function loadStructureData(structure, metric) {
        if (!structure) return {};

        try {
            const safeStructureName = structure.replace(/[\\/]/g, '_');
            const response = await fetch(`data/${safeStructureName}_${metric}.json.gz`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const compressedData = await response.arrayBuffer();
            const decompressedData = pako.inflate(compressedData, { to: 'string' });
            return JSON.parse(decompressedData);
        } catch (error) {
            console.error(`Error loading ${metric} data for ${structure}:`, error);
            return {};
        }
    }

    function addPaginationControls() {
        const totalPages = Math.ceil(filteredData.length / itemsPerPage);

        const existingPagination = document.querySelector('.pagination');
        if (existingPagination) {
            existingPagination.remove();
        }

        if (filteredData.length === 0 || totalPages <= 1) return;

        const paginationContainer = document.createElement('div');
        paginationContainer.className = 'pagination';

        const prevButton = document.createElement('button');
        prevButton.innerHTML = '<i class="fas fa-chevron-left"></i> Previous';
        prevButton.disabled = currentPage === 1;
        prevButton.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                displayResults();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });

        const pageInfo = document.createElement('span');
        pageInfo.className = 'page-info';
        pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;

        const nextButton = document.createElement('button');
        nextButton.innerHTML = 'Next <i class="fas fa-chevron-right"></i>';
        nextButton.disabled = currentPage === totalPages;
        nextButton.addEventListener('click', () => {
            if (currentPage < totalPages) {
                currentPage++;
                displayResults();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });

        paginationContainer.appendChild(prevButton);
        paginationContainer.appendChild(pageInfo);
        paginationContainer.appendChild(nextButton);

        resultsList.appendChild(paginationContainer);
    }

    async function loadFileData() {
        try {
            console.log("Starting data load...");
            loadingIndicator.style.display = 'flex';
            resultsList.style.display = 'none';

            const response = await fetch('data/gene_data_counts.json.gz');
            console.log("Fetch response status:", response.status);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const compressedData = await response.arrayBuffer();
            console.log("Got compressed data, length:", compressedData.byteLength);

            const decompressedData = pako.inflate(compressedData, { to: 'string' });
            console.log("Decompressed data length:", decompressedData.length);

            const rawData = JSON.parse(decompressedData);
            console.log("Parsed JSON data keys:", Object.keys(rawData));

            fileData = rawData.gene_name.map((name, index) => ({
                gene_name: name,
                gene_description: rawData.gene_description[index],
                synonyms: rawData.synonyms[index] || [],
                ensembl_ids: rawData.ensembl_ids[index] || [],
                number_of_animals: rawData.number_of_animals[index] || 0,
                specificity: 0,
                intensity: 0,
                expression_pct: 0,
                expression_specificity: 0
            }));

            console.log("First gene object:", fileData[0]);
            console.log("Total genes:", fileData.length);

            filteredData = [...fileData];
            sortResults();
            displayResults();
            loadingIndicator.style.display = 'none';

        } catch (error) {
            console.error('Full error:', error);
            loadingIndicator.innerHTML = `
                <p class="error">Failed to load data</p>
                <p>${error.message}</p>
                <p>Check console for details</p>
            `;
        }
    }

    function sortResults() {
        const selectedStructure = structureSelect.value;
        const selectedMetric = metricSelect.value;
        const isDescending = sortOrder.value === 'desc';
        const sortBy = document.querySelector('input[name="sortBy"]:checked').value;

        if (sortBy === 'multi-region') {
            filteredData.sort((a, b) => {
                const valA = a._multiRegionAvgRank || Infinity;
                const valB = b._multiRegionAvgRank || Infinity;
                // Lower average rank is better
                return valA - valB;
            });
            return;
        }

        filteredData.sort((a, b) => {
            let valA, valB;

            if (sortBy === 'structure' && selectedStructure) {
                valA = a[selectedMetric] || 0;
                valB = b[selectedMetric] || 0;
            } else {
                valA = a.number_of_animals;
                valB = b.number_of_animals;
            }

            return isDescending ? valB - valA : valA - valB;
        });
    }
    function displayResults() {
        console.log("Displaying results. Total:", filteredData.length);

        resultsList.style.display = 'block';
        loadingIndicator.style.display = 'none';
        resultsList.innerHTML = '';
        resultCount.textContent = filteredData.length;

        if (filteredData.length === 0) {
            resultsList.innerHTML = '<li class="result-item no-results">No matching genes found. Try a different search term.</li>';
            return;
        }

        const startIndex = (currentPage - 1) * itemsPerPage;
        const paginatedItems = filteredData.slice(startIndex, startIndex + itemsPerPage);
        const selectedStructure = structureSelect.value;
        const isStructureSort = document.querySelector('input[name="sortBy"][value="structure"]').checked;
    const isMultiRegion = document.querySelector('input[name="sortBy"][value="multi-region"]').checked;

        paginatedItems.forEach(gene => {
            const li = document.createElement('li');
            li.className = 'result-item';
            const filePath = `https://atlases.ebrains.eu/viewer-staging/#/a:juelich:iav:atlas:v1.0.0:2/t:minds:core:referencespace:v1.0.0:265d32a0-3d84-40a5-926f-bf89f68212b9/p:minds:core:parcellationatlas:v1.0.0:05655b58-3b6f-49db-b285-64b5a0276f83/x-overlay-layer:nifti:%2F%2Fhttps:%2F%2Fdata-proxy.ebrains.eu%2Fapi%2Fv1%2Fbuckets%2Fdeepslice%2Fexpression_volumes%2F${encodeURIComponent(gene.gene_name)}_interp_25um.nii.gz`;

            // Format values
            const specificityValue = gene.specificity ? gene.specificity.toFixed(3) : 'N/A';
            const intensityValue = gene.intensity ? gene.intensity.toFixed(3) : 'N/A';
            const expressionPctValue = gene.expression_pct ? (gene.expression_pct * 100).toFixed(1) + '%' : 'N/A';
            const expressionSpecValue = gene.expression_specificity ? gene.expression_specificity.toFixed(3) : 'N/A';

            li.innerHTML = `
                <a href="${filePath}" target="_blank" class="result-link">
                    <div class="result-header">
                        <i class="fas fa-dna"></i>
                        <div class="result-title">
                            <h3 class="gene-name">${gene.gene_name}</h3>
                            <p class="gene-description">${gene.gene_description || 'No description available'}</p>
                        </div>
                        <div class="mouse-count">
                            <i class="fas fa-paw"></i>
                            <span>${gene.number_of_animals} mice</span>
                        </div>
                        <i class="fas fa-external-link-alt link-icon"></i>
                    </div>
                    
                    <div class="metric-values">
                        <div class="metric-value">
                            <span class="metric-label">Specificity of Coverage:</span>
                            <span class="metric-number">${expressionSpecValue}</span>
                            ${gene.expressionSpecRank ? `<span class="metric-rank">#${gene.expressionSpecRank}</span>` : ''}
                        </div>
                        <div class="metric-value">
                            <span class="metric-label">Coverage:</span>
                            <span class="metric-number">${expressionPctValue}</span>
                            ${gene.expressionPctRank ? `<span class="metric-rank">#${gene.expressionPctRank}</span>` : ''}
                        </div>
                        <div class="metric-value">
                            <span class="metric-label">Specificity of Intensity:</span>
                            <span class="metric-number">${specificityValue}</span>
                            ${gene.specificityRank ? `<span class="metric-rank">#${gene.specificityRank}</span>` : ''}
                        </div>
                        <div class="metric-value">
                            <span class="metric-label">Intensity:</span>
                            <span class="metric-number">${intensityValue}</span>
                            ${gene.intensityRank ? `<span class="metric-rank">#${gene.intensityRank}</span>` : ''}
                        </div>
                    </div>
                    
                    <div class="gene-meta">
                        ${gene.synonyms && gene.synonyms.length ? `
                        <div class="meta-section">
                            <span class="meta-label">Synonyms:</span>
                            <span class="meta-value">${Array.isArray(gene.synonyms) ? gene.synonyms.join(', ') : gene.synonyms}</span>
                        </div>
                        ` : ''}
                        
                        ${gene.ensembl_ids && gene.ensembl_ids.length ? `
                        <div class="meta-section">
                            <span class="meta-label">Ensembl IDs:</span>
                            <span class="meta-value">${Array.isArray(gene.ensembl_ids) ? gene.ensembl_ids.join(', ') : gene.ensembl_ids}</span>
                        </div>
                        ` : ''}
                    </div>
                </a>
            `;

            // Toggle metrics visibility without affecting layout
            if (isStructureSort && selectedStructure) {
                li.classList.add('show-metrics');
            } else {
                li.classList.remove('show-metrics');
            }

            if (isMultiRegion) {
                const header = li.querySelector('.result-header');
                let badgesRow = li.querySelector('.multi-region-badges');
                if (!badgesRow) {
                    badgesRow = document.createElement('div');
                    badgesRow.className = 'multi-region-badges';
                    header.insertAdjacentElement('afterend', badgesRow);
                }
                // Clear previous
                badgesRow.innerHTML = '';
                // Avg rank first
                const avgBadge = document.createElement('div');
                avgBadge.className = 'avg-rank-badge';
                const avg = gene._multiRegionAvgRank ? gene._multiRegionAvgRank.toFixed(1) : '—';
                avgBadge.textContent = `Avg Rank ${avg}`;
                avgBadge.title = 'Average of per-criterion ranks (lower is better)';
                badgesRow.appendChild(avgBadge);

                // Per-criterion badges after
                if (Array.isArray(gene._multiRegionCriterionRanks)) {
                    gene._multiRegionCriterionRanks.forEach(cr => {
                        const cBadge = document.createElement('div');
                        cBadge.className = 'criterion-rank-badge';
                        const dirLabel = cr.direction === 'desc' ? 'high' : 'low';
                        const polarity = cr.direction === 'desc' ? 'High' : 'Low';
                        const rankTxt = cr.rank ? `Rank ${cr.rank}` : 'Rank —';
                        cBadge.textContent = `${cr.structure}: ${dirLabel} ${cr.metricLabel.toLowerCase()} ${rankTxt}`;
                        cBadge.title = `${cr.structure} • ${polarity} values preferred`;
                        badgesRow.appendChild(cBadge);
                    });
                }
            }

            resultsList.appendChild(li);
        });

        addPaginationControls();
    }
    async function updateGeneMetrics() {
        const structure = structureSelect.value;

        if (!structure) {
            fileData.forEach(gene => {
                gene.specificity = 0;
                gene.intensity = 0;
                gene.expression_pct = 0;
                gene.expression_specificity = 0;
                gene.overall = 0;
                gene.specificityRank = 0;
                gene.intensityRank = 0;
                gene.expressionPctRank = 0;
                gene.expressionSpecRank = 0;
                gene.inStructure = true;
            });
            return;
        }

        loadingIndicator.style.display = 'flex';

        try {
            const [specificityData, intensityData, expressionPctData, expressionSpecData] = await Promise.all([
                loadStructureData(structure, 'specificity'),
                loadStructureData(structure, 'intensity'),
                loadStructureData(structure, 'expression_pct'),
                loadStructureData(structure, 'expression_specificity')
            ]);

            // First pass to collect all values for ranking
            const allValues = {
                specificity: [],
                intensity: [],
                expression_pct: [],
                expression_specificity: []
            };

            fileData.forEach(gene => {
                if (specificityData.hasOwnProperty(gene.gene_name)) {
                    allValues.specificity.push(specificityData[gene.gene_name] || 0);
                    allValues.intensity.push(intensityData[gene.gene_name] || 0);
                    allValues.expression_pct.push(expressionPctData[gene.gene_name] || 0);
                    allValues.expression_specificity.push(expressionSpecData[gene.gene_name] || 0);
                }
            });

            // Create sorted arrays for each metric
            const sortedMetrics = {};
            Object.keys(allValues).forEach(metric => {
                sortedMetrics[metric] = [...allValues[metric]].sort((a, b) => b - a); // Sort descending
            });

            // Second pass to calculate overall score and store rankings
            fileData.forEach(gene => {
                if (specificityData.hasOwnProperty(gene.gene_name)) {
                    gene.specificity = specificityData[gene.gene_name] || 0;
                    gene.intensity = intensityData[gene.gene_name] || 0;
                    gene.expression_pct = expressionPctData[gene.gene_name] || 0;
                    gene.expression_specificity = expressionSpecData[gene.gene_name] || 0;

                    // Calculate and store rankings for each metric
                    gene.specificityRank = sortedMetrics.specificity.indexOf(gene.specificity) + 1;
                    gene.intensityRank = sortedMetrics.intensity.indexOf(gene.intensity) + 1;
                    gene.expressionPctRank = sortedMetrics.expression_pct.indexOf(gene.expression_pct) + 1;
                    gene.expressionSpecRank = sortedMetrics.expression_specificity.indexOf(gene.expression_specificity) + 1;

                    // Calculate overall score with coverage weighted half as much
                    const weights = {
                        specificity: 1,
                        intensity: 0.005,
                        expression_pct: 0.012,
                        expression_specificity: 1
                    };

                    const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);

                    const weightedRanks = [
                        (gene.specificityRank / fileData.length) * weights.specificity,
                        (gene.intensityRank / fileData.length) * weights.intensity,
                        (gene.expressionPctRank / fileData.length) * weights.expression_pct,
                        (gene.expressionSpecRank / fileData.length) * weights.expression_specificity
                    ];

                    gene.overall = 1 - (weightedRanks.reduce((a, b) => a + b, 0) / totalWeight);

                    gene.inStructure = true;
                } else {
                    gene.specificity = 0;
                    gene.intensity = 0;
                    gene.expression_pct = 0;
                    gene.expression_specificity = 0;
                    gene.overall = 0;
                    gene.specificityRank = 0;
                    gene.intensityRank = 0;
                    gene.expressionPctRank = 0;
                    gene.expressionSpecRank = 0;
                    gene.inStructure = false;
                }
            });

            searchFiles(searchInput.value);
        } catch (error) {
            console.error('Error updating gene metrics:', error);
        } finally {
            loadingIndicator.style.display = 'none';
        }
    }


    function searchFiles(query) {
        console.group("Search Execution");
        try {
            currentPage = 1;
            const lowerQuery = query.toLowerCase().trim();
            const selectedStructure = structureSelect.value;
            console.log("Search query:", `"${lowerQuery}"`);

            if (!lowerQuery) {
                console.log("Empty query - showing all results");
                filteredData = fileData.filter(gene => {
                    return !selectedStructure || gene.inStructure;
                });
            } else {
                console.log("Filtering data...");
                filteredData = fileData.filter(gene => {
                    if (selectedStructure && !gene.inStructure) {
                        return false;
                    }

                    const searchFields = [
                        gene.gene_name,
                        gene.gene_description,
                        ...(Array.isArray(gene.synonyms) ? gene.synonyms : [gene.synonyms]),
                        ...(Array.isArray(gene.ensembl_ids) ? gene.ensembl_ids : [gene.ensembl_ids])
                    ].filter(Boolean).map(f => String(f).toLowerCase());

                    return searchFields.some(field => field.includes(lowerQuery));
                });
                console.log(`Found ${filteredData.length} matches`);
            }

            sortResults();
            displayResults();
        } catch (error) {
            console.error("Search error:", error);
        } finally {
            console.groupEnd();
        }
    }

    // Event listeners
    searchButton.addEventListener('click', () => {
        searchFiles(searchInput.value);
    });

    searchInput.addEventListener('input', () => {
        searchFiles(searchInput.value);
    });

    document.querySelectorAll('input[name="sortBy"]').forEach(radio => {
        radio.addEventListener('change', () => {
            toggleStructureControls();
            sortResults();
            displayResults();
        });
    });

    structureSelect.addEventListener('change', async () => {
        await updateGeneMetrics();
    });

    metricSelect.addEventListener('change', () => {
        sortResults();
        displayResults();
    });

    sortOrder.addEventListener('change', () => {
        sortResults();
        displayResults();
    });
    const metricsInfoButton = document.getElementById('metricsInfoButton');
    const metricsInfoBox = document.getElementById('metricsInfoBox');
    const closeInfoBox = document.getElementById('closeInfoBox');

    metricsInfoButton.addEventListener('click', () => {
        metricsInfoBox.style.display = 'block';
    });

    closeInfoBox.addEventListener('click', () => {
        metricsInfoBox.style.display = 'none';
    });

    // Close when clicking outside
    document.addEventListener('click', (e) => {
        if (!metricsInfoBox.contains(e.target) && e.target !== metricsInfoButton) {
            metricsInfoBox.style.display = 'none';
        }
    });

    // THEME TOGGLE
    const themeToggle = document.getElementById('themeToggle');
    const themeLabel = themeToggle ? themeToggle.querySelector('.theme-label') : null;
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('sbm-theme');
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        if (themeLabel) { themeLabel.textContent = theme === 'dark' ? 'Light' : 'Dark'; }
        if (themeToggle) { themeToggle.querySelector('i').className = theme === 'dark' ? 'fas fa-lightbulb' : 'fas fa-moon'; }
        localStorage.setItem('sbm-theme', theme);
        // No special handling needed for custom select; styles adjust via CSS vars.
    }
    applyTheme(savedTheme || (prefersDark ? 'dark' : 'light'));
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme') || 'dark';
            applyTheme(current === 'dark' ? 'light' : 'dark');
        });
    }

    // Initial load
    toggleStructureControls();
    loadStructuresList();
    loadFileData();

    // ------------- Multi-region logic -------------
    function createRegionCriterionRow(initial = {}) {
        const id = crypto.randomUUID();
        const row = document.createElement('div');
        row.className = 'region-criterion';
        row.dataset.id = id;
        // Native select (hidden) + custom trigger (reuse pattern) for structure
        const structureSel = document.createElement('select');
        structureSel.className = 'visually-hidden';
        const placeholderOpt = document.createElement('option');
        placeholderOpt.value = '';
        placeholderOpt.textContent = 'Select a Structure...';
        structureSel.appendChild(placeholderOpt);
        structuresList.forEach(s => { const opt = document.createElement('option'); opt.value = s; opt.textContent = s; structureSel.appendChild(opt); });
        structureSel.value = initial.structure || '';

        const structureWrapper = document.createElement('div');
        structureWrapper.className = 'structure-select-wrapper mini';
        const trigger = document.createElement('button');
        trigger.type = 'button';
        trigger.className = 'structure-select-trigger';
        trigger.textContent = structureSel.value || 'Select a Structure...';
        const panel = document.createElement('div');
        panel.className = 'structure-select-panel';
        const searchWrap = document.createElement('div');
        searchWrap.className = 'structure-select-searchwrap';
        const searchBox = document.createElement('input');
        searchBox.type = 'text';
        searchBox.className = 'structure-select-search';
        searchBox.placeholder = 'Search...';
        searchWrap.appendChild(searchBox);
        const list = document.createElement('ul');
        list.className = 'structure-select-options';
        panel.appendChild(searchWrap); panel.appendChild(list);
        structureWrapper.appendChild(trigger); structureWrapper.appendChild(panel); structureWrapper.appendChild(structureSel);

        function buildOptions(filter='') {
            list.innerHTML='';
            const f = filter.toLowerCase();
            [...structureSel.options].forEach(opt=>{ if(opt.value==='' ) return; if(filter && !opt.textContent.toLowerCase().includes(f)) return; const li=document.createElement('li'); li.className='structure-option'; li.dataset.value=opt.value; li.textContent=opt.textContent; if(opt.value===structureSel.value) li.classList.add('selected','focused'); list.appendChild(li); });
        }
        function open(){ structureWrapper.classList.add('open'); trigger.setAttribute('aria-expanded','true'); buildOptions(searchBox.value.trim()); searchBox.focus(); }
        function close(){ structureWrapper.classList.remove('open'); trigger.setAttribute('aria-expanded','false'); }
        function selectValue(val){ structureSel.value=val; trigger.textContent=val||'Structure'; updateCriterion(); close(); }
        trigger.addEventListener('click',()=>{ structureWrapper.classList.contains('open')?close():open(); });
        searchBox.addEventListener('input',()=>buildOptions(searchBox.value.trim()));
        list.addEventListener('click',e=>{ const li=e.target.closest('.structure-option'); if(li) selectValue(li.dataset.value); });
        document.addEventListener('click',e=>{ if(!structureWrapper.contains(e.target)) close(); });

    const metricSel = document.createElement('select');
    metricSel.classList.add('styled-select','mr-metric');
        ['intensity','expression_pct'].forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m === 'intensity' ? 'Intensity' : 'Coverage';
            metricSel.appendChild(opt);
        });
        metricSel.value = initial.metric || 'intensity';

    const directionSel = document.createElement('select');
    directionSel.classList.add('styled-select','mr-direction');
    [{v:'desc',t:'High to Low'},{v:'asc',t:'Low to High'}].forEach(o => {
            const opt = document.createElement('option');
            opt.value = o.v;
            opt.textContent = o.t;
            directionSel.appendChild(opt);
        });
        directionSel.value = initial.direction || 'desc';

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'remove-region';
        removeBtn.innerHTML = '<i class="fas fa-times"></i> Remove';
        removeBtn.addEventListener('click', () => {
            multiRegionCriteria = multiRegionCriteria.filter(c => c.id !== id);
            row.remove();
            computeMultiRegionRanks();
            updateRemoveButtonsVisibility();
        });

        function updateCriterion(){
            const crit = multiRegionCriteria.find(c=>c.id===id);
            if(crit){ crit.structure=structureSel.value; crit.metric=metricSel.value; crit.direction=directionSel.value; }
            computeMultiRegionRanks();
            updateMultiRegionColumnWidths();
        }
        [metricSel, directionSel].forEach(sel=> sel.addEventListener('change', updateCriterion));

    // Add inline labels
    // Labels updated to mirror the single-structure controls ("Brain Structure", "Metric", "Sort Order")
    const structLabel = document.createElement('span'); structLabel.className='crit-label'; structLabel.textContent='Brain Structure:';
    const metricLabelEl = document.createElement('span'); metricLabelEl.className='crit-label'; metricLabelEl.textContent='Metric:';
    const dirLabel = document.createElement('span'); dirLabel.className='crit-label'; dirLabel.textContent='Sort Order:';
    row.appendChild(structLabel);
    row.appendChild(structureWrapper);
    row.appendChild(metricLabelEl);
    row.appendChild(metricSel);
    row.appendChild(dirLabel);
    row.appendChild(directionSel);
    row.appendChild(removeBtn);
    if (regionCriteriaList) regionCriteriaList.appendChild(row);

    multiRegionCriteria.push({ id, structure: structureSel.value, metric: metricSel.value, direction: directionSel.value });
    updateMultiRegionColumnWidths();
    updateRemoveButtonsVisibility();
    }

    // Rebuild existing criteria UI after structures list arrives (handles user switching early)
    function rebuildMultiRegionCriteriaUI() {
        if (!regionCriteriaList) return;
        if (!structuresList.length) return; // nothing to populate yet
        if (!multiRegionCriteria.length) return; // no criteria to rebuild
        const snapshot = multiRegionCriteria.map(c => ({ ...c }));
        regionCriteriaList.innerHTML = '';
        multiRegionCriteria = [];
        snapshot.forEach(c => createRegionCriterionRow(c));
        computeMultiRegionRanks();
        updateMultiRegionColumnWidths();
    }

    function updateRemoveButtonsVisibility() {
        const buttons = document.querySelectorAll('.region-criterion .remove-region');
        if (multiRegionCriteria.length <= 1) {
            buttons.forEach(btn => btn.style.display = 'none');
        } else {
            buttons.forEach(btn => btn.style.display = 'inline-flex');
        }
    }

    // Equalize widths of structure trigger, metric select, direction select across rows for alignment
    function updateMultiRegionColumnWidths() {
        const rows = Array.from(document.querySelectorAll('.region-criterion'));
        if (!rows.length) return;
        let wStruct = 0, wMetric = 0, wDir = 0;
        // Reset to auto to measure natural widths
        rows.forEach(r => {
            const st = r.querySelector('.structure-select-wrapper .structure-select-trigger');
            const ms = r.querySelector('select.mr-metric');
            const ds = r.querySelector('select.mr-direction');
            if (st) st.style.width = 'auto';
            if (ms) ms.style.width = 'auto';
            if (ds) ds.style.width = 'auto';
        });
        rows.forEach(r => {
            const st = r.querySelector('.structure-select-wrapper .structure-select-trigger');
            const ms = r.querySelector('select.mr-metric');
            const ds = r.querySelector('select.mr-direction');
            if (st) wStruct = Math.max(wStruct, st.offsetWidth);
            if (ms) wMetric = Math.max(wMetric, ms.offsetWidth);
            if (ds) wDir = Math.max(wDir, ds.offsetWidth);
        });
        rows.forEach(r => {
            const st = r.querySelector('.structure-select-wrapper .structure-select-trigger');
            const ms = r.querySelector('select.mr-metric');
            const ds = r.querySelector('select.mr-direction');
            if (st) st.style.width = wStruct + 'px';
            if (ms) ms.style.width = wMetric + 'px';
            if (ds) ds.style.width = wDir + 'px';
        });
    }

    // Recalculate on window resize (debounced)
    let mrWidthTO;
    window.addEventListener('resize', () => {
        clearTimeout(mrWidthTO);
        mrWidthTO = setTimeout(updateMultiRegionColumnWidths, 120);
    });

    if (addRegionBtn) addRegionBtn.addEventListener('click', () => createRegionCriterionRow());

    async function ensureStructureMetrics(structure) {
        if (!structure) return null;
        if (!multiRegionCache[structure]) {
            multiRegionCache[structure] = {};
        }
        const needed = ['intensity','expression_pct'].filter(m => !multiRegionCache[structure][m]);
        if (!needed.length) return multiRegionCache[structure];
        try {
            const loads = await Promise.all(needed.map(m => loadStructureData(structure, m)));
            needed.forEach((m,i) => multiRegionCache[structure][m] = loads[i]);
            return multiRegionCache[structure];
        } catch (e) {
            console.error('Failed loading multi-region metrics', structure, e);
            return multiRegionCache[structure];
        }
    }

    async function computeMultiRegionRanks() {
        const active = multiRegionCriteria.filter(c => c.structure);
        if (!document.querySelector('input[name="sortBy"][value="multi-region"]').checked) return; // not active
        if (!active.length) {
            filteredData = [...fileData];
            filteredData.forEach(g => delete g._multiRegionAvgRank);
            sortResults();
            displayResults();
            return;
        }
        loadingIndicator.style.display = 'flex';
        try {
            // Load needed structures
            await Promise.all(active.map(c => ensureStructureMetrics(c.structure)));
            // For each criterion, build rank map { gene -> rank }
            const criterionRanks = [];
            for (const crit of active) {
                const dataset = multiRegionCache[crit.structure]?.[crit.metric] || {};
                // Build sorted array of values (desc always for ranking by value high best) then adjust for direction
                const entries = Object.entries(dataset);
                if (!entries.length) continue;
                // For coverage metric we might treat values naturally
                // Sort descending by value so index+1 is rank for high-to-low preference
                entries.sort((a,b) => b[1] - a[1]);
                const rankMapHigh = new Map();
                entries.forEach(([gene,val],idx) => rankMapHigh.set(gene, idx+1));
                let rankMap;
                if (crit.direction === 'desc') {
                    rankMap = rankMapHigh; // high desirable
                } else {
                    // For low desirable, invert ranks: newRank = (N - originalRank + 1)
                    const N = entries.length;
                    rankMap = new Map();
                    entries.forEach(([gene,_val],idx) => {
                        const originalRank = idx+1;
                        rankMap.set(gene, N - originalRank + 1);
                    });
                }
                criterionRanks.push({ crit, rankMap });
            }
            // Compute average rank across criteria for each gene present in all criteria
            filteredData = fileData.filter(gene => {
                return criterionRanks.every(obj => obj.rankMap.has(gene.gene_name));
            });
            filteredData.forEach(g => {
                const ranks = criterionRanks.map(obj => obj.rankMap.get(g.gene_name));
                g._multiRegionAvgRank = ranks.reduce((a,b)=>a+b,0)/ranks.length;
                g._multiRegionCriterionRanks = criterionRanks.map(obj => ({
                    structure: obj.crit.structure,
                    metric: obj.crit.metric,
                    metricLabel: obj.crit.metric === 'intensity' ? 'Intensity' : 'Coverage',
                    direction: obj.crit.direction,
                    rank: obj.rankMap.get(g.gene_name)
                }));
            });
            sortResults();
            displayResults();
        } catch (e) {
            console.error('Error computing multi-region ranks', e);
        } finally {
            loadingIndicator.style.display = 'none';
        }
    }

    // If user switches to multi-region mode and there are no criteria yet, add one automatically
    const multiRegionRadio = document.querySelector('input[name="sortBy"][value="multi-region"]');
    if (multiRegionRadio) {
        multiRegionRadio.addEventListener('change', e => {
            if (e.target.checked && !multiRegionCriteria.length) createRegionCriterionRow();
            toggleStructureControls();
            computeMultiRegionRanks();
        });
    }

    // -------- Custom Structure Select Implementation --------
    function initCustomStructureSelect() {
        // Prevent re-init
        if (document.querySelector('.structure-select-wrapper')) return;
        const select = structureSelect;
        select.classList.add('visually-hidden');
        const wrapper = document.createElement('div');
        wrapper.className = 'structure-select-wrapper';
        const trigger = document.createElement('button');
        trigger.type = 'button';
        trigger.className = 'structure-select-trigger';
        trigger.setAttribute('aria-haspopup', 'listbox');
        trigger.setAttribute('aria-expanded', 'false');
        trigger.textContent = select.options[select.selectedIndex]?.textContent || 'Select a structure...';
        const panel = document.createElement('div');
        panel.className = 'structure-select-panel';
        panel.setAttribute('role', 'listbox');
        const searchWrap = document.createElement('div');
        searchWrap.className = 'structure-select-searchwrap';
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.className = 'structure-select-search';
        searchInput.placeholder = 'Search structures...';
        searchWrap.appendChild(searchInput);
        const list = document.createElement('ul');
        list.className = 'structure-select-options';
        panel.appendChild(searchWrap);
        panel.appendChild(list);
        wrapper.appendChild(trigger);
        wrapper.appendChild(panel);
        select.parentNode.insertBefore(wrapper, select);

        function buildOptions(filter = '') {
            list.innerHTML = '';
            const filterLower = filter.toLowerCase();
            [...select.options].forEach(opt => {
                if (opt.value === '') return; // skip placeholder
                if (filter && !opt.textContent.toLowerCase().includes(filterLower)) return;
                const li = document.createElement('li');
                li.className = 'structure-option';
                li.setAttribute('role', 'option');
                li.dataset.value = opt.value;
                li.textContent = opt.textContent;
                if (opt.selected) li.classList.add('selected');
                list.appendChild(li);
            });
            // Prefer focusing the currently selected option; fall back to first option.
            const selectedLi = list.querySelector('.structure-option.selected');
            const focusTarget = selectedLi || list.querySelector('.structure-option');
            if (focusTarget) {
                list.querySelectorAll('.focused').forEach(el => el.classList.remove('focused'));
                focusTarget.classList.add('focused');
            }
        }

        function open() {
            wrapper.classList.add('open');
            trigger.setAttribute('aria-expanded', 'true');
            buildOptions(searchInput.value.trim());
            searchInput.focus();
        }
        function close() {
            wrapper.classList.remove('open');
            trigger.setAttribute('aria-expanded', 'false');
        }
        function selectValue(value) {
            select.value = value;
            trigger.textContent = select.options[select.selectedIndex].textContent;
            select.dispatchEvent(new Event('change'));
            close();
        }
        function focusFirst() {
            const first = list.querySelector('.structure-option');
            if (first) {
                list.querySelectorAll('.focused').forEach(el => el.classList.remove('focused'));
                first.classList.add('focused');
            }
        }
        function move(delta) {
            const items = [...list.querySelectorAll('.structure-option')];
            if (!items.length) return;
            let idx = items.findIndex(i => i.classList.contains('focused'));
            if (idx === -1) idx = 0; else idx = (idx + delta + items.length) % items.length;
            items.forEach(i => i.classList.remove('focused'));
            items[idx].classList.add('focused');
            items[idx].scrollIntoView({ block: 'nearest' });
        }

        trigger.addEventListener('click', () => {
            if (trigger.classList.contains('disabled')) return; // ignore when disabled
            if (wrapper.classList.contains('open')) close(); else open();
        });
        searchInput.addEventListener('input', () => buildOptions(searchInput.value.trim()));
        list.addEventListener('click', e => {
            const li = e.target.closest('.structure-option');
            if (li) selectValue(li.dataset.value);
        });
        wrapper.addEventListener('keydown', e => {
            if (trigger.classList.contains('disabled')) return; // ignore keyboard when disabled
            if (!wrapper.classList.contains('open')) { if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } return; }
            switch (e.key) {
                case 'Escape': close(); trigger.focus(); break;
                case 'ArrowDown': e.preventDefault(); move(1); break;
                case 'ArrowUp': e.preventDefault(); move(-1); break;
                case 'Enter': case ' ': {
                    e.preventDefault();
                    const focused = list.querySelector('.focused');
                    if (focused) selectValue(focused.dataset.value);
                    break;
                }
                case 'Home': e.preventDefault(); focusFirst(); break;
            }
        });
        document.addEventListener('click', e => { if (!wrapper.contains(e.target)) close(); });

    // Do NOT auto-select first real option; keep placeholder until user chooses

    // Ensure disabled state reflects current sort mode after initialization
    toggleStructureControls();
    }
});