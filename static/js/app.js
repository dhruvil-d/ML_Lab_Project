document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navSearch = document.getElementById('nav-search');
    const navDash = document.getElementById('nav-dashboard');
    const secSearch = document.getElementById('section-search');
    const moviesSec = document.getElementById('movies-section');
    const secDash = document.getElementById('section-dashboard');

    navSearch.addEventListener('click', () => {
        navSearch.classList.add('active');
        navDash.classList.remove('active');
        secSearch.style.display = '';
        if (document.getElementById('movies-grid').childElementCount > 0) {
            moviesSec.style.display = '';
        }
        secDash.style.display = 'none';
    });

    navDash.addEventListener('click', () => {
        navDash.classList.add('active');
        navSearch.classList.remove('active');
        secSearch.style.display = 'none';
        moviesSec.style.display = 'none';
        secDash.style.display = '';
        loadDashboard();
    });

    // Search Elements
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const moviesGrid = document.getElementById('movies-grid');
    const resultsTitle = document.getElementById('results-title');
    const resultsCount = document.getElementById('results-count');
    const loadMoreBtn = document.getElementById('load-more-btn');
    const loadMoreContainer = document.getElementById('load-more-container');
    const loadingOverlay = document.getElementById('loading-overlay');

    let currentQuery = '';
    let currentPage = 1;
    let isLoading = false;

    // Search Actions
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });
    loadMoreBtn.addEventListener('click', loadMore);

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        currentQuery = query;
        currentPage = 1;
        moviesGrid.innerHTML = '';
        moviesSec.style.display = '';
        
        await fetchResults();
    }

    async function loadMore() {
        currentPage++;
        await fetchResults();
    }

    async function fetchResults() {
        if (isLoading) return;
        isLoading = true;
        
        // Show loading state
        if (currentPage === 1) {
            loadingOverlay.style.display = 'flex';
        } else {
            const btnOriginal = loadMoreBtn.textContent;
            loadMoreBtn.innerHTML = '<div class="spinner" style="margin: 0 auto;"></div>';
            loadMoreBtn.disabled = true;
        }

        try {
            const response = await fetch(`/api/search?q=${encodeURIComponent(currentQuery)}&page=${currentPage}`);
            const data = await response.json();

            // API Error handling loop
            if (data.error) {
                if (data.error.includes("Request limit reached")) {
                    alert('OMDB API Request Limit Reached!\nPlease try again tomorrow, or deploy with a premium key.');
                } else if (data.error.includes("Invalid API key")) {
                    alert('Invalid OMDB API Key. Please check your configuration.');
                } else {
                    alert(`Error: ${data.error}`);
                }
                moviesSec.style.display = 'none';
                return;
            }

            if (data.error) throw new Error(data.error);

            const movies = data.movies || [];
            
            if (currentPage === 1) {
                resultsTitle.textContent = `Results for "${currentQuery}"`;
                resultsCount.textContent = `${data.totalResults || 0} movies found`;
                
                if (movies.length === 0) {
                    moviesGrid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--text-muted);">No movies found for "${currentQuery}"</div>`;
                }
            }

            movies.forEach(movie => {
                moviesGrid.appendChild(createMovieCard(movie));
            });

            // Handle load more visibility
            if (movies.length === 10 && (currentPage * 10) < (data.totalResults || 0)) {
                loadMoreContainer.style.display = 'block';
            } else {
                loadMoreContainer.style.display = 'none';
            }

        } catch (error) {
            console.error('Search error:', error);
            if (currentPage === 1) {
                moviesGrid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--accent-red);">Failed to fetch results</div>`;
            }
        } finally {
            isLoading = false;
            loadingOverlay.style.display = 'none';
            if (currentPage > 1) {
                loadMoreBtn.textContent = 'Load More';
                loadMoreBtn.disabled = false;
            }
        }
    }

    function createMovieCard(movie) {
        const card = document.createElement('div');
        card.className = 'movie-card';
        
        const poster = movie.poster && movie.poster !== 'N/A' 
            ? `<img src="${movie.poster}" alt="${movie.title} pointer">` 
            : `<div class="no-poster">🎬</div>`;

        card.innerHTML = `
            ${poster}
            <div class="movie-card-info">
                <div class="movie-card-title" title="${movie.title}">${movie.title}</div>
                <div class="movie-card-year">${movie.year}</div>
            </div>
        `;
        
        card.addEventListener('click', () => openMovieModal(movie.imdbID));
        return card;
    }

    // ============ MODAL ============
    const modalOverlay = document.getElementById('modal-overlay');
    const modalClose = document.getElementById('modal-close');
    const wordInput = document.getElementById('word-input');
    const wordBtn = document.getElementById('word-btn');
    const wordResults = document.getElementById('word-results');
    
    let currentMoviePlot = '';

    modalClose.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });

    function closeModal() {
        modalOverlay.style.display = 'none';
        currentMoviePlot = '';
        wordResults.style.display = 'none';
        wordInput.value = '';
    }

    async function openMovieModal(imdbID) {
        modalOverlay.style.display = '';
        
        // Reset cards to loading state
        ['card-nb', 'card-lr', 'card-svm'].forEach(id => {
            document.getElementById(id).innerHTML = '<div class="spinner"></div>';
            document.getElementById(id).className = 'model-card loading-card';
        });
        wordResults.style.display = 'none';
        wordInput.value = '';
        
        try {
            const resp = await fetch(`/api/movie/${imdbID}`);
            const m = await resp.json();
            
            if (m.error) {
                alert('Movie not found');
                closeModal();
                return;
            }
            
            // Populate modal
            const posterEl = document.getElementById('modal-poster');
            if (m.poster && m.poster !== 'N/A') {
                posterEl.src = m.poster;
                posterEl.style.display = '';
            } else {
                posterEl.style.display = 'none';
            }
            
            document.getElementById('modal-title').textContent = m.title || '';
            document.getElementById('modal-rating').textContent = `⭐ ${m.rating || '—'}`;
            document.getElementById('modal-year').textContent = `📅 ${m.year || '—'}`;
            document.getElementById('modal-runtime').textContent = `⏱️ ${m.runtime || '—'}`;
            document.getElementById('modal-director').textContent = m.director ? `🎬 Director: ${m.director}` : '';
            document.getElementById('modal-actors').textContent = m.actors ? `👥 Cast: ${m.actors}` : '';
            document.getElementById('modal-plot').textContent = m.plot || 'No plot available.';
            
            // Genre tags
            const genreTags = document.getElementById('modal-genre-tags');
            genreTags.innerHTML = '';
            if (m.genre) {
                m.genre.split(',').forEach(g => {
                    const tag = document.createElement('span');
                    tag.className = 'genre-tag';
                    tag.textContent = g.trim();
                    genreTags.appendChild(tag);
                });
            }
            
            currentMoviePlot = m.plot || '';
            
            // Analyze with models
            if (currentMoviePlot) {
                analyzeWithModels(currentMoviePlot);
            }
            
            loadSimilarMovies(m.imdbID);
            
        } catch (err) {
            console.error('Modal error:', err);
            alert('Error loading movie details');
            closeModal();
        }
    }

    async function loadSimilarMovies(imdbID) {
        const listEl = document.getElementById('similar-movies-list');
        const secEl = document.getElementById('similar-movies-section');
        listEl.innerHTML = '<div class="spinner" style="margin: 20px auto;"></div>';
        secEl.style.display = '';
        
        try {
            const resp = await fetch(`/api/similar/${imdbID}`);
            const data = await resp.json();
            
            if (data.error || !data.similar_movies || data.similar_movies.length === 0) {
                secEl.style.display = 'none';
                return;
            }
            
            listEl.innerHTML = '';
            data.similar_movies.forEach(m => {
                const card = document.createElement('div');
                card.className = 'similar-movie-card';
                const poster = m.poster && m.poster !== 'N/A' 
                    ? `<img src="${m.poster}" alt="${m.title}">` 
                    : `<div class="no-poster" style="height:160px;font-size:24px;">🎬</div>`;
                card.innerHTML = `
                    ${poster}
                    <div class="similar-movie-info">
                        <div class="similar-movie-title" title="${m.title}">${m.title}</div>
                        <div class="similar-movie-score">${m.similarity}% Match</div>
                    </div>
                `;
                // Clicking similar movie opens the modal for that movie
                card.addEventListener('click', () => {
                    openMovieModal(m.imdbID);
                });
                listEl.appendChild(card);
            });
            
        } catch (e) {
            console.error('Error fetching similar movies', e);
            secEl.style.display = 'none';
        }
    }

    async function analyzeWithModels(plot) {
        try {
            const resp = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ plot })
            });
            const data = await resp.json();
            
            if (data.error) {
                ['card-nb', 'card-lr', 'card-svm'].forEach(id => {
                    document.getElementById(id).innerHTML = `<p style="color: var(--text-muted); font-size: 13px;">Models not loaded. Run train_models.py first.</p>`;
                    document.getElementById(id).className = 'model-card';
                });
                return;
            }
            
            const colors = {
                'Naive Bayes': 'var(--accent-blue)',
                'Logistic Regression': 'var(--accent-purple)',
                'SVM': 'var(--accent-green)'
            };
            const ids = {
                'Naive Bayes': 'card-nb',
                'Logistic Regression': 'card-lr',
                'SVM': 'card-svm'
            };
            
            for (const [name, result] of Object.entries(data)) {
                const card = document.getElementById(ids[name]);
                card.className = 'model-card';
                card.innerHTML = `
                    <div class="model-card-name">${name}</div>
                    <div class="model-card-pred">${result.prediction}</div>
                    <div class="model-card-conf">${result.confidence}% confidence</div>
                    <div class="conf-bar">
                        <div class="conf-bar-fill" style="width: 0%; background: ${colors[name]};"></div>
                    </div>
                `;
                // Animate bar
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        card.querySelector('.conf-bar-fill').style.width = `${result.confidence}%`;
                    });
                });
            }
            
        } catch (err) {
            console.error('Analysis error:', err);
        }
    }

    // ============ WORD RELEVANCE ============
    wordBtn.addEventListener('click', checkWordRelevance);
    wordInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') checkWordRelevance();
    });

    async function checkWordRelevance() {
        const word = wordInput.value.trim();
        if (!word || !currentMoviePlot) return;
        
        wordBtn.disabled = true;
        wordBtn.textContent = '...';
        
        try {
            const resp = await fetch('/api/word-relevance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word, plot: currentMoviePlot })
            });
            const data = await resp.json();
            
            renderWordResults(data, word);
            
        } catch (err) {
            console.error('Word relevance err:', err);
            wordResults.innerHTML = `<div class="word-not-found">Error checking relevance.</div>`;
            wordResults.style.display = 'block';
        } finally {
            wordBtn.disabled = false;
            wordBtn.textContent = 'Check';
        }
    }

    function renderWordResults(data, requestedWord) {
        wordResults.style.display = 'block';
        
        if (data.error) {
            wordResults.innerHTML = `<div class="word-not-found">${data.error}</div>`;
            return;
        }
        
        const score = data.score || 0;
        let modelScoresHTML = '';
        if (data.models) {
            for (const [mName, mScore] of Object.entries(data.models)) {
                modelScoresHTML += `
                    <div class="model-score-item">
                        <div class="model-score-label">${mName}</div>
                        <div class="model-score-val" style="color:var(--accent-blue);">${mScore}</div>
                    </div>
                `;
            }
        }
        
        let headerColor = 'var(--text-primary)';
        if (score === 0) headerColor = 'var(--text-muted)';
        else if (score > 80) headerColor = 'var(--accent-green)';
        else if (score > 50) headerColor = 'var(--accent-yellow)';
        else headerColor = 'var(--accent-orange)';
        
        wordResults.innerHTML = `
            <div class="relevance-header">
                <div>
                    <span style="color:var(--text-muted);">Relevance for:</span>
                    <span class="relevance-word" style="color:${headerColor}; margin-left:6px;">"${data.word}"</span>
                </div>
                <div class="relevance-score-big" style="color:${headerColor};">${score}%</div>
            </div>
            
            ${data.message ? `<div style="font-size:13px; color:var(--text-muted); margin-bottom:12px; font-style:italic;">${data.message}</div>` : ''}
            
            <div class="relevance-bar">
                <div class="relevance-bar-fill" style="width: 0%; background: ${headerColor};"></div>
            </div>
            
            ${modelScoresHTML ? `
                <div style="font-size:13px;color:var(--text-muted);margin-bottom:10px;">Model Similarity Confidence:</div>
                <div class="model-scores">${modelScoresHTML}</div>
            ` : ''}
        `;
        
        // Animate bar
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const fill = wordResults.querySelector('.relevance-bar-fill');
                if (fill) fill.style.width = `${score}%`;
            });
        });
        
    }

    // ============ DASHBOARD ============
    let dashboardLoaded = false;

    async function loadDashboard() {
        if (dashboardLoaded) return;
        
        try {
            const resp = await fetch('/api/metrics');
            const data = await resp.json();
            
            if (data.error) {
                document.getElementById('dashboard-grid').innerHTML = `
                    <div class="dash-card dash-card-wide">
                        <p style="color: var(--text-muted); text-align: center; padding: 40px;">
                            Models not trained yet. Run <code>python train_models.py</code> first.
                        </p>
                    </div>
                `;
                return;
            }
            
            const metrics = data.metrics;
            const models = Object.keys(metrics);
            const colors = ['rgba(59,130,246,0.8)', 'rgba(139,92,246,0.8)', 'rgba(34,197,94,0.8)'];
            const borderColors = ['#3b82f6', '#8b5cf6', '#22c55e'];
            
            // Chart.js defaults for dark theme
            Chart.defaults.color = '#94a3b8';
            Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
            
            const chartOptions = (label) => ({
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { backgroundColor: '#1e293b', padding: 12 }
                },
                scales: {
                    y: {
                        beginAtZero: true, max: 1,
                        ticks: { callback: v => `${(v * 100).toFixed(0)}%` },
                        grid: { color: 'rgba(255,255,255,0.04)' }
                    },
                    x: { grid: { display: false } }
                },
                animation: { duration: 1200, easing: 'easeOutQuart' }
            });
            
            // Create charts
            ['accuracy', 'precision', 'recall', 'f1'].forEach(metric => {
                const ctx = document.getElementById(`chart-${metric}`).getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: models,
                        datasets: [{
                            data: models.map(m => metrics[m][metric]),
                            backgroundColor: colors,
                            borderColor: borderColors,
                            borderWidth: 2,
                            borderRadius: 8,
                            barPercentage: 0.6,
                        }]
                    },
                    options: chartOptions(metric)
                });
            });
            
            // Params table
            const tableContainer = document.getElementById('params-table');
            let tableHTML = `
                <table class="params-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Best Hyperparameters</th>
                            <th>CV Accuracy (5-fold)</th>
                            <th>Test Accuracy</th>
                            <th>Test F1 Score</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            models.forEach(name => {
                const m = metrics[name];
                const params = Object.entries(m.best_params || {}).map(([k, v]) => `${k}=${v}`).join(', ');
                tableHTML += `
                    <tr>
                        <td style="font-weight:600;">${name}</td>
                        <td><code style="color:var(--accent-purple);background:rgba(139,92,246,0.1);padding:2px 8px;border-radius:4px;">${params}</code></td>
                        <td>${(m.cv_accuracy * 100).toFixed(2)}%</td>
                        <td>${(m.accuracy * 100).toFixed(2)}%</td>
                        <td>${(m.f1 * 100).toFixed(2)}%</td>
                    </tr>
                `;
            });
            tableHTML += '</tbody></table>';
            tableContainer.innerHTML = tableHTML;
            
            // --- EDA VISUALIZATIONS ---
            const edaResp = await fetch('/api/eda');
            const edaData = await edaResp.json();
            
            if (!edaData.error) {
                // Genre Distribution Donut Chart
                const gDist = edaData.genre_distribution || {};
                const gLabels = Object.keys(gDist).slice(0, 10); // top 10
                const gValues = gLabels.map(g => gDist[g]);
                const gColors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#3b82f6', '#8b5cf6', '#d946ef', '#f43f5e', '#64748b'];
                
                const ctxG = document.getElementById('chart-genres').getContext('2d');
                new Chart(ctxG, {
                    type: 'doughnut',
                    data: {
                        labels: gLabels,
                        datasets: [{
                            data: gValues,
                            backgroundColor: gColors,
                            borderWidth: 0,
                            hoverOffset: 10
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'right', labels: { color: '#f1f5f9' } },
                            tooltip: { backgroundColor: '#1e293b', padding: 12 }
                        },
                        cutout: '65%'
                    }
                });
                
                // Top Feature Words by Genre
                const topWordsMap = edaData.top_words || {};
                const selectorDiv = document.getElementById('top-words-selector');
                const cloudContainer = document.getElementById('word-cloud-container');
                
                let currentActivePill = null;
                
                function renderWordCloud(genre) {
                    if (currentActivePill) currentActivePill.classList.remove('active');
                    const newPill = document.getElementById(`pill-${genre.replace(/\\W+/g, '')}`);
                    if (newPill) {
                        newPill.classList.add('active');
                        currentActivePill = newPill;
                    }
                    
                    cloudContainer.innerHTML = '';
                    const words = topWordsMap[genre] || [];
                    
                    if (words.length === 0) {
                        cloudContainer.innerHTML = '<p style="color:var(--text-muted);">No top words available.</p>';
                        return;
                    }
                    
                    // Max value to normalize sizing
                    const maxVal = Math.max(...words.map(w => w.value));
                    
                    words.forEach((w, idx) => {
                        const tag = document.createElement('div');
                        tag.className = 'word-tag';
                        
                        // Sizing mapping based on value (relative to maxVal)
                        // max font 32px, min font 12px
                        const ratio = w.value / maxVal;
                        const fontSize = Math.max(12, Math.floor(ratio * 36)); 
                        
                        // Coloring mapping
                        // Random color from the palette
                        const color = gColors[idx % gColors.length];
                        
                        tag.style.fontSize = `${fontSize}px`;
                        tag.style.color = color;
                        tag.style.fontWeight = '700';
                        tag.style.animationDelay = `${idx * 0.05}s`;
                        tag.textContent = w.text;
                        tag.title = `TF-IDF Score: ${w.value.toFixed(4)}`;
                        
                        cloudContainer.appendChild(tag);
                    });
                }
                
                // Create selector pills
                Object.keys(topWordsMap).forEach(g => {
                    const pill = document.createElement('div');
                    pill.className = 'genre-pill';
                    pill.id = `pill-${g.replace(/\\W+/g, '')}`;
                    pill.textContent = g;
                    pill.addEventListener('click', () => renderWordCloud(g));
                    selectorDiv.appendChild(pill);
                });
                
                // Select first genre by default
                if (Object.keys(topWordsMap).length > 0) {
                    renderWordCloud(Object.keys(topWordsMap)[0]);
                }
            }
            
            dashboardLoaded = true;
            
        } catch (err) {
            console.error('Dashboard error:', err);
        }
    }
});
