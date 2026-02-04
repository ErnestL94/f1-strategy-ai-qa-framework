// Load and display dashboard data
async function loadDashboardData() {
    try {
        const response = await fetch('data/results.json');
        const data = await response.json();

        updateSummary(data.summary);
        updateVersion(data.version, data.generated_at);
        updateTestSuites(data.test_suites);
        updateDatasets(data.golden_datasets);
        updateCapabilities(data.agent_capabilities);

    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showError();
    }
}

function updateSummary(summary) {
    document.getElementById('total-tests').textContent = `${summary.passing_tests}/${summary.total_tests}`;
    document.getElementById('coverage').textContent = `${summary.coverage_percent}%`;
    document.getElementById('scenarios').textContent = summary.total_scenarios;
    document.getElementById('accuracy').textContent = `${summary.overall_accuracy}%`;
}

function updateVersion(version, timestamp) {
    document.getElementById('version').textContent = `Agent ${version}`;

    const date = new Date(timestamp);
    const formatted = date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
    document.getElementById('timestamp').textContent = `Last updated: ${formatted}`;
}

function updateTestSuites(suites) {
    const container = document.getElementById('test-suites');
    container.innerHTML = '';

    Object.values(suites).forEach(suite => {
        const percentage = (suite.passing / suite.total) * 100;

        const card = document.createElement('div');
        card.className = 'suite-card';
        card.innerHTML = `
            <div class="suite-header">
                <span class="suite-name">${suite.name}</span>
                <span class="suite-status">${suite.passing}/${suite.total}</span>
            </div>
            <div class="suite-description">${suite.description}</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${percentage}%"></div>
            </div>
        `;
        container.appendChild(card);
    });
}

function updateDatasets(datasets) {
    const container = document.getElementById('datasets');
    container.innerHTML = '';

    datasets.forEach(dataset => {
        const card = document.createElement('div');
        card.className = 'dataset-card';
        card.innerHTML = `
            <div class="dataset-header">
                <span class="dataset-name">${dataset.name}</span>
                <span class="dataset-status">${dataset.accuracy}%</span>
            </div>
            <div class="dataset-info">
                ${dataset.scenarios} scenarios
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${dataset.accuracy}%"></div>
            </div>
        `;
        container.appendChild(card);
    });
}

function updateCapabilities(capabilities) {
    const container = document.getElementById('capabilities');
    container.innerHTML = '';

    capabilities.forEach(capability => {
        const item = document.createElement('div');
        item.className = 'capability-item';
        item.textContent = capability;
        container.appendChild(item);
    });
}

function showError() {
    document.body.innerHTML = `
        <div style="text-align: center; padding: 50px; color: white;">
            <h1>⚠️ Error Loading Dashboard</h1>
            <p>Run: python scripts/generate_dashboard_data.py</p>
        </div>
    `;
}

// Load data when page loads
loadDashboardData();

// Auto-refresh every 30 seconds (optional)
// setInterval(loadDashboardData, 30000);
