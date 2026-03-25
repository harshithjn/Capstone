// Dashboard JavaScript functions

function fillInputs() {
    let values = document.getElementById("bulkInput").value.split(",");
    let ids = [
        "workload_type", "workload_name", "workload_complexity", "cpu_cores",
        "memory_total_gb", "cpu_avg_pct", "effective_cpu", "memory_avg_gb",
        "memory_pressure", "disk_read_mb", "disk_write_mb", "io_intensity", "disk_speed_class"
    ];

    for(let i = 0; i < ids.length; i++) {
        if(values[i]) {
            document.getElementById(ids[i]).value = values[i].trim();
        }
    }
}

function showTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons in the same container
    event.target.parentElement.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabId).classList.add('active');
    event.target.classList.add('active');
}

function createSHAPChart(canvasId, data, title) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    if (!data || Object.keys(data).length === 0) {
        ctx.fillStyle = '#718096';
        ctx.font = '16px "Matter Regular", Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Feature contribution data not available', ctx.canvas.width/2, ctx.canvas.height/2);
        return;
    }

    const features = Object.keys(data);
    const values = Object.values(data);

    // Sort by absolute value for better visualization
    const sortedData = features.map((feature, i) => ({
        feature: feature.replace(/_/g, ' '),
        value: values[i]
    })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 8);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedData.map(d => d.feature),
            datasets: [{
                label: 'Feature Contribution',
                data: sortedData.map(d => d.value),
                backgroundColor: sortedData.map(d => d.value >= 0 ? '#48bb78' : '#f56565'),
                borderColor: sortedData.map(d => d.value >= 0 ? '#38a169' : '#e53e3e'),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    color: '#2d3748',
                    font: {
                        family: '"Matter Regular", Arial',
                        size: 14,
                        weight: '600'
                    }
                },
                legend: {
                    labels: {
                        color: '#4a5568',
                        font: {
                            family: '"Matter Regular", Arial'
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#4a5568',
                        font: {
                            family: '"Matter Regular", Arial'
                        }
                    },
                    grid: {
                        color: '#e2e8f0'
                    }
                },
                x: {
                    ticks: {
                        color: '#4a5568',
                        maxRotation: 45,
                        font: {
                            family: '"Matter Regular", Arial'
                        }
                    },
                    grid: {
                        color: '#e2e8f0'
                    }
                }
            }
        }
    });
}

function createImportanceChart(canvasId, data, title) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    if (!data || Object.keys(data).length === 0) {
        ctx.fillStyle = '#718096';
        ctx.font = '16px "Matter Regular", Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Feature importance data not available', ctx.canvas.width/2, ctx.canvas.height/2);
        return;
    }

    const features = Object.keys(data).map(f => f.replace(/_/g, ' '));
    const values = Object.values(data);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Importance Score',
                data: values,
                backgroundColor: '#2d2d2d',
                borderColor: '#1a1a1a',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    color: '#2d3748',
                    font: {
                        family: '"Matter Regular", Arial',
                        size: 14,
                        weight: '600'
                    }
                },
                legend: {
                    labels: {
                        color: '#4a5568',
                        font: {
                            family: '"Matter Regular", Arial'
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#4a5568',
                        font: {
                            family: '"Matter Regular", Arial'
                        }
                    },
                    grid: {
                        color: '#e2e8f0'
                    }
                },
                x: {
                    ticks: {
                        color: '#4a5568',
                        maxRotation: 45,
                        font: {
                            family: '"Matter Regular", Arial'
                        }
                    },
                    grid: {
                        color: '#e2e8f0'
                    }
                }
            }
        }
    });
}

// Initialize charts function
function initializeCharts(explanations) {
    if (!explanations || Object.keys(explanations).length === 0) return;
    
    // SHAP Charts
    createSHAPChart('shapChartLR', explanations.shap_values?.linear, 'Linear Regression Feature Contributions');
    createSHAPChart('shapChartDT', explanations.shap_values?.decision_tree, 'Decision Tree Feature Contributions');
    createSHAPChart('shapChartRF', explanations.shap_values?.random_forest, 'Random Forest Feature Contributions');
    createSHAPChart('shapChartXGB', explanations.shap_values?.xgboost, 'XGBoost Feature Contributions');

    // Importance Charts
    createImportanceChart('importanceChartDT', explanations.feature_importance?.decision_tree, 'Decision Tree Feature Importance');
    createImportanceChart('importanceChartRF', explanations.feature_importance?.random_forest, 'Random Forest Feature Importance');
    createImportanceChart('importanceChartXGB', explanations.feature_importance?.xgboost, 'XGBoost Feature Importance');
}

// Fetch explanations data from API
async function loadExplanations() {
    try {
        const response = await fetch('/api/explanations');
        const explanations = await response.json();
        initializeCharts(explanations);
    } catch (error) {
        console.log('No explanations data available yet');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Load explanations if available
    loadExplanations();
    
    // Check for explanations every 2 seconds if charts aren't loaded yet
    const checkInterval = setInterval(() => {
        if (document.querySelector('#shapChartLR canvas') && 
            !document.querySelector('#shapChartLR canvas').getContext('2d').canvas.width) {
            loadExplanations();
        } else {
            clearInterval(checkInterval);
        }
    }, 2000);
});

// SLA Threshold functions
function applySLAThreshold() {
    const threshold = parseFloat(document.getElementById('slaThresholdInput').value);
    const resultsDiv = document.getElementById('slaResults');

    if (isNaN(threshold) || threshold < 0) {
        resultsDiv.innerHTML = '<p style="color: #e53e3e;">Please enter a valid positive threshold value.</p>';
        return;
    }

    const predictions = window.modelPredictions;
    if (!predictions) {
        resultsDiv.innerHTML = '<p style="color: #718096;">No predictions available. Run a prediction first.</p>';
        return;
    }

    let html = `<p style="margin-bottom: 1rem; color: #4a5568;">Results against SLA threshold of <strong>${threshold} seconds</strong>:</p>`;
    html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">';

    for (const [model, value] of Object.entries(predictions)) {
        const breaches = value > threshold;
        const diff = (value - threshold).toFixed(2);
        const color = breaches ? '#e53e3e' : '#38a169';
        const bg = breaches ? '#fff5f5' : '#f0fff4';
        const border = breaches ? '#feb2b2' : '#9ae6b4';
        const label = breaches ? `+${diff}s over SLA` : `${Math.abs(diff)}s under SLA`;
        const icon = breaches ? '✗' : '✓';

        html += `
        <div style="background: ${bg}; border: 1px solid ${border}; border-radius: 10px; padding: 1rem; text-align: center;">
            <div style="font-weight: 600; color: #4a5568; margin-bottom: 0.5rem;">${model}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: ${color};">${icon} ${value}s</div>
            <div style="font-size: 0.8rem; color: ${color}; margin-top: 0.25rem;">${label}</div>
        </div>`;
    }

    html += '</div>';
    resultsDiv.innerHTML = html;
}

function clearSLAThreshold() {
    document.getElementById('slaThresholdInput').value = '';
    document.getElementById('slaResults').innerHTML = '';
}
