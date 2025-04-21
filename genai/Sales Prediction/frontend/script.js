document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const tv = parseFloat(document.getElementById('tv').value);
    const radio = parseFloat(document.getElementById('radio').value);
    const newspaper = parseFloat(document.getElementById('newspaper').value);

    try {
        const response = await fetch('http://localhost:8080/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ TV: tv, Radio: radio, Newspaper: newspaper }),
        });

        const result = await response.json();

        if (result.prediction) {
            document.getElementById('result').textContent = `${result.prediction.toFixed(2)} units`;
        } else {
            throw new Error(result.error || 'Unexpected error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        document.getElementById('result').textContent = 'An error occurred.';
    }
});

async function fetchAndRenderGraph() {
    try {
        const response = await fetch('http://localhost:8080/get-data');
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        const labels = data.map(item => `TV: ${item.TV}, Radio: ${item.Radio}, Newspaper: ${item.Newspaper}`);
        const sales = data.map(item => item.Sales);

        const ctx = document.getElementById('salesChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Actual Sales',
                    data: sales,
                    backgroundColor: 'rgba(0, 123, 255, 0.5)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 1,
                }]
            },
            options: {
                responsive: false, // Disable dynamic resizing
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Sales Prediction Graph',
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Advertising Budgets (TV, Radio, Newspaper)',
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45,
                        },
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Sales',
                        },
                        beginAtZero: true,
                    },
                },
            },
        });

    } catch (error) {
        console.error('Graph fetch error:', error);
        document.getElementById('salesChart').textContent = 'Error loading graph.';
    }
}

fetchAndRenderGraph();