// Set maximum number of snowflakes:
const maxSnowflakes = 50;

// Function to create a snowflake:
function createSnowflakes() {
    const container = document.querySelector('.header');
    if (!container) return;

    // Check if there are already enough snowflakes:
    const currentSnowflakes = container.querySelectorAll('.snowflake').length;
    if (currentSnowflakes >= maxSnowflakes) return;

    // Create a new snowflake:
    const snowflake = document.createElement('img');
    snowflake.className = 'snowflake';
    snowflake.src = '/assets/snowflake.png';

    // Position and style:
    snowflake.style.left = `${50 + Math.random() * 50}%`;
    snowflake.style.width = `${Math.random() * 15 + 15}px`;
    snowflake.style.animationDuration = `${10 + Math.random() * 2}s`;

    // Add snowflake to container:
    container.appendChild(snowflake);

    // Remove snowflake when animation ends:
    snowflake.addEventListener('animationend', () => {
        snowflake.remove();
    });
}

// Function to start snowing with initial snowflakes:
function startSnowfall() {
    for (let i = 0; i < maxSnowflakes; i++) {
        createSnowflakes(); // Create initial snowflakes
    }
    setInterval(createSnowflakes, 5000); // Add one snowflake every 5 seconds
}

// Start snowfall:
startSnowfall();
