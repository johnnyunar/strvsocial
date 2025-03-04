document.addEventListener('DOMContentLoaded', () => {
    themeSwitcherInit();
});

function themeSwitcherInit() {
    // Define available themes
    const themes = ["light", "dracula"];

    // Select the theme toggle input
    const themeToggle = document.querySelector('.theme-controller');
    const body = document.body;

    // Initialize theme from localStorage or default to the first theme
    const savedTheme = localStorage.getItem('theme') || themes[0];
    body.setAttribute('data-theme', savedTheme);

    // Set the initial state of the toggle based on the saved theme
    if (savedTheme === themes[1]) {
        themeToggle.checked = true;
    }

    // Add event listener to handle theme switching
    themeToggle.addEventListener('change', () => {
        // Determine the new theme based on toggle state
        const newTheme = themeToggle.checked ? themes[1] : themes[0];
        body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });
}